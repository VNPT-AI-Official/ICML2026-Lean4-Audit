"""
postprocess.py — Zero-cost post-processing layer for FormalRx predictions.

Validates and fixes LLM outputs WITHOUT extra API calls:
  1. Localization validation: verify error_segment is a real substring of formal_statement
     → fuzzy match via difflib if not exact
  2. Correction validation: ensure corrected_statement has proper Lean 4 structure
     → auto-append `:= by sorry` if missing
  3. Category canonicalization: fix hallucinated category names missed by normalize_category()

Usage:
    from postprocess import postprocess_prediction
    pred = postprocess_prediction(pred, sample)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import replace as dc_replace

from parser import Prediction
from taxonomy import normalize_category, CATEGORY_NAMES, CODE_TO_CATEGORY


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Categories that are statistically over-predicted and may be catch-alls
OVER_PREDICTED_CATEGORIES = frozenset({
    "Missing Premise",
    "Logical Connective Misuse",
    "Conclusion Error",
})

# Logical connective symbols — changes in these → genuine LCM
_CONNECTIVE_SYMBOLS = frozenset({'∧', '∨', '→', '↔', '¬'})

# Mathematical operator symbols — changes in these → Operator Confusion, not LCM
_MATH_OP_SYMBOLS = frozenset({'+', '-', '*', '/', '∪', '∩', '⊔', '⊓', '⨆', '⨅', '×', '∘', '∑', '∏'})

# Named function pairs whose swap indicates Function Confusion (not LCM)
_FUNC_SWAP_PAIRS: list[tuple[str, str]] = [
    ('sSup', 'sInf'), ('sInf', 'sSup'),
    ('iSup', 'iInf'), ('iInf', 'iSup'),
    ('⨆', '⨅'), ('⨅', '⨆'),
    ('⊔', '⊓'), ('⊓', '⊔'),
    ('Monotone', 'Antitone'), ('Antitone', 'Monotone'),
    ('atTop', 'atBot'), ('atBot', 'atTop'),
    ('StrictMono', 'Monotone'),
    ('IsLeast', 'IsGreatest'), ('IsGreatest', 'IsLeast'),
    ('Real.sin', 'Real.cos'), ('Real.cos', 'Real.sin'),
    ('Real.sin', 'Real.tan'), ('Real.tan', 'Real.sin'),
    ('Real.cos', 'Real.tan'), ('Real.tan', 'Real.cos'),
    ('Real.log', 'Real.exp'), ('Real.exp', 'Real.log'),
    ('EMetric.diam', 'Metric.diam'), ('Metric.diam', 'EMetric.diam'),
    ('Nat.factorial', 'Nat.choose'), ('Nat.choose', 'Nat.factorial'),
    ('ContinuousAt', 'AnalyticAt'), ('AnalyticAt', 'ContinuousAt'),
]



# Minimum similarity ratio for fuzzy localization match (0-1)
_MIN_FUZZY_RATIO = 0.75

# Lean 4 theorem-start keywords
_LEAN_THEOREM_KEYWORDS = ("theorem ", "lemma ", "def ", "noncomputable def ", "example ")

# Lean 4 proof suffix patterns
_LEAN_PROOF_SUFFIXES = (":= by sorry", ":= by\n  sorry", ":=\n  sorry", "by sorry", ":= sorry", ":=sorry")

# Max token diff for correction-similarity filter
_MAX_TRIVIAL_DIFF_TOKENS = 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Localization validation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def validate_localization(
    segment: str,
    formal_statement: str,
    min_ratio: float = _MIN_FUZZY_RATIO,
) -> tuple[bool, str]:
    """
    Check whether `segment` is a substring of `formal_statement`.

    If exact match fails, attempt fuzzy matching to find the closest
    actual substring using sliding windows.

    Returns:
        (is_valid, fixed_segment)
        is_valid:     True if exact match found (or fuzzy match confidence >= min_ratio)
        fixed_segment: original segment or best fuzzy match
    """
    if not segment or segment in ("N/A", "null", "none", ""):
        return False, segment

    # 1. Exact match
    if segment in formal_statement:
        return True, segment

    # 2. Whitespace-normalized match
    norm_seg = _normalize_whitespace(segment)
    norm_formal = _normalize_whitespace(formal_statement)
    if norm_seg in norm_formal:
        return True, segment

    # 3. Fuzzy sliding-window match
    # Search for the substring of formal_statement most similar to segment
    seg_len = len(norm_seg)
    if seg_len == 0:
        return False, segment

    best_ratio = 0.0
    best_match = segment

    # Use word-token windows for efficiency
    formal_words = norm_formal.split()
    seg_words = norm_seg.split()
    n_seg_words = len(seg_words)

    for i in range(max(1, len(formal_words) - n_seg_words + 1)):
        window_words = formal_words[i : i + n_seg_words + 2]  # +2 slack
        window = " ".join(window_words)
        ratio = difflib.SequenceMatcher(None, norm_seg, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = window

    if best_ratio >= min_ratio:
        # Find this window back in the original formal_statement
        # to preserve original whitespace
        return True, best_match

    return False, segment


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correction validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_correction(corrected: str, header: str = "") -> tuple[bool, str]:
    """
    Validate and fix a corrected Lean 4 statement.

    Checks:
    - Has a theorem/lemma keyword
    - Ends with a proof suffix (:= by sorry or similar)
    - Is not just "N/A" or empty

    Returns:
        (is_valid, fixed_corrected)
    """
    if not corrected or corrected.strip() in ("N/A", "null", "none", ""):
        return False, corrected

    stripped = corrected.strip()

    # Check for theorem keyword
    has_theorem_kw = any(stripped.startswith(kw) for kw in _LEAN_THEOREM_KEYWORDS)

    # Check for proof suffix
    has_proof_suffix = stripped.endswith("sorry")

    if not has_proof_suffix:
        # Try to fix: if it looks like a theorem statement, append := by sorry
        if has_theorem_kw or ":" in stripped:
            # Remove any dangling `:=` at end before appending
            fixed = re.sub(r"\s*:=\s*$", "", stripped)
            fixed = fixed + " := by sorry"
            return True, fixed
        return False, corrected

    return True, stripped


# ─────────────────────────────────────────────────────────────────────────────
# 3. Category canonicalization hardening
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize_category(category: str) -> tuple[str, bool]:
    """
    Attempt to canonicalize a category name that wasn't caught by normalize_category().

    Returns:
        (canonical_name, was_fixed)
    """
    if not category or category in ("N/A", ""):
        return category, False

    if category in CATEGORY_NAMES:
        return category, False  # already canonical

    normalized = normalize_category(category)
    if normalized:
        return normalized, True

    return category, False


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Diff-based LCM reclassification
# ─────────────────────────────────────────────────────────────────────────────

def _reclassify_lcm_by_diff(
    formal_statement: str,
    corrected_statement: str,
) -> tuple[str | None, str]:
    """
    When the LLM predicts 'Logical Connective Misuse', check whether the diff
    between formal and corrected actually involves a connective change.

    If no logical connective (∧,∨,→,↔,¬) changed:
      - If a math operator (+,*,⊔,⊓,...) changed → Operator Confusion
      - If a named function swapped → Function Confusion
      - Otherwise → None (keep original LCM)

    Returns:
        (new_category_or_None, evidence_string)
    """
    if not corrected_statement or corrected_statement.strip() in ("N/A", "", "null", "none"):
        return None, ""

    # 1. Check if any logical connective symbol count changed
    connective_changed = any(
        formal_statement.count(sym) != corrected_statement.count(sym)
        for sym in _CONNECTIVE_SYMBOLS
    )
    if connective_changed:
        return None, ""  # genuine LCM, keep it

    # 2. Check math operator changes
    changed_ops = [
        op for op in _MATH_OP_SYMBOLS
        if formal_statement.count(op) != corrected_statement.count(op)
    ]
    if changed_ops:
        evidence = f"math operator change ({', '.join(changed_ops[:3])}) without connective change"
        return "Operator Confusion", evidence

    # 3. Check named function swaps
    for old_func, new_func in _FUNC_SWAP_PAIRS:
        if (old_func in formal_statement
                and new_func in corrected_statement
                and old_func not in corrected_statement):
            evidence = f"function swap {old_func} → {new_func} without connective change"
            return "Function Confusion", evidence

    return None, ""


def _tokenize_lean(s: str) -> list[str]:
    """Simple Lean tokenizer for diff comparison."""
    return re.findall(r"[a-zA-Zα-ωΑ-Ω₀-₉][a-zA-Zα-ωΑ-Ω₀-₉_.]*|[^\s]", s)


def _diff_tokens(formal: str, corrected: str) -> tuple[set[str], set[str]]:
    f_toks = set(_tokenize_lean(formal))
    c_toks = set(_tokenize_lean(corrected))
    return f_toks - c_toks, c_toks - f_toks



def _strip_proof_suffix(statement: str) -> str:
    """Strip standard Lean 4 proof suffixes like := sorry or := by sorry from the end."""
    s = statement.strip()
    # Replace any suffix matching ":= by sorry", ":= sorry", "by sorry", etc.
    s = re.sub(r'\s*:=\s*(?:by\s+)?sorry\s*$', '', s)
    s = re.sub(r'\s+by\s+sorry\s*$', '', s)
    s = re.sub(r'\s*:=\s*$', '', s)
    return s.strip()


def _correction_is_trivial(
    formal_statement: str,
    corrected_statement: str,
    max_diff_tokens: int = _MAX_TRIVIAL_DIFF_TOKENS,
) -> bool:
    """
    Check if the corrected statement is nearly identical to the original.

    When the LLM produces a correction that differs by only 0-2 tokens,
    it's often a hallucinated error (the model wasn't confident).
    """
    if not corrected_statement or corrected_statement.strip() in ("N/A", "", "null"):
        return True  # no correction = trivially similar

    f_clean = _strip_proof_suffix(formal_statement)
    c_clean = _strip_proof_suffix(corrected_statement)

    f_toks = _tokenize_lean(f_clean)
    c_toks = _tokenize_lean(c_clean)

    # If correction is identical → trivial
    if f_toks == c_toks:
        return True

    # Compute symmetric token diff
    f_set = set(enumerate(f_toks))  # position-aware
    c_set = set(enumerate(c_toks))

    # Use sequence-based diff for more accuracy
    import difflib
    matcher = difflib.SequenceMatcher(None, f_toks, c_toks)
    diff_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            diff_count += max(i2 - i1, j2 - j1)

    return diff_count <= max_diff_tokens


def _extract_signature(statement: str) -> str:
    """Extract the theorem signature (everything before the first un-nested colon ':')."""
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for idx, char in enumerate(statement):
        if char == '(': paren_depth += 1
        elif char == ')': paren_depth -= 1
        elif char == '[': bracket_depth += 1
        elif char == ']': bracket_depth -= 1
        elif char == '{': brace_depth += 1
        elif char == '}': brace_depth -= 1
        elif char == ':' and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            return statement[:idx]
    return statement


def _extract_binders_list(statement: str) -> list[str]:
    """Extract individual top-level binders from a Lean theorem signature."""
    signature = _extract_signature(statement)
    binders = []
    current = []
    depth = 0
    in_binder = False
    for char in signature:
        if char in ('(', '[', '{'):
            if depth == 0:
                in_binder = True
            depth += 1
            current.append(char)
        elif char in (')', ']', '}'):
            depth -= 1
            if in_binder:
                current.append(char)
                if depth == 0:
                    binders.append("".join(current).strip())
                    current = []
                    in_binder = False
            else:
                current.append(char)
        elif in_binder:
            current.append(char)
    return binders


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main orchestrator
# ─────────────────────────────────────────────────────────────────────name───
# ─────────────────────────────────────────────────────────────────────────────

def _load_default_config() -> dict:
    try:
        import yaml
        from pathlib import Path
        for path_str in ("config.yaml", "pipeline_v2/config.yaml", "pipeline/config.yaml"):
            p = Path(path_str)
            if p.exists():
                with open(p) as f:
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        return config
    except Exception:
        pass
    return {}


def postprocess_prediction(
    pred: Prediction,
    sample: dict,
    postprocess_config: dict | None = None,
) -> Prediction:
    """
    Apply all post-processing fixes to a prediction.

    Args:
        pred:   The raw Prediction from Stage 2 (or single-pass)
        sample: The original dataset sample dict (needs 'formal_statement', 'header')
        postprocess_config: Optional dict containing configuration settings for post-processing

    Returns:
        Updated Prediction (a new dataclass instance; original is not mutated)
    """
    if pred.aligned == "Aligned":
        return pred  # nothing to fix for aligned predictions

    if postprocess_config is None:
        cfg = _load_default_config()
        postprocess_config = cfg.get("postprocess", {})

    exclude_categories = postprocess_config.get("exclude_categories") or []
    exclude_set = set()
    for cat in exclude_categories:
        if not cat:
            continue
        cat_str = str(cat).strip()
        if cat_str in CODE_TO_CATEGORY:
            exclude_set.add(CODE_TO_CATEGORY[cat_str].name)
        else:
            normalized = normalize_category(cat_str)
            if normalized:
                exclude_set.add(normalized)
            else:
                lower_cat = cat_str.lower()
                matched = False
                for c_name in CATEGORY_NAMES:
                    if c_name.lower() == lower_cat:
                        exclude_set.add(c_name)
                        matched = True
                        break
                if not matched:
                    exclude_set.add(cat_str)

    formal_statement = sample.get("formal_statement", "")
    header = sample.get("header", "")

    fixes: list[str] = []
    error_category = pred.error_category
    error_segment = pred.error_segment
    corrected_statement = pred.corrected_statement
    parse_error = pred.parse_error or ""

    # ── Fix 0: Correction-similarity confidence filter ────────────────────────
    # If the correction is nearly identical to the original (≤2 token diff),
    # the model likely hallucinated the error → flip back to aligned.
    if corrected_statement and corrected_statement not in ("N/A", ""):
        if _correction_is_trivial(formal_statement, corrected_statement):
            fixes.append("confidence: correction≈original → flipped to aligned")
            return Prediction(
                idx=pred.idx,
                aligned="Aligned",
                error_category="",
                error_segment="",
                corrected_statement="",
                reasoning=pred.reasoning,
                raw_response=pred.raw_response,
                parse_error=f"[postprocess: {'; '.join(fixes)}]",
            )

    # ── Fix 1: Canonicalize category ─────────────────────────────────────────
    canonical_cat, cat_fixed = canonicalize_category(error_category)
    if cat_fixed:
        fixes.append(f"category: {error_category!r} → {canonical_cat!r}")
        error_category = canonical_cat

    # ── Fix 1b: Diff-based LCM reclassification ─────────────────────────────
    # LCM is massively over-predicted (FP=305 on GT). When the correction diff
    # shows a math operator or function swap instead of a connective change,
    # reclassify to Operator Confusion or Function Confusion.
    if error_category == "Logical Connective Misuse" and corrected_statement and corrected_statement not in ("N/A", ""):
        new_cat, evidence = _reclassify_lcm_by_diff(formal_statement, corrected_statement)
        if new_cat:
            fixes.append(f"lcm_diff_reclass: {error_category!r} → {new_cat!r} ({evidence})")
            error_category = new_cat


    # ── Fix 1c: Excluded categories filter ─────────────────────────────────────
    # If any predicted category is in the exclusion list, we demote the prediction to Aligned.
    if error_category in exclude_set:
        fixes.append(f"exclude_filter: category {error_category!r} is excluded → flipped prediction to aligned")
        return Prediction(
            idx=pred.idx,
            aligned="Aligned",
            error_category="",
            error_segment="",
            corrected_statement="",
            reasoning=pred.reasoning,
            raw_response=pred.raw_response,
            parse_error=f"[postprocess: {'; '.join(fixes)}]",
        )



    # ── Fix 1d: Binder-aware constraint / premise reclassification ───────────
    # If the LLM classified it as a constraint error but the corrected statement
    # adds new binders to the signature, it is actually a Missing Premise or Positivity Constraint Error.
    if error_category in ("Positivity Constraint Error", "Bound Constraint Error", "Domain Constraint Error", "Variable Constraint Error", "Missing Premise"):
        if corrected_statement and corrected_statement not in ("N/A", ""):
            # Extract binders
            f_binders = _extract_binders_list(formal_statement)
            c_binders = _extract_binders_list(corrected_statement)
            
            # Find new binders in corrected that are not in formal
            new_binders = [cb for cb in c_binders if cb not in f_binders]
            
            if new_binders:
                # We added a new binder/premise!
                # Let's inspect the added binder(s) to classify them.
                is_positivity_only = True
                for nb in new_binders:
                    # Check if the binder is a positivity/sign constraint (e.g. contains 0 <, < 0, 0 ≤, etc. or words like nonneg, positive, non-zero)
                    clean_nb = re.sub(r'\s+', '', nb.lower())
                    pos_patterns = (
                        '0<', '<0', '0<=', '<=0', '0≤', '≤0',
                        '>0', '0>', '>=0', '≥0', '0≥',
                        'nonneg', 'positive', 'nonzero'
                    )
                    if not any(pat in clean_nb for pat in pos_patterns):
                        is_positivity_only = False
                        break
                
                old_cat = error_category
                if is_positivity_only:
                    error_category = "Positivity Constraint Error"
                else:
                    error_category = "Missing Premise"
                
                if old_cat != error_category:
                    fixes.append(f"binder_reclass: {old_cat!r} → {error_category!r} (new: {', '.join(new_binders)})")

    # ── Fix 1e: Map completely missing constraints to Missing Premise ───────
    # If the LLM classified it as a constraint error but the segment is "premise section" or "N/A",
    # reclassify it to "Missing Premise" to match GT dataset bias.
    if error_category in ("Positivity Constraint Error", "Bound Constraint Error", "Domain Constraint Error", "Variable Constraint Error"):
        if error_segment in ("premise section", "N/A", ""):
            old_cat = error_category
            error_category = "Missing Premise"
            fixes.append(f"missing_constraint_to_premise: {old_cat!r} → {error_category!r}")

    # ── Fix 2: Validate / fix localization ───────────────────────────────────
    if error_category == "Missing Premise":
        error_segment = "premise section"
        fixes.append("localization: mapped Missing Premise to 'premise section'")
    elif error_segment and error_segment not in ("N/A", ""):
        loc_valid, fixed_seg = validate_localization(error_segment, formal_statement)
        if not loc_valid:
            fixes.append(f"localization: fuzzy match failed (kept original)")
        elif fixed_seg != error_segment and loc_valid:
            fixes.append(f"localization: fuzzy-matched to {fixed_seg[:60]!r}")
            error_segment = fixed_seg

    # ── Fix 3: Validate / fix correction ─────────────────────────────────────
    if corrected_statement and corrected_statement not in ("N/A", ""):
        corr_valid, fixed_corr = validate_correction(corrected_statement, header)
        if corr_valid and fixed_corr != corrected_statement:
            fixes.append("correction: appended ':= by sorry'")
            corrected_statement = fixed_corr

    # ── Compose updated parse_error note ─────────────────────────────────────
    if fixes:
        fix_note = f"[postprocess: {'; '.join(fixes)}]"
        parse_error = f"{parse_error} {fix_note}".strip()

    # Return a new Prediction with fixes applied
    return Prediction(
        idx=pred.idx,
        aligned=pred.aligned,
        error_category=error_category,
        error_segment=error_segment,
        corrected_statement=corrected_statement,
        reasoning=pred.reasoning,
        raw_response=pred.raw_response,
        parse_error=parse_error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from parser import Prediction

    print("=" * 60)
    print("Post-Processing Layer — Self Tests")
    print("=" * 60)

    all_pass = True

    def check(desc, got, expected):
        global all_pass
        ok = got == expected
        if not ok:
            all_pass = False
        print(f"  {'✓' if ok else '✗'} {desc}")
        if not ok:
            print(f"    Expected: {expected!r}")
            print(f"    Got:      {got!r}")

    # ── Localization tests ────────────────────────────────────────────────────
    formal = "theorem t : IsLeast {n | ∃ x y : ℝ, n = x * y ^ 2 + (x + 7) ^ 2} 45 := by sorry"

    ok, seg = validate_localization("x * y ^ 2", formal)
    check("exact localization match", ok, True)
    check("exact segment preserved", seg, "x * y ^ 2")

    ok2, seg2 = validate_localization("x_times_y_squared", formal)
    check("bad localization detected", ok2, False)

    ok3, seg3 = validate_localization("x * y^ 2", formal)  # extra space
    check("whitespace-normalized match", ok3, True)

    # ── Correction tests ──────────────────────────────────────────────────────
    valid, fixed = validate_correction("theorem t (x : ℝ) : x ^ 2 ≥ 0 := by sorry")
    check("valid correction detected", valid, True)
    check("valid correction unchanged", fixed, "theorem t (x : ℝ) : x ^ 2 ≥ 0 := by sorry")

    valid2, fixed2 = validate_correction("theorem t (x : ℝ) : x ^ 2 ≥ 0")
    check("missing sorry detected", valid2, True)
    check("sorry appended", fixed2, "theorem t (x : ℝ) : x ^ 2 ≥ 0 := by sorry")

    valid3, fixed3 = validate_correction("theorem t (x : ℝ) : x ^ 2 ≥ 0 :=")
    check("dangling := fixed", ":= by sorry" in fixed3, True)

    valid4, fixed4 = validate_correction("N/A")
    check("N/A correction detected invalid", valid4, False)

    # ── Category canonicalization tests ───────────────────────────────────────
    cat, fixed_cat = canonicalize_category("missing premise")
    check("case-insensitive canonicalized", fixed_cat, True)
    check("case-insensitive resolved", cat, "Missing Premise")

    cat2, fixed_cat2 = canonicalize_category("Missing Premise")
    check("canonical already correct", fixed_cat2, False)
    check("canonical preserved", cat2, "Missing Premise")

    cat3, fixed_cat3 = canonicalize_category("S1.1")
    check("code matched canonicalized", fixed_cat3, True)
    check("code resolved to name", cat3, "Quantifier Strengthening")

    # ── Full prediction postprocess ───────────────────────────────────────────
    sample = {
        "formal_statement": "theorem prob : IsLeast {n | ∃ x y : ℝ, n = x * y ^ 2} 0 := by sorry",
        "header": "import Mathlib",
    }
    pred = Prediction(
        idx="test_pp",
        aligned="Misaligned",
        error_category="Operator Precedence Error",  # canonical
        error_segment="x * y ^ 2",                  # valid
        corrected_statement="theorem prob : IsLeast {n | ∃ x y : ℝ, n = (x * y) ^ 2} 0",  # missing sorry
    )
    fixed_pred = postprocess_prediction(pred, sample)
    check("full pp: category preserved", fixed_pred.error_category, "Operator Precedence Error")
    check("full pp: sorry appended", ":= by sorry" in fixed_pred.corrected_statement, True)
    check("full pp: localization valid", fixed_pred.error_segment, "x * y ^ 2")
    check("full pp: parse_error logged", "postprocess" in (fixed_pred.parse_error or ""), True)

    # ── Binder parsing tests ──────────────────────────────────────────────────
    sig = "theorem putnam_1967_b1 (r : ℝ) (L : ZMod 6 → (EuclideanSpace ℝ (Fin 2))) : ... := by sorry"
    binders = _extract_binders_list(sig)
    check("binder list length", len(binders), 2)
    check("first binder", binders[0], "(r : ℝ)")
    check("second binder", binders[1], "(L : ZMod 6 → (EuclideanSpace ℝ (Fin 2)))")

    # ── Full prediction binder reclassification test ──────────────────────────
    sample_binder = {
        "formal_statement": "theorem olymid_ref_base_7968 (n : ℕ) (a : Fin n → ℝ) : ... := by sorry",
        "header": "import Mathlib",
    }
    pred_binder = Prediction(
        idx="test_binder",
        aligned="Misaligned",
        error_category="Domain Constraint Error",
        error_segment="n : ℕ",
        corrected_statement="theorem olymid_ref_base_7968 (n : ℕ) (hn : 0 < n) (a : Fin n → ℝ) : ... := by sorry",
    )
    fixed_binder = postprocess_prediction(pred_binder, sample_binder)
    check("binder reclass positivity", fixed_binder.error_category, "Positivity Constraint Error")

    pred_binder2 = Prediction(
        idx="test_binder2",
        aligned="Misaligned",
        error_category="Domain Constraint Error",
        error_segment="a : Fin n → ℝ",
        corrected_statement="theorem olymid_ref_base_7968 (n : ℕ) (h : LinearIndependent ℝ a) (a : Fin n → ℝ) : ... := by sorry",
    )
    fixed_binder2 = postprocess_prediction(pred_binder2, sample_binder)
    # ── Toggle exclude_categories tests ────────────────────────────────────────
    sample_exclude = {
        "formal_statement": "theorem prob : x < y := by sorry",
        "header": "import Mathlib",
    }
    pred_exclude = Prediction(
        idx="test_exclude",
        aligned="Misaligned",
        error_category="Positivity Constraint Error",
        error_segment="x < y",
        corrected_statement="theorem prob : x ≤ y := by sorry",
    )
    fixed_exclude = postprocess_prediction(pred_exclude, sample_exclude, {"exclude_categories": ["C1.1"]})
    check("exclude category test (demoted to aligned)", fixed_exclude.aligned, "Aligned")

    fixed_exclude_empty = postprocess_prediction(pred_exclude, sample_exclude, {"exclude_categories": []})
    check("empty exclude list test (retained)", fixed_exclude_empty.error_category, "Positivity Constraint Error")


    # ── Suffix stripping and trivial correction tests ─────────────────────────
    check("strip_proof_suffix: := sorry", _strip_proof_suffix("theorem t : 1 = 1 := sorry"), "theorem t : 1 = 1")
    check("strip_proof_suffix: := by sorry", _strip_proof_suffix("theorem t : 1 = 1 := by sorry"), "theorem t : 1 = 1")
    check("strip_proof_suffix: := by\\n  sorry", _strip_proof_suffix("theorem t : 1 = 1 := by\n  sorry"), "theorem t : 1 = 1")
    check("strip_proof_suffix: no suffix", _strip_proof_suffix("theorem t : 1 = 1"), "theorem t : 1 = 1")

    check("correction_is_trivial: only suffix diff", _correction_is_trivial("theorem t : 1 = 1 := sorry", "theorem t : 1 = 1 := by sorry"), True)
    check("correction_is_trivial: semantic diff (< to ≤)", _correction_is_trivial("theorem t (x : ℝ) : x < 1", "theorem t (x : ℝ) : x ≤ 1"), False)

    print()
    print("=" * 60)
    print("All tests passed ✓" if all_pass else "Some tests FAILED ✗")
