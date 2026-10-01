"""
prompt.py — V2 prompt construction with split verdict/correction, dynamic few-shot,
and Mathlib search reference injection.
"""

from taxonomy import (
    get_taxonomy_prompt_block, get_group_taxonomy_block,
    get_category_taxonomy_block, get_direct_classify_prompt_block,
    CATEGORY_NAMES, GROUP_NAMES, SINGLE_CATEGORY_GROUPS,
    CODE_TO_CATEGORY, normalize_category, _GROUP_TO_CATS,
)
from search_reference import format_search_reference_block

# ── Stage 1a: Verdict-Only ───────────────────────────────────────────────────

VERDICT_ONLY_SYSTEM_PROMPT = """\
You are an expert in Lean 4 and Mathlib. Determine if a candidate formalization is aligned or misaligned with the informal statement.

Think step-by-step (Chain-of-Thought)

Output ONLY valid JSON (no markdown fences):
{"reasoning": "<step-by-step CoT>", "verdict": "aligned"|"misaligned"}

## ⚠ CRITICAL: Guidelines on Uncertainty
Most formalizations are correct, but do not hesitate to mark "misaligned" if there is a clear semantic error, missing premise, or incorrect operator. If you are uncertain but find a strong evidence of mathematical discrepancy (e.g. missing positivity constraint, wrong bounds), prefer "misaligned".

## ALIGNED (do NOT flag):
- Section/context variables omitted from binders (standard Lean practice)
- Notation/definitional equivalence, alpha-equivalence, binder styles
- Subtype coercions, equivalent math forms, universe polymorphism
- Constraints implied by types (e.g. n : ℕ is non-negative)
- Partial formalization (one direction of iff/least)
- sorry vs by sorry vs := sorry
- Dot-notation chaining (e.g. `a.foo.bar` is equivalent to `(a.foo).bar`, or `a.alternatization.domCoprod` vs `(a.alternatization).domCoprod`). Do NOT flag syntax/formatting choices.
- Subsets vs Opens (e.g. `S : Set (Set α)` with `IsOpen s` is equivalent to `S : Set (Opens α)`). Equivalent type representations of topological sets are aligned.
- Equivalent math expressions (e.g. `‖x, y‖ * ‖y, x‖` vs `‖x, y‖ ^ 2` under symmetry, or `re ⟪x, x⟫` vs `⟪x, x⟫` when the inner product is real-valued).

## MISALIGNED (genuine errors only):
- Missing premise explicitly required by informal and NOT implied by types/context (e.g. missing positivity if informal explicitly says "positive", missing prime condition if informal says "prime").
- Wrong types/bounds/constants (< vs ≤, 3 vs 2)
- Quantifier swap (∀ vs ∃), wrong conclusion
- Extended vs Standard Metric Space mismatch: e.g. using `EMetric.diam` (returns `ℝ≥0∞`) when the informal statement implies a metric/pseudometric space where diameter is a real number (`Metric.diam` returns `ℝ`).
- Overly restrictive typeclass assumptions: e.g. using `[Ring R] [CommRing S]` when the mathlib theorem generalizes to `[Semiring R] [CommSemiring S]`.
- Operator precedence in coercions: e.g. writing `(a - b : ℤ)` when `a` is not cast (this parses as `a - (b : ℤ)` and causes a type mismatch). It must be written as `((a - b) : ℤ)`.
- Natural number subtraction underflow: e.g. writing `N - x` for `x : ℕ` without a `x ≤ N` constraint in the hypotheses (natural subtraction wraps/underflows in Lean).


## Rules
1. verdict: exactly "aligned" or "misaligned" (lowercase)
2. When uncertain → "misaligned"
"""

# ── Stage 1b: Correction + Localization (merged) ────────────────────────────

CORRECTION_LOCALIZATION_SYSTEM_PROMPT = """\
You are an expert in Lean 4 and Mathlib. A previous analysis identified that this formalization is MISALIGNED.

Your task: explain the error, locate the erroneous substring, and write the FULL corrected Lean 4 statement.

Output ONLY valid JSON (no markdown fences):
{"reasoning": "<brief explanation the error>", "error_location": "<MINIMAL erroneous substring from the formal statement>", "corrected_statement": "<full corrected Lean 4 statement ending with := by sorry>"}

## Rules
1. reasoning: explain what is wrong and why
2. error_location: must be a LITERAL substring of the formal statement. Pick the MINIMAL span that captures the error. If a part of hypothesis is MISSING or Incorrect, use "premise section".
3. corrected_statement: must be the COMPLETE corrected Lean 4 theorem statement. Must end with := by sorry (or := sorry). Fix ONLY the identified error — do not change unrelated parts.
4. DO NOT return "N/A", "none", "null", or empty strings for any field.
"""

# ── Stage 2b: Localization Only ──────────────────────────────────────────────

LOCALIZATION_SYSTEM_PROMPT = """\
You are an expert in formal mathematics and the Lean 4 theorem prover.
A previous analysis has determined:
- This formalization is MISALIGNED.
- The error category is: {error_category}
- The corrected statement has been provided.

Your ONLY task is to identify the MINIMAL erroneous substring in the original formal statement.

Output ONLY a JSON object with exactly these fields:

{{"reasoning": "<brief explanation>", "error_location": "<MINIMAL substring of formal_statement that is erroneous>"}}

## Rules
1. error_location must be a LITERAL substring of the formal statement. Pick the MINIMAL span that captures the error.
3. If the error category is "Missing Premise", Incorrect Premise , Redundant Premise the error_location must be "premise section".
4. DO NOT return "N/A", "none", "null", or an empty string.
5. Output ONLY valid JSON. No markdown fences.
"""

# ── Stage 2c Call 1: Group Classification ────────────────────────────────────

GROUP_CLASSIFY_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in formal mathematics and the Lean 4 theorem prover.
A previous analysis determined this formalization is MISALIGNED.
Your task is to identify which ERROR GROUP the error belongs to.

Common formalization errors:

{group_taxonomy_block}

## Selection Strategy
1. Is the mathematical MEANING wrong? → Semantic groups (Logical Structure Error, Mathematical Object Error, Mathematical Concept Error)
2. Is a CONSTRAINT, premise, or conclusion wrong/missing/extra? → Constraint groups (Variable Constraint Error, Range Error, Premise Error, Conclusion Error, Auxiliary Construction Error)
3. Is it a Lean-specific TECHNICAL issue? → Implementation Error

## ⚠ Priority Rules (Crucial)
Specific groups (Logical Structure, Mathematical Object, Mathematical Concept, Variable Constraint, Range, Implementation) ALWAYS take precedence over location-based catch-all groups:
- **Premise Error**: only choose this group if the error is inside a hypothesis/premise, AND no specific Semantic or specific Constraint group applies.
- **Conclusion Error**: only choose this group if the error is inside the main goal/conclusion, AND no specific Semantic or specific Constraint group applies.
- **Auxiliary Construction Error**: only choose this group if the error is inside a helper/witness binder (like let/have), AND no specific Semantic/Constraint group applies.

## Decision Procedure (Apply in this exact order)
1. PARSE CHECK: If the formal statement is syntactically incomplete (unbalanced brackets, cut-off expressions) -> Implementation Error (specifically I1 Truncation Error).
2. PRECEDENCE CHECK: If grouping of symbols is wrong -> Implementation Error (specifically I2 Operator Precedence Error).
3. SPECIFIC SEMANTIC: Try to fit in Semantic groups: Logical Structure, Mathematical Object, Mathematical Concept.
4. SPECIFIC CONSTRAINT: Try to fit in Constraint groups: Variable Constraint, Range Error.
5. LOCATION CATCH-ALL: If and only if steps 1-4 fail, categorize by location:
   - Error is in a hypothesis -> Premise Error.
   - Error is in the main goal -> Conclusion Error.
   - Error is in an auxiliary helper definition -> Auxiliary Construction Error.

Output ONLY valid JSON:
{{"reasoning": "<brief explanation>", "error_group": "<one of the {n_groups} group names>"}}

Rules: error_group must be EXACTLY one of: {group_names_str}
"""

# ── Stage 2c Call 2: Category Classification Template ────────────────────────

CATEGORY_CLASSIFY_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in formal mathematics.
A previous analysis determined:
- This formalization is MISALIGNED.

Your task is to identify the EXACT error category within this group, locate it.

{category_block}

## Confusion Guide
{confusion_guide}

Output ONLY valid JSON:
{{{{
  "reasoning": "<brief explanation>",
  "error_type": "<exact category name>"
}}}}

Rules:
1. error_type must be EXACTLY one of: {category_names_str}
"""

# ── Group Confusion Guides ───────────────────────────────────────────────────

_GROUP_CONFUSION_GUIDES: dict[str, str] = {
    "Logical Structure Error": (
        "- Quantifier Strengthening vs Weakening: Strengthening = ∃→∀ or domain narrowing. Weakening = ∀→∃ or domain broadening.\n"
        "- Quantifier error vs Logical Connective Misuse: if the root cause is a ∀/∃ flip, choose Quantifier Strengthening/Weakening even if connectives also change to match the quantifier change.\n"
        "- Logical Connective Misuse is statistically over-predicted. Check if it's truly a connective issue (∧ vs ∨, ↔ vs →, implication reversal, missing/extra ¬)."
    ),
    "Mathematical Object Error": (
        "- Object Type Error vs Domain Constraint Error: Object Type (S2.1) = wrong type of an object defined inside the goal. Domain Constraint (C1.3) = wrong type/domain declared for a variable in the hypotheses binder.\n"
        "- Function Confusion vs Operator Confusion: Function = named functions (like Real.log, sin). Operator = symbol operators (+, *, ∪, ∩).\n"
        "- Coefficient/Constant Error vs Bound Constraint Error: S2.5 = incorrect numbers/constants in formulas. C1.2 = incorrect upper/lower bounds on variable constraints."
    ),
    "Mathematical Concept Error": (
        "- Extremum Concept Error: Confusable pairs include: IsLeast ↔ IsGreatest, IsLeast ↔ IsGLB, IsGreatest ↔ IsLUB, sSup ↔ sInf, iSup ↔ iInf, IsMax ↔ IsLocalMax."
    ),
    "Variable Constraint Error": (
        "- Positivity Constraint Error: Specifically missing sign constraints (x > 0, n ≠ 0, x ≥ 0).\n"
        "- Bound Constraint Error: Wrong upper/lower bounds (e.g. [0,1] vs (0,1]).\n"
        "- Domain Constraint Error: Wrong domain/type declared in hypothesis binders.\n"
        "- Variable Constraint Error (C1.4): General catch-all for variable constraints (distinctness, coprimality, parity). Prefer C1.1-C1.3 first."
    ),
    "Range Error": (
        "- Range Shift (C2.1) vs Range Error (C2.2): C2.1 = entire range uniformly shifted (e.g. 0..n vs 1..n+1). C2.2 = specific endpoint is wrong (catch-all for range)."
    ),
    "Premise Error": (
        "- Missing Premise (C3.1) vs specific Variable Constraints: Before choosing Missing Premise, verify if the missing condition is a sign/positivity constraint (C1.1) or bound (C1.2).\n"
        "- Missing Premise vs Redundant Premise vs Incorrect Premise: Missing = absent; Redundant = extra/not needed; Incorrect = present but with wrong value/condition."
    ),
    "Implementation Error": (
        "- Truncation Error (I1) vs Operator Precedence (I2): I1 = syntax incomplete (cut-off). I2 = syntactically valid but parsed grouping is wrong due to precedence (e.g. x * y ^ 2)."
    ),
    "Conclusion Error": "Only choose this when the error is in the goal, and no specific Semantic/Constraint category fits.",
    "Auxiliary Construction Error": "Only choose this when the error is in a helper definition/let-binding, and no specific category fits.",
}

GROUP_CLASSIFY_FEW_SHOT = [
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\nopen Nat\n```\n## Informal Statement\nProve that there exists a prime number greater than 100.\n## Formal Statement\n```lean\ntheorem exists_prime_gt_100 : ∀ p : ℕ, Nat.Prime p → p > 100 := by sorry\n```\nThis formalization is misaligned. Identify which error group this belongs to."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Analyze error: The informal statement requires existence (∃) of a prime number, but the formal uses a universal quantifier (∀). This is a quantifier mismatch. (2) Decision Tree: First 2 steps check syntax completeness and operator precedence (both ok). Step 3 checks Specific Semantic. This is a quantifier flip, falling under Logical Structure Error.", "error_group": "Logical Structure Error"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nThe graph of a measurable function is a measurable set.\n## Formal Statement\n```lean\ntheorem measurableSet_graph : MeasurableSet { p : α × ℝ | p.snd = f p.fst } := sorry\n```\nThis formalization is misaligned. Identify which error group this belongs to."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Analyze error: The informal statement assumes f is measurable, but the hypothesis `Measurable f` is entirely missing from the formal theorem statement. (2) Decision Tree: Syntax and precedence are ok. Specific semantic and specific variable constraint groups do not fit. Step 5 Location Catch-All applies since a hypothesis is absent. Hence, it belongs to the Premise Error group.", "error_group": "Premise Error"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nProve that the sum of the first n positive integers is n(n+1)/2.\n## Formal Statement\n```lean\ntheorem sum_first_n (n : ℕ) : ∑ i ∈ Finset.range (n + 1), i = n * (n + 1) / 2 := by sorry\n```\nThis formalization is misaligned. Identify which error group this belongs to."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Analyze error: Finset.range (n+1) sums from 0 to n (size n+1), but the informal specifies the first n positive integers (1 to n). The upper/lower bounds of the summation are shifted. (2) Decision Tree: Syntax and precedence are ok. Semantic doesn\'t apply. Specific constraint matches summation range endpoints, which belongs to Range Error.", "error_group": "Range Error"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nFor all real numbers x, if x > 0 then x + 1 > 1.\n## Formal Statement\n```lean\ntheorem lt_add_one (x : ℝ) : x > 0 → x + 1 >\n```\nThis formalization is misaligned. Identify which error group this belongs to."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Analyze error: The formal statement is incomplete and cuts off at the very end. (2) Decision Tree: Step 1 (Parse Check) flags this immediately as an incomplete syntax truncation issue. This belongs to the Implementation Error group.", "error_group": "Implementation Error"}'
    }
]

CATEGORY_CLASSIFY_FEW_SHOT = [
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nThe graph of a measurable function is a measurable set.\n## Formal Statement\n```lean\ntheorem measurableSet_graph :\n MeasurableSet { p : α × ℝ | p.snd = f p.fst } := sorry\n```\nThis formalization is misaligned. The error belongs to the **Premise Error** group. Diagnose the exact category."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Within Premise Error: The hypothesis `Measurable f` is entirely missing from the formal theorem statement. Since this is an absent necessary condition, it is a Missing Premise. (2) Category: Missing Premise. (3) Location: premise section. (4) Fix: add (hf : Measurable f) to the hypotheses.", "error_type": "Missing Premise"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nIf the chromatic number of G is n+1, then G is (n+1)-colorable and not n-colorable.\n## Formal Statement\n```lean\ntheorem chromaticNumber_eq (hn : G.chromaticNumber = n + 1) :\n G.Colorable (n + 1) → ¬G.Colorable n := sorry\n```\nThis formalization is misaligned. The error belongs to the **Logical Structure Error** group. Diagnose the exact category."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Within Logical Structure Error: The conclusion should be a conjunction (G.Colorable (n+1) ∧ ¬G.Colorable n) but uses implication (→). This is a specific connective swap: ∧ vs →. (2) Category: Logical Connective Misuse. (3) Location: G.Colorable (n + 1) → ¬G.Colorable n. (4) Fix: change → to ∧.", "error_type": "Logical Connective Misuse"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nProve that for any positive real number x, x + 1/x ≥ 2.\n## Formal Statement\n```lean\ntheorem sum_recip_ge_two (x : ℝ) : x + 1 / x ≥ 2 := by sorry\n```\nThis formalization is misaligned. The error belongs to the **Variable Constraint Error** group. Diagnose the exact category."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Within Variable Constraint Error: The missing hypothesis is specifically the positivity constraint \'0 < x\'. Although a hypothesis is missing, specific variable constraints take priority over the catch-all Missing Premise (C3.1). (2) Category: Positivity Constraint Error. (3) Location: x : ℝ. (4) Fix: add (hx : 0 < x) as hypothesis.", "error_type": "Positivity Constraint Error"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\nopen Nat\n```\n## Informal Statement\nProve that there exists a prime number greater than 100.\n## Formal Statement\n```lean\ntheorem exists_prime_gt_100 : ∀ p : ℕ, Nat.Prime p → p > 100 := by sorry\n```\nThis formalization is misaligned. The error belongs to the **Logical Structure Error** group. Diagnose the exact category."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Within Logical Structure Error: the quantifier is wrong — informal uses ∃ but formal uses ∀, a strictly stronger and false claim. This is NOT Logical Connective Misuse — the root cause is the ∀/∃ flip; the → ↔ ∧ change is just the consequent fix-up. (2) Category: Quantifier Strengthening. (3) Location: the entire quantified expression. (4) Fix: change to ∃ with ∧.", "error_type": "Quantifier Strengthening"}'
    },
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nOn a preconnected set, a meromorphic function has finite order at one point if it has finite order at another point.\n## Formal Statement\n```lean\ntheorem meromorphicOrderAt_ne_top_of_isPreconnected (hf : MeromorphicOn f U) {y : 𝕜}\n (hU : IsPreconnected U) (h₁x : x ∈ U) (hy : y ∈ U) (h₂x : meromorphicOrderAt f x ≠ ⊤) :\n ∃ n : ℤ, meromorphicOrderAt f y = n := sorry\n```\nThis formalization is misaligned. The error belongs to the **Conclusion Error** group. Diagnose the exact category."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "(1) Within Conclusion Error: The error is in the main conclusion. Saying `∃ n : ℤ, meromorphicOrderAt f y = n` is not the canonical way to state finite order in this context in Mathlib, which is `meromorphicOrderAt f y ≠ ⊤`. Since the conclusion is incorrect and no specific semantic category fits, this is a Conclusion Error. (2) Category: Conclusion Error. (3) Location: ∃ n : ℤ, meromorphicOrderAt f y = n. (4) Fix: change to meromorphicOrderAt f y ≠ ⊤.", "error_type": "Conclusion Error"}'
    }
]



# ── Prompt Builders ──────────────────────────────────────────────────────────

def _build_sample_user_content(
    sample: dict,
    task_instruction: str,
    search_ref: dict | None = None,
) -> str:
    content = f"""\
## Header
```lean
{sample['header']}
```

## Informal Statement
{sample['informal_statement']}

## Formal Statement
```lean
{sample['formal_statement']}
```"""

    # Inject AST structure if available
    lean_struct = sample.get("lean_structure")
    if lean_struct:
        from lean_parser import LeanStructure, Hypothesis, Quantifier
        try:
            if isinstance(lean_struct, dict):
                hyps = [Hypothesis(**h) for h in lean_struct.get("hypotheses", [])]
                quants = [Quantifier(**q) for q in lean_struct.get("quantifiers", [])]
                struct_obj = LeanStructure(
                    theorem_name=lean_struct.get("theorem_name", ""),
                    command=lean_struct.get("command", ""),
                    hypotheses=hyps,
                    goal=lean_struct.get("goal", ""),
                    quantifiers=quants,
                    binder_count=lean_struct.get("binder_count", 0),
                    proof_term=lean_struct.get("proof_term", ""),
                    has_where_clause=lean_struct.get("has_where_clause", False),
                )
                struct_str = struct_obj.to_prompt_block()
                content += f"\n\n## Formal Structure (parsed)\n{struct_str}"
        except Exception:
            pass

    # Inject Mathlib search reference if available
    if search_ref:
        ref_block = format_search_reference_block(search_ref)
        if ref_block:
            content += f"\n\n{ref_block}"

    content += f"\n\n{task_instruction}"
    return content


VERDICT_ONLY_STATIC_FEW_SHOT = [
    # ── Misaligned: Operator Precedence ──
    {"role": "user", "content": """\
## Header
```lean
import Mathlib
open Real Set
```
## Informal Statement
Find the minimum of (xy)^2 + (x+7)^2 + (2y+7)^2. Answer: 45
## Formal Statement
```lean
theorem prob_42 : IsLeast {n | ∃ x y : ℝ, n = x * y ^ 2 + (x + 7) ^ 2 + (2 * y + 7) ^ 2} 45 := by sorry
```
Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement."""},
    {"role": "assistant", "content": '{"reasoning": "(xy)^2 requires (x * y) ^ 2 but formal has x * y ^ 2 which parses as x * (y^2).", "verdict": "misaligned" }'},
    # ── Misaligned: Missing Premise ──
    {"role": "user", "content": """\
## Header
```lean
import Mathlib
```
## Informal Statement
The graph of a measurable function is a measurable set.
## Formal Statement
```lean
theorem measurableSet_graph : MeasurableSet { p : α × ℝ | p.snd = f p.fst } := sorry
```
Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement."""},
    {"role": "assistant", "content": '{"reasoning": "Informal says \\"measurable function\\" — the hypothesis Measurable f is missing.", "verdict": "misaligned" }'},
    # ── Misaligned: Missing Positivity ──
    {"role": "user", "content": """\
## Header
```lean
import Mathlib
```
## Informal Statement
Prove that for any positive real number x, x + 1/x ≥ 2.
## Formal Statement
```lean
theorem sum_recip_ge_two (x : ℝ) : x + 1 / x ≥ 2 := by sorry
```
Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement."""},
    {"role": "assistant", "content": '{"reasoning": "Informal explicitly says \\"positive real number x\\". The hypothesis 0 < x is missing.", "verdict": "misaligned" }'},
    # ── Aligned: Section variables are NOT missing premises ──
    {"role": "user", "content": """\
## Header
```lean
import Mathlib
variable {α : Type*} [MeasurableSpace α] {μ : Measure α}
variable {f g : α → ℝ}
```
## Informal Statement
If f and g are integrable, then f + g is integrable.
## Formal Statement
```lean
theorem Integrable.add (hf : Integrable f μ) (hg : Integrable g μ) : Integrable (f + g) μ := sorry
```
Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement."""},
    {"role": "assistant", "content": '{"reasoning": "Variables α, μ, f, g are declared in section header. All premises present. Aligned.", "verdict": "aligned" }'},
    # ── Aligned: No need to add constraints not in informal ──
    {"role": "user", "content": """\
## Header
```lean
import Mathlib
```
## Informal Statement
Determine the smallest M such that |ab(a²-b²)+bc(b²-c²)+ca(c²-a²)| ≤ M·(a²+b²+c²)² for all reals a, b, c.
## Formal Statement
```lean
theorem olymid_ref (M : ℝ) : IsLeast {M | ∀ a b c : ℝ, |a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2)| ≤ M * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2} M := sorry
```
Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement."""},
    {"role": "assistant", "content": '{"reasoning": "Correctly captures finding smallest M. No need for M ≥ 0 — IsLeast handles it. No missing constraints.", "verdict": "aligned" }'},
    # ── Aligned: Correct formalization ──
    {"role": "user", "content": """\
## Header
```lean
import Mathlib
open Real
```
## Informal Statement
Prove that for all real numbers a and b, (a + b)^2 = a^2 + 2*a*b + b^2.
## Formal Statement
```lean
theorem sq_add (a b : ℝ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by sorry
```
Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement."""},
    {"role": "assistant", "content": '{"reasoning": "Correctly expresses the binomial identity. Precedence is correct.", "verdict": "aligned"}'},
]


def build_verdict_only_message(
    sample: dict,
    use_few_shot: bool = True,
    few_shot_count: int = 2,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 1a message list: Verdict + Draft Correction (for self-consistency)."""
    messages = [{"role": "system", "content": VERDICT_ONLY_SYSTEM_PROMPT}]
    if use_few_shot:
        messages.extend(VERDICT_ONLY_STATIC_FEW_SHOT[:few_shot_count * 2])
    messages.append({
        "role": "user",
        "content": _build_sample_user_content(
            sample,
            "Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement.",
            search_ref=search_ref,
        ),
    })
    return messages


def build_correction_localization_message(
    sample: dict,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 1b message list: Correction + Localization (merged)."""
    messages = [{"role": "system", "content": CORRECTION_LOCALIZATION_SYSTEM_PROMPT}]
    task_instruction = (
        "This formalization is misaligned. "
        "Explain the error, locate the erroneous substring, and write the corrected Lean 4 statement."
    )

    messages.append({
        "role": "user",
        "content": _build_sample_user_content(sample, task_instruction, search_ref=search_ref),
    })
    return messages


# Backward compatible alias
def build_correction_only_message(
    sample: dict,
    search_ref: dict | None = None,
) -> list[dict]:
    """Legacy alias for build_correction_localization_message."""
    return build_correction_localization_message(sample, search_ref=search_ref)



def build_verdict_correction_message(
    sample: dict,
    use_few_shot: bool = True,
    few_shot_count: int = 2,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 1 message (legacy compatible): Verdict + Correction."""
    messages = [{"role": "system", "content": VERDICT_ONLY_SYSTEM_PROMPT}]
    if use_few_shot:
        messages.extend(VERDICT_ONLY_STATIC_FEW_SHOT[:few_shot_count * 2])
    messages.append({
        "role": "user",
        "content": _build_sample_user_content(
            sample,
            "Determine if this formalization is aligned or misaligned. If misaligned, provide the corrected statement.",
            search_ref=search_ref,
        ),
    })
    return messages


def build_localization_message(
    sample: dict,
    error_category: str,
    corrected_statement: str,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 2b message list: Localization Only."""
    system_prompt = LOCALIZATION_SYSTEM_PROMPT.format(error_category=error_category)
    messages = [{"role": "system", "content": system_prompt}]
    task = f"This formalization is misaligned. The error category is '{error_category}'.\nCorrected statement:\n```lean\n{corrected_statement}\n```\nLocate the exact erroneous substring."
    messages.append({"role": "user", "content": _build_sample_user_content(sample, task, search_ref=search_ref)})
    return messages


def build_group_classify_message(
    sample: dict,
    use_few_shot: bool = True,
    few_shot_count: int = 2,
    exclude_categories: list[str] | None = None,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 2c Call 1: Group Classification."""
    exclude_set = set()
    if exclude_categories:
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

    group_taxonomy_block = get_group_taxonomy_block(exclude_set)
    active_groups = []
    for g in GROUP_NAMES:
        group_cats = _GROUP_TO_CATS.get(g, [])
        if any(c.name not in exclude_set for c in group_cats):
            active_groups.append(g)

    group_names_str = ", ".join(f'"{g}"' for g in active_groups)
    system_prompt = GROUP_CLASSIFY_SYSTEM_PROMPT_TEMPLATE.format(
        group_taxonomy_block=group_taxonomy_block,
        n_groups=len(active_groups),
        group_names_str=group_names_str,
    )

    messages = [{"role": "system", "content": system_prompt}]
    if use_few_shot:
        messages.extend(GROUP_CLASSIFY_FEW_SHOT[:few_shot_count * 2])
    messages.append({
        "role": "user",
        "content": _build_sample_user_content(
            sample,
            "This formalization is misaligned. Identify which error group this belongs to.",
            search_ref=search_ref,
        ),
    })
    return messages


def build_category_classify_message(
    sample: dict,
    group_name: str,
    use_few_shot: bool = True,
    few_shot_count: int = 2,
    exclude_categories: list[str] | None = None,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 2c Call 2: Category Classification within a group."""
    exclude_set = set()
    if exclude_categories:
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

    category_block = get_category_taxonomy_block(group_name, exclude_set)
    confusion_guide = _GROUP_CONFUSION_GUIDES.get(group_name, "No specific confusion pairs.")
    
    from taxonomy import get_categories_for_group
    cats = get_categories_for_group(group_name)
    active_cats = [c for c in cats if c.name not in exclude_set]
    category_names_str = ", ".join(f'"{c.name}"' for c in active_cats)

    system_prompt = CATEGORY_CLASSIFY_SYSTEM_PROMPT_TEMPLATE.format(
        category_block=category_block,
        confusion_guide=confusion_guide,
        category_names_str=category_names_str,
    )
    messages = [{"role": "system", "content": system_prompt}]
    if use_few_shot:
        messages.extend(CATEGORY_CLASSIFY_FEW_SHOT[:few_shot_count * 2])
    task = f"This formalization is misaligned. The error belongs to the **{group_name}** group. Diagnose the exact category."
    messages.append({"role": "user", "content": _build_sample_user_content(sample, task, search_ref=search_ref)})
    return messages


# ── Stage 2 (1-call): Direct Category Classification ────────────────────────

DIRECT_CLASSIFY_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in formal mathematics and the Lean 4 theorem prover.
A previous analysis determined this formalization is MISALIGNED.
Your task is to identify the EXACT error category from the SCI taxonomy.

{taxonomy_block}

## Decision Procedure (Apply in this exact order)
1. PARSE CHECK: If the formal statement is syntactically incomplete (unbalanced brackets, cut-off expressions) → I1 Truncation Error.
2. PRECEDENCE CHECK: If grouping of symbols is wrong due to Lean 4 operator precedence → I2 Operator Precedence Error.
3. QUANTIFIER CHECK: Is the quantifier wrong (∀ vs ∃, domain narrowed/broadened)? → S1.1 or S1.2.
4. CONNECTIVE CHECK: Is a logical connective wrong (∧ vs ∨, → vs ↔, missing/extra ¬)? → S1.3.
5. MATH OBJECT CHECK: Is a mathematical object, operator, function, constant, or index wrong? → S2.x.
6. MATH CONCEPT CHECK: Is a high-level mathematical concept wrong (infinity, extremum, cardinality, integration)? → S3.x.
7. CONSTRAINT CHECK: Is a variable constraint wrong or missing (sign, bounds, domain, general)? → C1.x.
8. RANGE CHECK: Is a summation/product/integral range shifted or wrong? → C2.x.
9. PREMISE CHECK: Is a hypothesis missing (C3.1), redundant (C3.2), or incorrect (C3.3)?
10. CONCLUSION CHECK: Is the main goal/conclusion wrong (and no specific category from steps 1-9 applies)? → C4.
11. AUXILIARY CHECK: Is a helper definition (let/have/where) wrong? → C5.

Output ONLY valid JSON (no markdown fences):
{{"reasoning": "<step-by-step analysis following the decision procedure>", "error_type": "<exact category name>"}}

Rules:
1. error_type must be EXACTLY one of: {category_names_str}
2. Follow the decision procedure strictly — specific categories (steps 1-8) ALWAYS take precedence over location-based catch-alls (steps 9-11).
3. CRITICAL: If a premise is missing and it is a sign/positivity constraint (0 < x, x > 0, x ≠ 0) → C1.1. If it is a bound constraint (x ≤ N, x < 1) → C1.2. If it is a domain/type (ℤ vs ℕ, Ring vs Semiring) → C1.3. Use C3.1 Missing Premise ONLY for premises that don't fit C1.1-C1.4.
"""

DIRECT_CLASSIFY_FEW_SHOT = [
    # ── Quantifier Strengthening (not Logical Connective Misuse) ──
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\nopen Nat\n```\n## Informal Statement\nProve that there exists a prime number greater than 100.\n## Formal Statement\n```lean\ntheorem exists_prime_gt_100 : ∀ p : ℕ, Nat.Prime p → p > 100 := by sorry\n```\nThis formalization is misaligned. Classify the error."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "Step 3 (Quantifier Check): The informal says \\"there exists\\" (∃) but the formal uses ∀ (for all). This is a quantifier flip from ∃ to ∀, making the claim strictly stronger and false. The → vs ∧ change is just a consequence of the quantifier swap. Category: Quantifier Strengthening.", "error_type": "Quantifier Strengthening"}'
    },
    # ── Positivity Constraint Error (not Missing Premise) ──
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nProve that for any positive real number x, x + 1/x ≥ 2.\n## Formal Statement\n```lean\ntheorem sum_recip_ge_two (x : ℝ) : x + 1 / x ≥ 2 := by sorry\n```\nThis formalization is misaligned. Classify the error."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "Step 7 (Constraint Check): The informal explicitly says \\"positive real number x\\", meaning x > 0. The hypothesis `0 < x` is missing. This is specifically a sign/positivity constraint, so C1.1 Positivity Constraint Error takes precedence over C3.1 Missing Premise.", "error_type": "Positivity Constraint Error"}'
    },
    # ── Conclusion Error (not Logical Connective Misuse) ──
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nOn a preconnected set, a meromorphic function has finite order at one point if it has finite order at another point.\n## Formal Statement\n```lean\ntheorem meromorphicOrderAt_ne_top (hf : MeromorphicOn f U) {y : 𝕜}\n (hU : IsPreconnected U) (hx : x ∈ U) (hy : y ∈ U) (h₂x : meromorphicOrderAt f x ≠ ⊤) :\n ∃ n : ℤ, meromorphicOrderAt f y = n := sorry\n```\nThis formalization is misaligned. Classify the error."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "Steps 1-4: No syntax/precedence/quantifier/connective issues. Steps 5-8: No math object/concept/constraint/range issues. Steps 9: Premises look complete. Step 10 (Conclusion Check): The conclusion `∃ n : ℤ, meromorphicOrderAt f y = n` is semantically wrong — the correct way to express finite order in Mathlib is `meromorphicOrderAt f y ≠ ⊤`. No specific connective is swapped (this is not S1.3). The goal itself is wrong. Category: Conclusion Error.", "error_type": "Conclusion Error"}'
    },
    # ── Logical Connective Misuse (not Conclusion Error) ──
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nIf the chromatic number of G is n+1, then G is (n+1)-colorable and not n-colorable.\n## Formal Statement\n```lean\ntheorem chromaticNumber_eq (hn : G.chromaticNumber = n + 1) :\n G.Colorable (n + 1) → ¬G.Colorable n := sorry\n```\nThis formalization is misaligned. Classify the error."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "Steps 1-2: No syntax/precedence issues. Step 3: No quantifier issues. Step 4 (Connective Check): The conclusion should be a conjunction (G.Colorable (n+1) ∧ ¬G.Colorable n) but uses implication (→). This is a specific connective swap: ∧ vs →. Category: Logical Connective Misuse.", "error_type": "Logical Connective Misuse"}'
    },
    # ── Incorrect Premise (not Missing Premise) ──
    {
        "role": "user",
        "content": "## Header\n```lean\nimport Mathlib\n```\n## Informal Statement\nFor a semiring R and commutative semiring S, f preserves multiplication.\n## Formal Statement\n```lean\ntheorem map_mul [Ring R] [CommRing S] (f : R →+* S) (a b : R) :\n f (a * b) = f a * f b := sorry\n```\nThis formalization is misaligned. Classify the error."
    },
    {
        "role": "assistant",
        "content": '{"reasoning": "Steps 1-8: No syntax/precedence/quantifier/connective/math object/concept/constraint/range issues. Step 9 (Premise Check): The premises [Ring R] [CommRing S] ARE present — they are not missing. But they are WRONG: the informal says Semiring/CommSemiring, not Ring/CommRing. These are overly restrictive typeclass assumptions. The hypothesis exists but is incorrect. Category: Incorrect Premise.", "error_type": "Incorrect Premise"}'
    },
]


def build_direct_classify_message(
    sample: dict,
    use_few_shot: bool = True,
    few_shot_count: int = 3,
    exclude_categories: list[str] | None = None,
    search_ref: dict | None = None,
) -> list[dict]:
    """Build Stage 2 (1-call) message: Direct Category Classification."""
    exclude_set = set()
    if exclude_categories:
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

    taxonomy_block = get_direct_classify_prompt_block(exclude_set)
    active_cats = [c for c in CATEGORY_NAMES if c not in exclude_set]
    category_names_str = ", ".join(f'"{c}"' for c in active_cats)

    system_prompt = DIRECT_CLASSIFY_SYSTEM_PROMPT_TEMPLATE.format(
        taxonomy_block=taxonomy_block,
        category_names_str=category_names_str,
    )

    messages = [{"role": "system", "content": system_prompt}]
    if use_few_shot:
        messages.extend(DIRECT_CLASSIFY_FEW_SHOT[:few_shot_count * 2])
    messages.append({
        "role": "user",
        "content": _build_sample_user_content(
            sample,
            "This formalization is misaligned. Classify the error.",
            search_ref=search_ref,
        ),
    })
    return messages


def build_compiler_feedback_message(messages: list[dict], error_output: str) -> list[dict]:
    """Append compiler feedback to message history."""
    new_messages = messages.copy()
    new_messages.append({"role": "user", "content": f"## Compiler Feedback\nYour previous corrected_statement failed to compile:\n```\n{error_output}\n```\nPlease fix and output the JSON again."})
    return new_messages
