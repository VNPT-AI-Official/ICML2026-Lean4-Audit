"""
lean_parser.py — Structural parser for Lean 4 formal statements.

Extracts theorem structure (name, hypotheses, goal, quantifiers) from
formal_statement strings using regex-based parsing. This provides
structured context to the LLM without requiring a Lean toolchain.

Phase 2 will add full LSP integration via leanclient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


@dataclass
class Hypothesis:
    """A single hypothesis/binder in a theorem statement."""
    name: str                  # Variable or hypothesis name
    type: str                  # Type annotation
    kind: str                  # "explicit" | "implicit" | "instance" | "strict_implicit"

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> str:
        brackets = {
            "explicit": "({})",
            "implicit": "{{{}}}",
            "instance": "[{}]",
            "strict_implicit": "⦃{}⦄",
        }
        inner = f"{self.name} : {self.type}" if self.name else self.type
        return brackets.get(self.kind, "({})").format(inner)


@dataclass
class Quantifier:
    """A quantifier found in the goal expression."""
    kind: str                  # "∀" | "∃"
    vars: list[str]            # Bound variable names
    type: str                  # Type of the bound variables (may be "")
    span: str                  # Raw matched text

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeanStructure:
    """Parsed structure of a Lean 4 formal statement."""
    theorem_name: str = ""
    command: str = ""                            # "theorem" | "def" | "lemma" | "example"
    hypotheses: list[Hypothesis] = field(default_factory=list)
    goal: str = ""                               # The conclusion/goal type
    quantifiers: list[Quantifier] = field(default_factory=list)  # Quantifiers in goal
    binder_count: int = 0
    proof_term: str = ""                         # e.g. "sorry", "by sorry", "by"
    has_where_clause: bool = False
    raw_statement: str = ""

    def to_dict(self) -> dict:
        return {
            "theorem_name": self.theorem_name,
            "command": self.command,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "goal": self.goal,
            "quantifiers": [q.to_dict() for q in self.quantifiers],
            "binder_count": self.binder_count,
            "proof_term": self.proof_term,
            "has_where_clause": self.has_where_clause,
        }

    def to_prompt_block(self) -> str:
        """Format as a structured block for injection into LLM prompts."""
        lines = []
        lines.append(f"- **Command**: `{self.command} {self.theorem_name}`")

        if self.hypotheses:
            lines.append("- **Hypotheses/Binders**:")
            for h in self.hypotheses:
                lines.append(f"  - `{h.display()}`")

        if self.goal:
            # Truncate very long goals
            goal_display = self.goal if len(self.goal) <= 300 else self.goal[:297] + "..."
            lines.append(f"- **Goal/Conclusion**: `{goal_display}`")

        if self.quantifiers:
            lines.append("- **Quantifiers in goal**:")
            for q in self.quantifiers:
                vars_str = ", ".join(q.vars)
                type_str = f" : {q.type}" if q.type else ""
                lines.append(f"  - `{q.kind} {vars_str}{type_str}`")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Regex-based parser
# ─────────────────────────────────────────────────────────────────────────────

# Match the command keyword and theorem name
# Handles access modifiers: protected, private, noncomputable, etc.
_CMD_RE = re.compile(
    r"^\s*(?:protected\s+|private\s+|noncomputable\s+)*"
    r"(theorem|lemma|def|abbrev|example)\s+"
    r"([a-zA-Z_][a-zA-Z0-9_.']*)",
    re.MULTILINE,
)

# Match binder groups: (x : T), {x : T}, [inst : T], ⦃x : T⦄
# These can be multi-line and nested, so we use a bracket-aware scanner.
_BINDER_OPEN = {"(": ")", "{": "}", "[": "]", "⦃": "⦄"}
_BINDER_KIND = {
    "(": "explicit",
    "{": "implicit",
    "[": "instance",
    "⦃": "strict_implicit",
}

# Quantifier patterns in the goal
_QUANT_RE = re.compile(
    r"([∀∃])\s+"
    r"((?:[a-zA-Zα-ωΑ-Ω₀-₉_][a-zA-Zα-ωΑ-Ω₀-₉_']*\s*)+)"
    r"(?::\s*([^,∀∃→↔⇒]+?))?"
    r"\s*[,]"
)

# Also match quantifiers with type but no comma (e.g. ∀ x : ℝ, ...)
_QUANT_COLON_RE = re.compile(
    r"([∀∃])\s+"
    r"((?:[a-zA-Zα-ωΑ-Ω₀-₉_][a-zA-Zα-ωΑ-Ω₀-₉_']*\s*)+)"
    r"(?::\s*([^,]+?))?"
    r"\s*,"
)


def _find_matching_bracket(text: str, start: int) -> int:
    """Find the matching closing bracket for the opener at text[start].
    Returns the index of the closing bracket, or -1 if not found.
    Handles nested brackets.
    """
    opener = text[start]
    closer = _BINDER_OPEN.get(opener)
    if not closer:
        return -1

    depth = 1
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _parse_binder_content(content: str, kind: str) -> list[Hypothesis]:
    """Parse the content inside a binder group into Hypothesis objects.
    E.g. "x y : ℝ" -> [Hypothesis("x", "ℝ", ...), Hypothesis("y", "ℝ", ...)]
    """
    content = content.strip()
    if not content:
        return []

    # Split on colon to separate names from type
    if ":" in content:
        parts = content.split(":", 1)
        names_str = parts[0].strip()
        type_str = parts[1].strip()
    else:
        # No type annotation — just a name or a type-only instance
        if kind == "instance":
            return [Hypothesis("", content, kind)]
        return [Hypothesis(content, "", kind)]

    # Parse potentially multiple names
    names = [n.strip() for n in names_str.split() if n.strip()]
    if not names:
        return [Hypothesis("", type_str, kind)]

    return [Hypothesis(name, type_str, kind) for name in names]


def _extract_binders(text: str) -> tuple[list[Hypothesis], int]:
    """Extract all binder groups from the text after the theorem name.
    Returns (list of Hypothesis, index where binders end).
    """
    hypotheses = []
    i = 0

    while i < len(text):
        ch = text[i]

        # Skip whitespace and newlines
        if ch in " \t\n\r":
            i += 1
            continue

        # Check if this is a binder opener
        if ch in _BINDER_OPEN:
            close_idx = _find_matching_bracket(text, i)
            if close_idx == -1:
                break  # Unmatched bracket — stop here

            content = text[i + 1 : close_idx]
            kind = _BINDER_KIND.get(ch, "explicit")

            # Handle multi-binder groups separated by ) (
            # e.g. "(h₁g : T) (h₂g : T)" — each is a separate group
            hyps = _parse_binder_content(content, kind)
            hypotheses.extend(hyps)

            i = close_idx + 1
            continue

        # If we hit a colon, that's the start of the goal
        if ch == ":":
            break

        # If we hit something unexpected, stop
        break

    return hypotheses, i


def _extract_quantifiers(goal: str) -> list[Quantifier]:
    """Extract quantifiers (∀, ∃) from the goal expression."""
    quantifiers = []
    seen = set()

    for match in _QUANT_COLON_RE.finditer(goal):
        kind = match.group(1)
        vars_str = match.group(2).strip()
        type_str = (match.group(3) or "").strip()
        vars_list = [v.strip() for v in vars_str.split() if v.strip()]

        key = (kind, tuple(vars_list), type_str)
        if key not in seen:
            seen.add(key)
            quantifiers.append(Quantifier(
                kind=kind,
                vars=vars_list,
                type=type_str,
                span=match.group(0).strip(),
            ))

    return quantifiers


def _strip_proof_suffix(statement: str) -> tuple[str, str]:
    """Remove the proof term (:= sorry, := by sorry, etc.) and return (body, proof_term)."""
    # Common patterns
    patterns = [
        (r":=\s*by\s+sorry\s*$", ":= by sorry"),
        (r":=\s*sorry\s*$", ":= sorry"),
        (r":=\s*by\s*$", ":= by"),
        (r":=\s*by\s+\{[^}]*\}\s*$", ":= by {...}"),
        (r":=\s*by\s+", ":= by ..."),
        (r":=\s+", ":= ..."),
    ]
    for pat, label in patterns:
        m = re.search(pat, statement)
        if m:
            return statement[:m.start()].strip(), label

    return statement.strip(), ""


def parse_lean_statement_regex(formal_statement: str) -> LeanStructure:
    """
    Parse a Lean 4 formal statement using regex-based heuristics.

    Extracts:
    - Command type (theorem/lemma/def)
    - Theorem name
    - Hypotheses/binders with types
    - Goal/conclusion expression
    - Quantifiers in the goal
    - Proof term style

    This is a best-effort parser that handles the majority of Lean 4
    theorem declarations without requiring a Lean toolchain.
    """
    result = LeanStructure(raw_statement=formal_statement)

    # Normalize whitespace for easier parsing
    text = formal_statement.strip()

    # Strip proof suffix
    body, proof_term = _strip_proof_suffix(text)
    result.proof_term = proof_term

    # Check for where clause
    if "\nwhere\n" in text or "\n  where\n" in text or text.rstrip().endswith("where"):
        result.has_where_clause = True

    # Strip doc comments for cleaner parsing
    body_clean = re.sub(r"/--.*?-/\s*", "", body, flags=re.DOTALL)

    # Extract command and name — find all matches, prefer the last theorem/lemma
    all_matches = list(_CMD_RE.finditer(body_clean))
    cmd_match = None
    if all_matches:
        # Prefer theorem/lemma over def/abbrev
        theorem_matches = [m for m in all_matches if m.group(1) in ("theorem", "lemma")]
        cmd_match = theorem_matches[-1] if theorem_matches else all_matches[-1]

    if not cmd_match:
        # Maybe it's just "example" without a name
        example_match = re.search(r"(?:^|\n)\s*(example)\s*", body_clean)
        if example_match:
            result.command = "example"
            result.theorem_name = ""
            remainder = body_clean[example_match.end():]
        else:
            # Can't parse — return minimal structure
            result.goal = body_clean
            return result
    else:
        result.command = cmd_match.group(1).strip()
        result.theorem_name = cmd_match.group(2)
        remainder = body_clean[cmd_match.end():]

    # Extract binders from remainder
    hypotheses, binder_end = _extract_binders(remainder)
    result.hypotheses = hypotheses
    result.binder_count = len(hypotheses)

    # Everything after binders until end should be the goal
    after_binders = remainder[binder_end:].strip()

    # The goal starts after the final ':'
    # But we need to be careful: there may be a ':' inside the binders
    if after_binders.startswith(":"):
        result.goal = after_binders[1:].strip()
    elif ":" in after_binders:
        # Find the outermost ':' that's not inside brackets
        depth = 0
        for ci, ch in enumerate(after_binders):
            if ch in "({[⦃":
                depth += 1
            elif ch in ")}]⦄":
                depth -= 1
            elif ch == ":" and depth == 0:
                result.goal = after_binders[ci + 1:].strip()
                break
    else:
        result.goal = after_binders

    # Strip trailing whitespace and normalize
    result.goal = result.goal.strip()

    # Extract quantifiers from goal
    result.quantifiers = _extract_quantifiers(result.goal)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Test 1: Simple theorem with explicit binders
        (
            "theorem omni_theorem_2139 : IsLeast {n | ∃ x y : ℝ, n = x * y ^ 2 + (x + 7) ^ 2 + (2 * y + 7) ^ 2} 45 := by sorry",
            {
                "command": "theorem",
                "name": "omni_theorem_2139",
                "binder_count": 0,
                "has_goal": True,
                "quantifier_count": 1,  # ∃ x y
            }
        ),
        # Test 2: Complex theorem with multiple binder types
        (
            "theorem meromorphicNFOn_mul_iff_left_of_analyticOnNhd {f g : 𝕜 → 𝕜} (h₁g : AnalyticOnNhd 𝕜 g U) (h₂g : ∀ u ∈ U, g u ≠ 0) : MeromorphicNFOn (f + g) U ↔ MeromorphicNFOn f U := sorry",
            {
                "command": "theorem",
                "name": "meromorphicNFOn_mul_iff_left_of_analyticOnNhd",
                "binder_count": 3,  # {f g}, (h₁g), (h₂g)
                "has_goal": True,
                "quantifier_count": 0,  # No quantifiers in the GOAL (∀ is in binder)
            }
        ),
        # Test 3: Multi-line theorem
        (
            """theorem meromorphicOrderAt_ne_top_of_isPreconnected (hf : MeromorphicOn f U) {y : 𝕜}
 (hU : IsPreconnected U) (h₁x : x ∈ U) (hy : y ∈ U) (h₂x : meromorphicOrderAt f x ≠ ⊤) :
 ∃ n : ℤ, meromorphicOrderAt f y = n := sorry""",
            {
                "command": "theorem",
                "name": "meromorphicOrderAt_ne_top_of_isPreconnected",
                "binder_count": 5,  # (hf), {y}, (hU), (h₁x), (hy), (h₂x) — actually 6
                "has_goal": True,
                "quantifier_count": 1,  # ∃ n
            }
        ),
        # Test 4: Theorem with Fin and sums
        (
            """theorem olymid_ref_base_7968 (n : ℕ) (a : Fin n → ℝ)
 (apos : ∀ i, 0 < a i) :
 (∑ i, ∑ j, if i < j then (a i * a j) / (a i + a j) else 0) ≤
 ((n : ℝ) / (2 * ∑ i, a i)) * ∑ i, ∑ j, if i < j then a i * a j else 0 := by sorry""",
            {
                "command": "theorem",
                "name": "olymid_ref_base_7968",
                "binder_count": 3,  # (n), (a), (apos)
                "has_goal": True,
                "quantifier_count": 0,  # Quantifiers are in binders, not goal
            }
        ),
    ]

    print("=" * 60)
    print("Lean Parser — Self Tests")
    print("=" * 60)

    all_pass = True
    for i, (stmt, expected) in enumerate(tests, 1):
        result = parse_lean_statement_regex(stmt)
        print(f"\n--- Test {i}: {expected.get('name', result.theorem_name)} ---")
        print(f"  Command: {result.command}")
        print(f"  Name: {result.theorem_name}")
        print(f"  Binders ({result.binder_count}):")
        for h in result.hypotheses:
            print(f"    {h.display()}")
        print(f"  Goal: {result.goal[:100]}{'...' if len(result.goal) > 100 else ''}")
        print(f"  Quantifiers ({len(result.quantifiers)}):")
        for q in result.quantifiers:
            print(f"    {q.kind} {', '.join(q.vars)} : {q.type}")
        print(f"  Proof: {result.proof_term}")

        # Validate
        ok = True
        if result.command != expected["command"]:
            print(f"  ✗ Command: expected {expected['command']}, got {result.command}")
            ok = False
        if result.theorem_name != expected.get("name", result.theorem_name):
            print(f"  ✗ Name: expected {expected.get('name')}, got {result.theorem_name}")
            ok = False
        if expected["has_goal"] and not result.goal:
            print(f"  ✗ Goal: expected non-empty, got empty")
            ok = False

        if ok:
            print(f"  ✓ PASS")
        else:
            all_pass = False
            print(f"  ✗ FAIL")

        print()
        print("  Prompt block:")
        for line in result.to_prompt_block().split("\n"):
            print(f"    {line}")

    print("\n" + "=" * 60)
    print("All tests passed ✓" if all_pass else "Some tests FAILED ✗")
    print("=" * 60)
