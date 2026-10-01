"""
SCI Error Taxonomy — 28 categories across 3 dimensions.

Reference: FormalRx paper (Wang et al., 2025)
Diagram: https://cdn-uploads.huggingface.co/production/uploads/68ac4471a1f07af43fc13153/SPKTUMGhTjMyW_CGlP4qf.png
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class ErrorCategory:
    code: str       # e.g. "S1.1"
    name: str       # Canonical name used in dataset
    dimension: str  # "Semantic" | "Constraint" | "Implementation"
    group: str      # e.g. "Logical Structure Error"
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Full 28-category SCI taxonomy
# ─────────────────────────────────────────────────────────────────────────────
TAXONOMY: list[ErrorCategory] = [
    # ── S1: Logical Structure Errors ──────────────────────────────────────────
    ErrorCategory(
        code="S1.1",
        name="Quantifier Strengthening",
        dimension="Semantic",
        group="Logical Structure Error",
        description=(
            "Quantifiers are modified in a way that strengthens the statement. "
            "Examples include replacing an existential quantifier (∃) with a "
            "universal quantifier (∀), or expanding the quantifier domain/scope "
            "(e.g., 'first n terms' → 'all terms', finite domain → infinite domain). "
            "The resulting formal statement is stronger than the intended meaning. "
            "NOTE: If ∃→∀ also changes → to ∧, classify the error as S1.1 rather "
            "than S1.3 because quantifier strengthening is the primary cause."
        ),
    ),
    ErrorCategory(
        code="S1.2",
        name="Quantifier Weakening",
        dimension="Semantic",
        group="Logical Structure Error",
        description=(
            "Quantifiers are modified in a way that weakens the statement. "
            "Examples include replacing a universal quantifier (∀) with an "
            "existential quantifier (∃), or reducing the quantifier domain/scope "
            "(e.g., 'all terms' → 'first n terms', infinite domain → finite domain). "
            "The resulting formal statement is weaker than the intended meaning. "
            "NOTE: If ∀→∃ also changes ∧ to →, classify the error as S1.2 rather "
            "than S1.3 because quantifier weakening is the primary cause."
        ),
    ),
    ErrorCategory(
        code="S1.3",
        name="Logical Connective Misuse",
        dimension="Semantic",
        group="Logical Structure Error",
        description=(
            "Logical connectives are used incorrectly, including conjunction (∧), "
            "disjunction (∨), implication (→), biconditional (↔), negation (¬), "
            "or related logical operators. Examples include swapping ∧ and ∨, "
            "using → instead of ↔, reversing implication direction, or introducing "
            "an incorrect negation. Equivalence-preserving transformations "
            "(e.g., ¬(a ∨ b) ↔ (¬a) ∧ (¬b)) are not considered errors."
        ),
    ),

    # ── S2: Mathematical Object Errors ────────────────────────────────────────
    ErrorCategory(
        code="S2.1",
        name="Object Type Error",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "The mathematical object is assigned an incorrect type or algebraic "
            "structure that changes its essential properties. Examples include "
            "replacing one number system with another (e.g., ℤ → ℕ), changing "
            "group/ring/field assumptions, or modifying algebraic hierarchy "
            "constraints (e.g., IntegralDomain, PID, UFD). The resulting statement "
            "relies on a different mathematical structure than intended."
        ),
    ),
    ErrorCategory(
        code="S2.2",
        name="Function Confusion",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "One mathematical function is incorrectly substituted for another, "
            "changing the operation being performed. Examples include confusing "
            "trigonometric functions (sin → cos), number-theoretic functions "
            "(gcd → lcm), logarithms with different bases, namespace variants "
            "(e.g., Nat.log → Real.log), or other named functions with distinct "
            "mathematical meanings. This category applies to functions rather than "
            "operators, logical connectives, or relations."
        ),
    ),
    ErrorCategory(
        code="S2.3",
        name="Operator Confusion",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "Mathematical operators are incorrectly substituted or modified, "
            "changing the operation being performed. Examples include confusing "
            "arithmetic operators (+, −, ×, ÷), modular arithmetic operators, "
            "set operators (∪, ∩, \\ , ∆), or altering operator precedence "
            "through incorrect parentheses. This category does not include "
            "logical connectives, mathematical functions, relations, or "
            "exponent/power errors."
        ),
    ),
    ErrorCategory(
        code="S2.4",
        name="Exponent/Power Error",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "Exponents or powers are specified incorrectly, changing the "
            "mathematical function or expression structure. Examples include "
            "using x² instead of x³, replacing a^n with a^(n+1), polynomial "
            "degree errors, exponential sequence errors, or other mistakes in "
            "power expressions and formulas."
        ),
    ),
    ErrorCategory(
        code="S2.5",
        name="Coefficient/Constant Error",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "Numerical coefficients or constants are incorrect, changing the "
            "value or scaling of an expression. Examples include 2π → π, "
            "2x → 3x, x + 1 → x + 2, or incorrect unit conversions "
            "(e.g., 45° used instead of π/4 radians)."
        ),
    ),
    ErrorCategory(
        code="S2.6",
        name="Index/Subscript Error",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "Indices or subscripts are incorrect, causing a different element "
            "of a sequence, array, or indexed collection to be referenced. "
            "Examples include aₙ → aₙ₊₁, xᵢ → xⱼ, or incorrect array indices."
        ),
    ),
    ErrorCategory(
        code="S2.7",
        name="Partial Order Errors",
        dimension="Semantic",
        group="Mathematical Object Error",
        description=(
            "Statements involving partial-order relations are misinterpreted. "
            "Examples include reversing, loosening, or tightening relations such "
            "as <, ≤, >, ≥, |, and ⊆."
        ),
    ),

    # ── S3: Mathematical Concept Errors ───────────────────────────────────────
    ErrorCategory(
        code="S3.1",
        name="Infinity Misinterpretation",
        dimension="Semantic",
        group="Mathematical Concept Error",
        description=(
            "Statements involving infinitude are misinterpreted. Examples include "
            "confusing 'infinitely many' with 'all', or treating an infinite set "
            "as countably infinite when it may be uncountable."
        ),
    ),
    ErrorCategory(
        code="S3.2",
        name="Extremum Concept Error",
        dimension="Semantic",
        group="Mathematical Concept Error",
        description=(
            "Extremum concepts are misinterpreted by confusing bounds with "
            "attained extrema. Examples include IsLeast → x ≥ c and "
            "IsGreatest → x ≤ c."
        ),
    ),
    ErrorCategory(
        code="S3.3",
        name="Cardinality Error",
        dimension="Semantic",
        group="Mathematical Concept Error",
        description=(
            "Set cardinality is misinterpreted or expressed incorrectly. "
            "Examples include using existence or membership conditions in place "
            "of cardinality constraints, or otherwise failing to represent the "
            "size of a finite or infinite set correctly."
        ),
    ),
    ErrorCategory(
        code="S3.4",
        name="Integration/Differentiation Confusion",
        dimension="Semantic",
        group="Mathematical Concept Error",
        description=(
            "The relationship between integration and differentiation is "
            "misinterpreted. Examples include confusing an antiderivative with a "
            "definite integral, or expressing an antiderivative using an "
            "integral value instead of a derivative relation."
        ),
    ),
    ErrorCategory(
        code="S3.5",
        name="Geometric Relationship/Object Error",
        dimension="Semantic",
        group="Mathematical Concept Error",
        description=(
            "Errors in selecting or expressing geometric objects, relationships, "
            "or constructions. Examples include confusing different geometric "
            "entities, relationships, or referenced points, lines, and angles."
        ),
    ),

    # ── C1: Variable Constraint Errors ────────────────────────────────────────
    ErrorCategory(
        code="C1.1",
        name="Positivity Constraint Error",
        dimension="Constraint",
        group="Variable Constraint Error",
        description=(
            "Positivity-related constraints on variables are missing, redundant, "
            "or incorrect. Examples include 'x is positive' → missing x > 0, "
            "'n is non-zero' → missing n ≠ 0, or using x ≥ 0 instead of x > 0."
        ),
    ),
    ErrorCategory(
        code="C1.2",
        name="Bound Constraint Error",
        dimension="Constraint",
        group="Variable Constraint Error",
        description=(
            "Upper or lower bound constraints on variables are missing, redundant, "
            "or incorrect. Examples include p > 1 → p ≥ 1, n ≤ 100 → n < 100, "
            "or changing n < 10 to n < 100."
        ),
    ),
    ErrorCategory(
        code="C1.3",
        name="Domain Constraint Error",
        dimension="Constraint",
        group="Variable Constraint Error",
        description=(
            "Variables or expressions are assigned an incorrect domain or type. "
            "Examples include f : ℚ → ℚ instead of f : ℝ → ℝ, or using Ring R "
            "instead of Semiring R."
        ),
    ),
    ErrorCategory(
        code="C1.4",
        name="Variable Constraint Error",
        dimension="Constraint",
        group="Variable Constraint Error",
        description=(
            "Variable constraints are missing, redundant, or incorrect and do "
            "not fall under positivity, bound, or domain constraint errors. "
            "Examples include distinctness, coprimality, parity, or independence "
            "conditions."
        ),
    ),

    # ── C2: Range Errors ──────────────────────────────────────────────────────
    ErrorCategory(
        code="C2.1",
        name="Range Shift",
        dimension="Constraint",
        group="Range Error",
        description=(
            "The index range of a summation, product, integral, or similar "
            "construction is shifted by a constant, resulting in a different but "
            "superficially similar range. Examples include i = 1..n → i = 0..n "
            "or Fin n → Fin (n + 1)."
        ),
    ),
    ErrorCategory(
        code="C2.2",
        name="Range Error",
        dimension="Constraint",
        group="Range Error",
        description=(
            "Index ranges are specified incorrectly, resulting in missing, extra, "
            "or unintended elements. Examples include off-by-one errors, "
            "incorrect interval boundaries, or replacing a finite range with an "
            "infinite one."
        ),
    ),

    # ── C3: Premise Errors ────────────────────────────────────────────────────
    ErrorCategory(
        code="C3.1",
        name="Missing Premise",
        dimension="Constraint",
        group="Premise Error",
        description=(
            "A necessary assumption or hypothesis is missing from the statement. "
            "Examples include omitted conditions such as b ≠ 0, Measurable f, "
            "or Nat.Prime p."
        ),
    ),
    ErrorCategory(
        code="C3.2",
        name="Redundant Premise",
        dimension="Constraint",
        group="Premise Error",
        description=(
            "Unnecessary or redundant assumptions are added to the statement. "
            "Examples include adding CompactSpace (ℕ → β) or Monoid R when not required."
        ),
    ),
    ErrorCategory(
        code="C3.3",
        name="Incorrect Premise",
        dimension="Constraint",
        group="Premise Error",
        description=(
            "A hypothesis is present but mathematically incorrect. Examples "
            "include (h : p.leadingCoeff * a = 0) instead of ≠ 0, or using "
            "Ring R instead of Semiring R."
        ),
    ),

    # ── C4: Conclusion Error ──────────────────────────────────────────────────
    ErrorCategory(
        code="C4",
        name="Conclusion Error",
        dimension="Constraint",
        group="Conclusion Error",
        description=(
            "The goal or conclusion is incorrectly formalized. Example includes "
            "using an incorrect logical or mathematical statement for the result "
            "to be proved."
        ),
    ),

    # ── C5: Auxiliary Construction Error ─────────────────────────────────────
    ErrorCategory(
        code="C5",
        name="Auxiliary Construction Error",
        dimension="Constraint",
        group="Auxiliary Construction Error",
        description=(
            "Errors in auxiliary constructions supporting the main result. "
            "Examples include incorrectly defined helper functions, witness "
            "objects, or intermediate constructions with wrong parameters or "
            "constraints."
        ),
    ),

    # ── I: Implementation Errors ──────────────────────────────────────────────
    ErrorCategory(
        code="I1",
        name="Truncation Error",
        dimension="Implementation",
        group="Implementation Error",
        description=(
            "Parts of the formal statement are cut off or truncated during formalization, "
            "resulting in an incomplete Lean 4 expression with unbalanced brackets or missing tokens."
        ),
    ),
    ErrorCategory(
        code="I2",
        name="Operator Precedence Error",
        dimension="Implementation",
        group="Implementation Error",
        description=(
            "Incorrect grouping of expressions due to missing or incorrect parentheses, "
            "leading to unintended operator precedence and changed semantics. "
            "Example: x * y ^ 2 interpreted differently depending on parentheses."
        ),
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_NAMES: list[str] = [c.name for c in TAXONOMY]
CATEGORY_NAMES_LOWER: dict[str, str] = {c.name.lower(): c.name for c in TAXONOMY}
CODE_TO_CATEGORY: dict[str, ErrorCategory] = {c.code: c for c in TAXONOMY}
NAME_TO_CATEGORY: dict[str, ErrorCategory] = {c.name: c for c in TAXONOMY}




def normalize_category(raw: str) -> str | None:
    """
    Normalize a potentially noisy LLM-generated category name to the canonical name.
    Returns None if no match found.

    Resolution order:
      1. Exact match
      2. Case-insensitive exact match
      3. Code extraction via regex (e.g. "S2.4", "[S2.4]")
      4. Substring name match
      5. SCI code match
    """
    if not raw:
        return None
    stripped = raw.strip()

    # 1. Exact match
    if stripped in NAME_TO_CATEGORY:
        return stripped

    # 2. Case-insensitive exact match
    lower = stripped.lower()
    if lower in CATEGORY_NAMES_LOWER:
        return CATEGORY_NAMES_LOWER[lower]

    # 3. Regex match for SCI code (e.g. "S2.4", "C3", etc.)
    import re
    match = re.search(r'\b([SIC]\d+(?:\.\d+)?)\b', stripped)
    if match:
        code = match.group(1)
        if code in CODE_TO_CATEGORY:
            return CODE_TO_CATEGORY[code].name

    # 4. Substring match for canonical names
    for cat_name in NAME_TO_CATEGORY:
        if cat_name.lower() in lower:
            return cat_name

    # 5. Code match (e.g., "S1.1", "C4", "I2")
    if stripped in CODE_TO_CATEGORY:
        return CODE_TO_CATEGORY[stripped].name

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy text block for prompt injection
# ─────────────────────────────────────────────────────────────────────────────

def get_taxonomy_prompt_block() -> str:
    """Return a compact, prompt-friendly description of all 28 categories."""
    lines = ["The SCI Error Taxonomy has 28 mutually exclusive error categories:\n"]
    current_dim = None
    for cat in TAXONOMY:
        if cat.dimension != current_dim:
            current_dim = cat.dimension
            lines.append(f"\n## {cat.dimension} Errors")
        lines.append(f"  [{cat.code}] {cat.name}: {cat.description}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Group-level helpers (for 2-call classification)
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of unique group names
GROUP_NAMES: list[str] = list(dict.fromkeys(c.group for c in TAXONOMY))

# Groups → categories mapping
_GROUP_TO_CATS: dict[str, list[ErrorCategory]] = {}
for _cat in TAXONOMY:
    _GROUP_TO_CATS.setdefault(_cat.group, []).append(_cat)

# Groups with only 1 category — Call 2 can be skipped
SINGLE_CATEGORY_GROUPS: dict[str, str] = {
    group: cats[0].name
    for group, cats in _GROUP_TO_CATS.items()
    if len(cats) == 1
}


def get_group_taxonomy_block(exclude_set: set[str] | None = None) -> str:
    """Return a compact description of the error groups (for Call 1)."""
    if exclude_set is None:
        exclude_set = set()
    active_groups = []
    for group in GROUP_NAMES:
        cats = _GROUP_TO_CATS[group]
        active_cats = [c for c in cats if c.name not in exclude_set]
        if active_cats:
            active_groups.append((group, active_cats))

    lines = [f"The SCI Error Taxonomy has {len(active_groups)} error groups:\n"]
    current_dim = None
    for group, cats in active_groups:
        dim = cats[0].dimension
        if dim != current_dim:
            current_dim = dim
            lines.append(f"\n## {dim} Errors")
        cat_codes = ", ".join(f"{c.code} {c.name}" for c in cats)
        lines.append(f"  **{group}** — includes: {cat_codes}")
    return "\n".join(lines)


def get_categories_for_group(group_name: str) -> list[ErrorCategory]:
    """Return all categories belonging to a group."""
    return _GROUP_TO_CATS.get(group_name, [])


def get_category_taxonomy_block(group_name: str, exclude_set: set[str] | None = None) -> str:
    """Return a detailed taxonomy block for categories within a single group (for Call 2)."""
    if exclude_set is None:
        exclude_set = set()
    cats = get_categories_for_group(group_name)
    active_cats = [c for c in cats if c.name not in exclude_set]
    if not active_cats:
        return f"No active categories within the **{group_name}** group."
    lines = [f"Categories within the **{group_name}** group:\n"]
    for cat in active_cats:
        lines.append(f"  [{cat.code}] {cat.name}: {cat.description}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Direct (1-call) classification prompt block — GT-optimized
# ─────────────────────────────────────────────────────────────────────────────

def get_direct_classify_prompt_block(exclude_set: set[str] | None = None) -> str:
    """
    Return a compact, GT-optimized taxonomy block for 1-call direct classification.
    Categories include embedded disambiguation notes from GT analysis.
    """
    if exclude_set is None:
        exclude_set = set()

    lines = ["## SCI Error Taxonomy\n"]
    current_group = None
    active_count = 0

    for cat in TAXONOMY:
        if cat.name in exclude_set:
            continue
        active_count += 1

        if cat.group != current_group:
            current_group = cat.group
            lines.append(f"\n**{cat.group}:**")

        lines.append(f"  - [{cat.code}] **{cat.name}**: {cat.description}")

    lines.append(f"\n\nTotal active categories: {active_count}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Total categories: {len(TAXONOMY)}")
    for c in TAXONOMY:
        print(f"  {c.code:5s} | {c.dimension:15s} | {c.name}")
    print(f"\nGroups: {len(GROUP_NAMES)}")
    for g in GROUP_NAMES:
        cats = _GROUP_TO_CATS[g]
        single = " [SINGLE]" if g in SINGLE_CATEGORY_GROUPS else ""
        print(f"  {g}: {len(cats)} categories{single}")

