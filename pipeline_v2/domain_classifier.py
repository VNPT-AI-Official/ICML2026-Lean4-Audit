"""
domain_classifier.py — Classify sample domain from header imports for dynamic few-shot selection.

Domains:
  - Analysis:     Mathlib.Analysis.*, Mathlib.MeasureTheory.*, Mathlib.Topology.*
  - Algebra:      Mathlib.Algebra.*, Mathlib.RingTheory.*, Mathlib.FieldTheory.*, Mathlib.LinearAlgebra.*
  - NumberTheory: Mathlib.NumberTheory.*, Mathlib.Data.Nat.*, Mathlib.Data.Int.*
  - Combinatorics: Mathlib.Combinatorics.*, Mathlib.Data.Finset.*, Mathlib.Data.Fintype.*
  - Competition:  Minimal imports + open Real/Nat/Finset (competition-math style)
  - General:      Fallback
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Domain definitions — prefix patterns
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_PATTERNS: dict[str, list[str]] = {
    "Analysis": [
        "Mathlib.Analysis.",
        "Mathlib.MeasureTheory.",
        "Mathlib.Topology.",
        "Mathlib.Order.Filter.",
    ],
    "Algebra": [
        "Mathlib.Algebra.",
        "Mathlib.RingTheory.",
        "Mathlib.FieldTheory.",
        "Mathlib.LinearAlgebra.",
        "Mathlib.GroupTheory.",
    ],
    "NumberTheory": [
        "Mathlib.NumberTheory.",
        "Mathlib.Data.Nat.",
        "Mathlib.Data.Int.",
        "Mathlib.Data.ZMod.",
    ],
    "Combinatorics": [
        "Mathlib.Combinatorics.",
        "Mathlib.Data.Finset.",
        "Mathlib.Data.Fintype.",
    ],
}

# Competition-math indicators (open statements typical of Olympiad/competition problems)
_COMPETITION_INDICATORS = [
    "open Real",
    "open Nat",
    "open Finset",
    "open Complex",
    "open BigOperators",
    "open scoped BigOperators",
]


def classify_domain(header: str, informal_statement: str = "") -> str:
    """
    Classify the mathematical domain of a sample based on its header imports.

    Args:
        header: Lean 4 header (imports + open statements)
        informal_statement: Informal problem statement (optional, for competition heuristics)

    Returns:
        One of: "Analysis", "Algebra", "NumberTheory", "Combinatorics", "Competition", "General"
    """
    if not header:
        return "General"

    header_lines = header.strip()

    # 1. Check specific domain patterns (first match wins by priority)
    domain_scores: dict[str, int] = {}
    for domain, patterns in _DOMAIN_PATTERNS.items():
        score = sum(1 for pat in patterns if pat in header_lines)
        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        # Return domain with highest match count
        return max(domain_scores, key=domain_scores.get)

    # 2. Check for competition-math style (minimal imports + open Real/Nat)
    has_competition_indicator = any(ind in header_lines for ind in _COMPETITION_INDICATORS)
    has_minimal_import = (
        "import Mathlib" in header_lines
        and header_lines.count("import") <= 2  # Only 1-2 import lines
    )

    if has_competition_indicator and has_minimal_import:
        return "Competition"

    # 3. Check informal statement for competition keywords
    if informal_statement:
        informal_lower = informal_statement.lower()
        competition_keywords = [
            "find the", "determine the", "compute", "calculate",
            "what is the", "how many", "prove that for all",
            "contest", "olympiad", "competition", "putnam", "imo",
        ]
        if has_minimal_import and any(kw in informal_lower for kw in competition_keywords):
            return "Competition"

    return "General"


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # (description, header, informal, expected_domain)
        (
            "Analysis: MeasureTheory import",
            "import Mathlib\nimport Mathlib.MeasureTheory.Measure.Lebesgue\nopen MeasureTheory",
            "Prove the monotone convergence theorem.",
            "Analysis",
        ),
        (
            "Algebra: RingTheory import",
            "import Mathlib\nimport Mathlib.RingTheory.Polynomial.Basic\nopen Polynomial",
            "Show that the ring of integers is a PID.",
            "Algebra",
        ),
        (
            "NumberTheory: Nat.Prime",
            "import Mathlib\nimport Mathlib.Data.Nat.Prime.Basic",
            "Prove there are infinitely many primes.",
            "NumberTheory",
        ),
        (
            "Combinatorics: Finset import",
            "import Mathlib\nimport Mathlib.Data.Finset.Basic\nimport Mathlib.Data.Fintype.Card",
            "Count the number of subsets.",
            "Combinatorics",
        ),
        (
            "Competition: open Real + minimal import",
            "import Mathlib\nopen Real",
            "Find the minimum of (xy)^2 + (x+7)^2.",
            "Competition",
        ),
        (
            "Competition: open Nat + contest keyword",
            "import Mathlib\nopen Nat",
            "Determine the number of solutions to the equation.",
            "Competition",
        ),
        (
            "General: bare Mathlib import",
            "import Mathlib",
            "Prove that f is continuous.",
            "General",
        ),
    ]

    print("=" * 60)
    print("Domain Classifier — Self Tests")
    print("=" * 60)
    all_pass = True
    for desc, header, informal, expected in tests:
        got = classify_domain(header, informal)
        ok = got == expected
        if not ok:
            all_pass = False
        print(f"\n{'✓' if ok else '✗'} [{desc}]")
        print(f"    Domain: {got} (expected: {expected})")
    print("\n" + "=" * 60)
    print("All tests passed ✓" if all_pass else "Some tests FAILED ✗")
