"""
rule_classifier.py — Diff-based rule classifier for SCI error categorization.

Compares the broken formal_statement against the corrected_statement to detect
which SCI error category applies. This is a zero-cost alternative to LLM
classification for cases where structural differences are unambiguous.

Only rules with ≥90% precision on ground truth are enabled.

Returns:
    RuleClassification or None if no rule fires.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleClassification:
    """Result from rule-based classification."""
    category: str | None       # Canonical SCI category name, or None
    confidence: str            # "HIGH" | "MEDIUM" | "LOW"
    evidence: str              # Human-readable explanation
    rule_id: str               # Internal rule identifier


def _diff_tokens(formal: str, corrected: str) -> tuple[set[str], set[str]]:
    """Return (removed_tokens, added_tokens) between formal and corrected."""
    tok_pat = re.compile(r"[a-zA-Zα-ωΑ-Ω₀-₉][a-zA-Zα-ωΑ-Ω₀-₉_.]*|[^\s]")
    f_toks = set(tok_pat.findall(formal))
    c_toks = set(tok_pat.findall(corrected))
    return f_toks - c_toks, c_toks - f_toks


# ─────────────────────────────────────────────────────────────────────────────
# Diff-based rules (all ≥90% precision on GT)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_truncation(formal: str, corrected: str, informal: str) -> RuleClassification | None:
    """Formal is syntactically incomplete (unbalanced brackets, trailing := etc).
    Precision: 100%
    """
    nf = re.sub(r"\s+", " ", formal.strip())

    if any([
        nf.endswith("..."),
        nf.endswith(","),
        re.search(r":=\s*$", nf),
        nf.count("(") > nf.count(")"),
        nf.count("{") > nf.count("}"),
    ]):
        return RuleClassification("Truncation Error", "HIGH",
                                  "Formal statement appears truncated", "RD-TRUNC")
    return None


def _rule_quantifier_swap(formal: str, corrected: str, informal: str) -> RuleClassification | None:
    """∀ ↔ ∃ swap."""
    f_forall, f_exists = formal.count("∀"), formal.count("∃")
    c_forall, c_exists = corrected.count("∀"), corrected.count("∃")

    if f_forall > c_forall and c_exists > f_exists:
        return RuleClassification("Quantifier Strengthening", "HIGH",
                                  "Formal uses ∀ where ∃ is correct (∃→∀ strengthens)", "RD-QSTR")
    if f_exists > c_exists and c_forall > f_forall:
        return RuleClassification("Quantifier Weakening", "HIGH",
                                  "Formal uses ∃ where ∀ is correct (∀→∃ weakens)", "RD-QWKN")
    return None


def _rule_extremum_swap(formal: str, corrected: str, informal: str) -> RuleClassification | None:
    """IsLeast ↔ IsGreatest, IsMax → IsMin swap."""
    f_greatest = "IsGreatest" in formal
    f_least = "IsLeast" in formal
    c_greatest = "IsGreatest" in corrected
    c_least = "IsLeast" in corrected

    if f_greatest and c_least and not f_least:
        return RuleClassification("Extremum Concept Error", "HIGH",
                                  "Corrected replaces IsGreatest → IsLeast", "RD-EXTR-1")
    if f_least and c_greatest and not f_greatest:
        return RuleClassification("Extremum Concept Error", "HIGH",
                                  "Corrected replaces IsLeast → IsGreatest", "RD-EXTR-2")

    if ("IsMax" in formal or "isMax" in formal) and ("IsMin" in corrected or "isMin" in corrected):
        return RuleClassification("Extremum Concept Error", "HIGH",
                                  "Corrected replaces IsMax → IsMin", "RD-EXTR-3")
    return None


def _rule_connective_swap(formal: str, corrected: str, informal: str) -> RuleClassification | None:
    """∧ ↔ ∨, → → ↔ swaps."""
    removed, added = _diff_tokens(formal, corrected)

    if "∧" in removed and "∨" in added:
        return RuleClassification("Logical Connective Misuse", "HIGH",
                                  "Corrected replaces ∧ → ∨", "RD-CONN-1")
    if "∨" in removed and "∧" in added:
        return RuleClassification("Logical Connective Misuse", "HIGH",
                                  "Corrected replaces ∨ → ∧", "RD-CONN-2")
    if "→" in removed and "↔" in added:
        return RuleClassification("Logical Connective Misuse", "HIGH",
                                  "Corrected replaces → → ↔", "RD-CONN-3")
    return None


def _rule_function_swap(formal: str, corrected: str, informal: str) -> RuleClassification | None:
    """Named function swap detection."""
    func_pairs = [
        # Trig functions
        ("Real.sin", "Real.cos"), ("Real.cos", "Real.sin"),
        ("Real.sin", "Real.tan"), ("Real.tan", "Real.sin"),
        ("Real.cos", "Real.tan"), ("Real.tan", "Real.cos"),
        # Log/Exp
        ("Real.log", "Real.exp"), ("Real.exp", "Real.log"),
        ("Complex.log", "Real.log"), ("Real.log", "Complex.log"),
        # Combinatorics
        ("Nat.factorial", "Nat.choose"), ("Nat.choose", "Nat.factorial"),
        # Monotonicity concepts
        ("Monotone", "Antitone"), ("Antitone", "Monotone"),
        ("StrictMono", "Monotone"), ("Monotone", "StrictMono"),
        ("StrictAnti", "Antitone"), ("Antitone", "StrictAnti"),
        # Filter/limit concepts
        ("atTop", "atBot"), ("atBot", "atTop"),
        # Metric spaces
        ("EMetric.diam", "Metric.diam"), ("Metric.diam", "EMetric.diam"),
        # Continuity/Analyticity
        ("ContinuousAt", "AnalyticAt"), ("AnalyticAt", "ContinuousAt"),
    ]
    for old, new in func_pairs:
        if old in formal and new in corrected and old not in corrected:
            return RuleClassification("Function Confusion", "HIGH",
                                      f"Corrected replaces {old} → {new}", "RD-FUNC")
    return None


def _rule_precedence_fix(formal: str, corrected: str, informal: str) -> RuleClassification | None:
    """Detect parenthesization: a * b ^ n → (a * b) ^ n."""
    prec_pat = re.compile(r"[a-zA-Zα-ωΑ-Ω₀-₉_]+\s*\*\s*[a-zA-Zα-ωΑ-Ω₀-₉_]+\s*\^\s*\d+")
    paren_pat = re.compile(r"\([a-zA-Zα-ωΑ-Ω₀-₉_]+\s*\*\s*[a-zA-Zα-ωΑ-Ω₀-₉_]+\)\s*\^\s*\d+")
    if prec_pat.search(formal) and paren_pat.search(corrected):
        return RuleClassification("Operator Precedence Error", "HIGH",
                                  "Corrected adds parentheses around product before power",
                                  "RD-PREC")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rule registry
# ─────────────────────────────────────────────────────────────────────────────

_DIFF_RULES = [
    _rule_truncation,
    _rule_quantifier_swap,
    _rule_extremum_swap,
    _rule_connective_swap,
    _rule_function_swap,
    _rule_precedence_fix,
]


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def classify_by_diff(
    informal_statement: str,
    formal_statement: str,
    corrected_statement: str,
) -> RuleClassification | None:
    """
    Attempt to classify the SCI error category by comparing the broken
    formal_statement against the corrected_statement.

    Returns the first HIGH-confidence match. Returns None if no rule fires.
    """
    if not corrected_statement or corrected_statement.strip() in ("N/A", "null", "none", ""):
        return None

    for rule in _DIFF_RULES:
        result = rule(formal_statement, corrected_statement, informal_statement)
        if result is not None:
            return result

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Quantifier Strengthening",
         "there exists a prime > 100",
         "theorem t : ∀ p : ℕ, Nat.Prime p → p > 100 := by sorry",
         "theorem t : ∃ p : ℕ, Nat.Prime p ∧ p > 100 := by sorry",
         "Quantifier Strengthening"),
        ("Extremum IsGreatest→IsLeast",
         "minimum of nonneg",
         "theorem t : IsGreatest {n : ℕ | 0 ≤ n} 0 := by sorry",
         "theorem t : IsLeast {n : ℕ | 0 ≤ n} 0 := by sorry",
         "Extremum Concept Error"),
        ("Connective ∧→∨",
         "even or odd",
         "theorem t (n : ℕ) : Even n ∧ ¬Even n := by sorry",
         "theorem t (n : ℕ) : Even n ∨ ¬Even n := by sorry",
         "Logical Connective Misuse"),
        ("Function sin→cos",
         "sin(x) <= 1",
         "theorem t (x : ℝ) : Real.cos x ≤ 1 := by sorry",
         "theorem t (x : ℝ) : Real.sin x ≤ 1 := by sorry",
         "Function Confusion"),
        ("Precedence x*y^2 → (x*y)^2",
         "find min of (xy)^2",
         "theorem t : IsLeast {n | ∃ x y : ℝ, n = x * y ^ 2} 45 := by sorry",
         "theorem t : IsLeast {n | ∃ x y : ℝ, n = (x * y) ^ 2} 45 := by sorry",
         "Operator Precedence Error"),
        ("Truncation",
         "x^2 >= 0",
         "theorem t (x : ℝ) : x ^ 2 ≥ 0 :=",
         "theorem t (x : ℝ) : x ^ 2 ≥ 0 := by sorry",
         "Truncation Error"),
    ]

    print("=" * 60)
    print("Rule Classifier — Self Tests")
    print("=" * 60)
    all_pass = True
    for desc, informal, formal, corrected, expected_cat in tests:
        result = classify_by_diff(informal, formal, corrected)
        got = result.category if result else None
        ok = got == expected_cat
        if not ok:
            all_pass = False
        print(f"\n{'✓' if ok else '✗'} [{desc}]")
        if result:
            print(f"    [{result.confidence}] {result.category}: {result.evidence}")
        else:
            print("    (no classification)")
        if not ok:
            print(f"    EXPECTED: {expected_cat}")
            print(f"    GOT:      {got}")

    print("\n" + "=" * 60)
    print("All tests passed ✓" if all_pass else "Some tests FAILED ✗")
