"""
evaluate.py — Local evaluation of predictions against a labeled validation set.

Usage:
    python evaluate.py --predictions predictions.jsonl --gold validation.jsonl

Output metrics:
  - Verdict F1 (macro)
  - Category F1 (macro, on misaligned only)
  - Localization accuracy (exact match + LLM-judge)
  - Correction accuracy (exact match + LLM-judge)
  - Joint accuracy (all 4 correct simultaneously)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> float:
    """Compute macro-averaged F1 over a list of labels."""
    if labels is None:
        labels = sorted(set(y_true + y_pred))

    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0


def exact_match(pred: str, gold: str) -> bool:
    """Exact string match after stripping whitespace."""
    return pred.strip() == gold.strip()


def normalize_lean(s: str) -> str:
    """Normalize Lean 4 code for comparison (collapse whitespace)."""
    import re
    return re.sub(r"\s+", " ", s.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> dict[str, dict]:
    """Load predictions.jsonl → {idx: flat_dict} using Codabench submission format."""
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj["idx"]
            preds[idx] = obj
    return preds


def load_gold(path: str) -> dict[str, dict]:
    """
    Load gold JSONL or JSON → {idx: flat_dict}.
    Supports flat submission format and old nested diagnosis format.
    """
    result = {}
    from pathlib import Path
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    idx = obj.get("id") or obj.get("idx")
                    result[idx] = {
                        "idx": idx,
                        "verdict": obj.get("verdict", "misaligned"),
                        "error_category": obj.get("gt_error_category") or obj.get("error_category"),
                        "error_segment": obj.get("location_ground_truth") or obj.get("error_segment"),
                        "corrected_statement": obj.get("correction_ground_truth") or obj.get("corrected_statement"),
                    }
                return result

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj["idx"]
            # Handle both formats
            if "verdict" in obj:
                result[idx] = obj  # already flat
            elif "diagnosis" in obj:
                # Convert from nested to flat
                diag = obj["diagnosis"]
                aligned = diag.get("aligned", "Misaligned")
                result[idx] = {
                    "idx": idx,
                    "verdict": "aligned" if aligned == "Aligned" else "misaligned",
                    "error_category": diag.get("error_type") if aligned != "Aligned" else None,
                    "error_segment": diag.get("error_location") if aligned != "Aligned" else None,
                    "corrected_statement": diag.get("corrected_statement") if aligned != "Aligned" else None,
                }
            else:
                result[idx] = obj
    return result


def evaluate(pred_path: str, gold_path: str, verbose: bool = True) -> dict:
    preds = load_predictions(pred_path)
    golds = load_gold(gold_path)

    # Align on common idx
    common_ids = sorted(set(preds.keys()) & set(golds.keys()))
    if not common_ids:
        raise ValueError("No common idx found between predictions and gold file.")

    if verbose:
        print(f"Evaluating {len(common_ids)} samples "
              f"({len(preds)} predicted, {len(golds)} gold)")

    def _na(v) -> str:
        """Normalize null/None/'N/A' to a sentinel for comparison."""
        if v is None or v == "N/A" or v == "":
            return "__null__"
        return v

    # Check if gold has category information (at least one misaligned sample has non-null category)
    misaligned_ids = [i for i in common_ids if golds[i]["verdict"] == "misaligned"]
    has_category = any(golds[i].get("error_category") is not None for i in misaligned_ids)

    # ── Task 1: Verdict F1 — uses 'verdict' field (lowercase) ————————————
    y_true_verdict = [golds[i]["verdict"] for i in common_ids]
    y_pred_verdict = [preds[i].get("verdict", "misaligned") for i in common_ids]
    verdict_f1 = macro_f1(y_true_verdict, y_pred_verdict, labels=["aligned", "misaligned"])

    # ── Task 2: Category F1 (on misaligned gold only) ————————————————
    if misaligned_ids and has_category:
        y_true_cat = [_na(golds[i].get("error_category")) for i in misaligned_ids]
        y_pred_cat = [_na(preds[i].get("error_category")) for i in misaligned_ids]
        category_f1 = macro_f1(y_true_cat, y_pred_cat)
    else:
        category_f1 = 0.0

    # ── Task 3: Localization accuracy (exact match on error_segment) ———————
    loc_correct = 0
    for i in misaligned_ids:
        gold_loc = normalize_lean(_na(golds[i].get("error_segment", "")))
        pred_loc = normalize_lean(_na(preds[i].get("error_segment", "")))
        if exact_match(pred_loc, gold_loc):
            loc_correct += 1
    loc_acc = loc_correct / len(misaligned_ids) if misaligned_ids else 0.0

    # ── Task 4: Correction accuracy (exact match on corrected_statement) —————
    corr_correct = 0
    for i in misaligned_ids:
        gold_corr = normalize_lean(_na(golds[i].get("corrected_statement", "")))
        pred_corr = normalize_lean(_na(preds[i].get("corrected_statement", "")))
        if exact_match(pred_corr, gold_corr):
            corr_correct += 1
    corr_acc = corr_correct / len(misaligned_ids) if misaligned_ids else 0.0

    # ── Joint accuracy —————————————————————————————————————————
    joint_correct = 0
    for i in common_ids:
        g = golds[i]
        p = preds[i]
        verdict_ok = g["verdict"] == p.get("verdict", "misaligned")
        if g["verdict"] == "aligned":
            all_ok = verdict_ok
        else:
            if has_category:
                cat_ok = _na(g.get("error_category")) == _na(p.get("error_category"))
            else:
                cat_ok = True
            loc_ok = exact_match(
                normalize_lean(_na(g.get("error_segment", ""))),
                normalize_lean(_na(p.get("error_segment", ""))),
            )
            corr_ok = exact_match(
                normalize_lean(_na(g.get("corrected_statement", ""))),
                normalize_lean(_na(p.get("corrected_statement", ""))),
            )
            all_ok = verdict_ok and cat_ok and loc_ok and corr_ok
        if all_ok:
            joint_correct += 1
    joint_acc = joint_correct / len(common_ids) if common_ids else 0.0

    results = {
        "n_samples": len(common_ids),
        "n_misaligned_gold": len(misaligned_ids),
        "verdict_macro_f1": round(verdict_f1, 4),
        "category_macro_f1": round(category_f1, 4),
        "localization_acc_exact": round(loc_acc, 4),
        "correction_acc_exact": round(corr_acc, 4),
        "joint_accuracy": round(joint_acc, 4),
    }

    if verbose:
        print("\n" + "=" * 55)
        print("  FormalRx Evaluation Results")
        print("=" * 55)
        print(f"  Samples evaluated:      {results['n_samples']}")
        print(f"  Misaligned (gold):      {results['n_misaligned_gold']}")
        print(f"  Verdict Macro F1:       {results['verdict_macro_f1']:.4f}")
        print(f"  Category Macro F1:      {results['category_macro_f1']:.4f}")
        print(f"  Localization Acc:       {results['localization_acc_exact']:.4f}")
        print(f"  Correction Acc:         {results['correction_acc_exact']:.4f}")
        print(f"  Joint Accuracy:         {results['joint_accuracy']:.4f}")
        print("=" * 55)

        # Per-category breakdown
        if misaligned_ids and has_category:
            print("\n  Per-Category F1 (top errors):")
            from collections import Counter
            cat_counter = Counter(y_true_cat)
            for cat, count in cat_counter.most_common(10):
                tp = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == cat and p == cat)
                fp = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t != cat and p == cat)
                fn = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == cat and p != cat)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_cat = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                print(f"    {cat:<40s} n={count:4d}  F1={f1_cat:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate FormalRx predictions")
    ap.add_argument("--predictions", required=True, help="predictions.jsonl file")
    ap.add_argument("--gold", required=True, help="Gold labels JSONL file")
    ap.add_argument("--json-out", default=None, help="Save metrics as JSON")
    args = ap.parse_args()

    results = evaluate(args.predictions, args.gold, verbose=True)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {args.json_out}")
