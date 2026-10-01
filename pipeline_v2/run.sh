#!/bin/bash
# run.sh — Full FormalRx Pipeline V2 (end-to-end)
#
# Stage 0  — Semantic search:  lean_finder.py
#              Reads  ../datasets/FormalRx_Test.jsonl
#              Writes ../datasets/search_results.json
#
# Stage 0b — Lean AST enrichment: preprocess_lsp.py
#              Reads  ../datasets/FormalRx_Test.jsonl
#              Writes ../datasets/enriched_dataset.jsonl
#
# Stage 1a — Verdict × 3 self-consistency  ┐
# Stage 1b — Correction + Localization     ├── pipeline.py
# Stage 2  — SCI categorize × 3 SC        ┘
#              Reads  ../datasets/FormalRx_Test.jsonl (via config)
#              Reads  ../datasets/search_results.json  (search ref)
#              Reads  ../datasets/enriched_dataset.jsonl (lean struct)
#              Writes predictions.jsonl + submission.zip

PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Starting FormalRx Pipeline V2 (End-to-End)"
echo "  Stage 0a  : Semantic Search (lean_finder.py)"
echo "  Stage 0b : Lean AST Enrichment (preprocess_lsp.py)"
echo "  Stage 1 : Verdict x3 self-consistency"
echo "  Stage 2 : Correction + Localization"
echo "  Stage 3  : SCI Categorize x3 self-consistency"
echo "Python environment: $PYTHON"
echo "Pipeline directory: $SCRIPT_DIR"
echo "=============================================="
echo ""

cd "$SCRIPT_DIR" || exit 1

# ── Stage 0: Semantic search ──────────────────────────────────────────────────
if [ "${SKIP_SEARCH:-0}" != "1" ]; then
    echo "[Stage 0] Running semantic search..."
    $PYTHON lean_finder.py
    echo "[Stage 0] Done -> ../datasets/search_results.json"
else
    echo "[Stage 0] SKIPPED (SKIP_SEARCH=1)"
fi

echo ""

# ── Stage 0b: Lean AST enrichment ────────────────────────────────────────────
if [ "${SKIP_ENRICH:-0}" != "1" ]; then
    echo "[Stage 0b] Running Lean AST enrichment..."
    $PYTHON preprocess_lsp.py --stats
    echo "[Stage 0b] Done -> ../datasets/enriched_dataset.jsonl"
else
    echo "[Stage 0b] SKIPPED (SKIP_ENRICH=1)"
fi

echo ""

# ── Stages 1a / 1b / 2: LLM pipeline ─────────────────────────────────────────
echo "[Stage 1a/1b/2] Running LLM pipeline..."
$PYTHON pipeline.py "$@"

echo ""
echo "Pipeline complete. Submit submission.zip to Codabench."
