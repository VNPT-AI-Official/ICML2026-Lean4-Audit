<div align="center">

# Retrieval-Augmented Hybrid Neuro-Symbolic Framework for Semantic Alignment Diagnosis in Lean Autoformalization

**ICML 2026 AI4Math Workshop — Track 1: Semantic Alignment Evaluation for Autoformalization (FormalRx)**

🥈 **2nd Place Winner**

[![Competition](https://img.shields.io/badge/Codabench-Competition-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4=)](https://www.codabench.org/competitions/16104/)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-red)](https://icml.cc/2026)
[![Lean 4](https://img.shields.io/badge/Lean-4-orange)](https://leanprover.github.io/)
[![Certificate](https://img.shields.io/badge/Award-Certificate-gold)](https://ai4math2026.github.io/assets/awards/challenges/track-1-second.pdf)

</div>

---

<div align="center">

### 🏆 🥈 2nd Place — ICML 2026 AI4Math Challenge

**Track 1: Semantic Alignment Evaluation for Autoformalization (FormalRx)**

[![📜 View Certificate](https://img.shields.io/badge/%F0%9F%93%9C_View_Certificate-PDF-blue?style=for-the-badge)](https://ai4math2026.github.io/assets/awards/challenges/track-1-second.pdf)

</div>

---

## 📋 Overview

Given a pair of *(informal math statement, Lean 4 formalization)*, the system diagnoses semantic alignment across four dimensions:

| Field | Description | Metric |
|---|---|---|
| `verdict` | `"aligned"` / `"misaligned"` | Macro F1 |
| `error_category` | One of the 28 SCI error categories ([→ full taxonomy](SCI_TAXONOMY.md)) | Macro F1 |
| `error_segment` | Minimal Lean 4 code snippet that is incorrect | Accuracy |
| `corrected_statement` | Full corrected Lean 4 statement | Accuracy |

- **7,030 test samples** (74.5 % misaligned, 25.5 % aligned)
- Error categories follow the **SCI taxonomy** — 3 dimensions, 28 fine-grained categories. See [SCI_TAXONOMY.md](SCI_TAXONOMY.md) for full details.

---

## 🏗️ Pipeline Architecture

The pipeline is a **multi-stage hybrid neuro-symbolic system** featuring self-consistency voting, merged correction + localization, and Mathlib search-reference injection.

![Pipeline Stages](assets/Pipeline%20Stages.jpg)

<details>
<summary><b>Stage-by-stage breakdown (text)</b></summary>

```
[HuggingFace: LARK-Lab/FormalRx-Test]
              ↓ 7,030 samples
  ┌───────────▼─────────────────────────────────────────┐
  │  STAGE 0 — Preprocessing & Enrichment               │
  │  • Semantic Search (lean_finder.py) for Mathlib refs│
  │  • Lean AST structure parsing (LSP / regex fallback)│
  └───────────┬─────────────────────────────────────────┘
              ↓
     [Check .checkpoint.jsonl]
              ↓ pending samples
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 1a — Verdict (Self-Consistency Voting)       │
  │  • N parallel votes (temperature=0.5 for diversity) │
  │  • Strategy: any_misaligned → 1 Misaligned vote     │
  │    triggers full diagnosis                          │
  │  • Mathlib search reference injected in prompt      │
  └─────────────────┬───────────────────────────────────┘
          aligned ↓             ↓ misaligned
        [done, skip]            │
                   ┌────────────▼────────────────────────┐
                   │  STAGE 1b — Correction + Localization│
                   │  • Single merged LLM call            │
                   │  • Returns corrected_statement AND   │
                   │    error_segment together            │
                   │  • Validates N/A output → retries   │
                   │    with increasing temperature       │
                   └────────────┬────────────────────────┘
                                │
  ┌─────────────────────────────▼───────────────────────┐
  │  STAGE 2 — Error Categorization                     │
  │  2a: Rule-based diff classifier (zero API cost)     │
  │  2b: [if rule matched] use localization from 1b     │
  │  2c: [if no rule] LLM direct 1-call classification  │
  │  • Supports N parallel votes for category SC        │
  └─────────────────────────────┬───────────────────────┘
                                │
  ┌─────────────────────────────▼───────────────────────┐
  │  LEAN COMPILER FEEDBACK (lean_verifier.py)          │
  │  • Verifies corrected_statement syntactically       │
  │  • If failed → feed error back to LLM (≤3 retries) │
  │  [Disabled by default: use_lean_compiler: false]    │
  └─────────────────────────────┬───────────────────────┘
                                │
  ┌─────────────────────────────▼───────────────────────┐
  │  POST-PROCESSING (postprocess.py) — zero API calls  │
  │  • Reclassify categories based on diff patterns     │
  │  • Validate correction structure (add sorry suffix) │
  │  • Canonicalize hallucinated category names         │
  │  • Final N/A field validation with warning logging  │
  └─────────────────────────────┬───────────────────────┘
                                ↓
  [Checkpoint] → predictions.jsonl → submission.zip
```

</details>

---

## 📊 Results

| Model | Overall | Verdict F1 | Category F1 | Locate | Rectify | Notes |
|---|---|---|---|---|---|---|
| Qwen3-235B | 0.38 | 0.73 | 0.30 | 0.67 | 0.76 | Baseline |
| Qwen3-235B | 0.43 | 0.76 | 0.35 | 0.74 | 0.78 | + Self-consistency |
| Qwen3-235B | 0.48 | 0.82 | 0.37 | 0.69 | 0.80 | + Search reference |
| DeepSeek v4 Pro | 0.50 | 0.86 | 0.47 | 0.70 | 0.85 | + Self-consistency + Search ref |
| **Claude Opus 4.8** | **0.77** | **0.93** | **0.72** | **0.93** | **0.97** | **Final submission** |

---

## 📂 Repository Structure

```
formalrx-challenge/
├── README.md                        # This file
├── SCI_TAXONOMY.md                  # Full SCI Error Taxonomy (28 categories)
├── EVALUATION_PROTOCOL.md           # Evaluation protocol documentation
├── assets/
│   ├── Pipeline Stages.jpg          # Pipeline architecture diagram
│   ├── sci_taxonomy.png             # SCI Error Taxonomy diagram
│   ├── Certificates.pdf             # Award certificates
│   └── icml2026_final.pdf           # Camera-ready paper
├── datasets/
│   ├── FormalRx_Test.jsonl          # Input test samples (7,030)
│   ├── search_results.json          # Semantic search results (from lean_finder.py)
│   ├── enriched_dataset.jsonl       # Pre-processed Lean structure data (LSP)
│   └── predictions.jsonl            # Final predictions for submission
└── pipeline_v2/                     # Pipeline source code
    ├── config.yaml                  # API keys, model routing, concurrency, flags
    ├── pipeline.py                  # Main async runner (multi-stage + self-consistency)
    ├── prompt.py                    # Prompt builders: verdict / correction+loc / classify
    ├── taxonomy.py                  # SCI error categories + normalize_category()
    ├── parser.py                    # LLM output parsers (stage 1a, 1b, 2)
    ├── rule_classifier.py           # Rule-based diff classifier (zero-cost)
    ├── search_reference.py          # Mathlib search reference injection
    ├── lean_verifier.py             # Lean 4 compiler feedback loop
    ├── lean_parser.py               # Lean 4 AST structure parser
    ├── postprocess.py               # Post-processing: reclassification, validation
    ├── evaluate.py                  # Local evaluation metrics
    ├── utils.py                     # Logging, config, retry, checkpoint, rate limiter
    ├── requirements.txt             # Python dependencies
    └── run.sh                       # Shortcut run script
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GPU with ≥16 GB VRAM (for semantic search server)
- API key for the LLM provider (e.g., `ANTHROPIC_API_KEY`)

### 1. Install Dependencies

```bash
cd pipeline_v2
pip install -r requirements.txt
```

### 2. Semantic Search Server (Stage 0)

```bash
# Clone and install lean-finder
git clone https://github.com/delta-lab-ai/lean-finder.git
cd lean-finder
pip install -r requirements.txt

# Download corpus and FAISS indices
python download_corpus.py

# Start the server
python server.py
# Wait for "Handler ready — accepting requests" at http://localhost:8000

# Query the server (from pipeline_v2/)
cd ..
python lean_finder.py
```

### 3. Run the Pipeline

```bash
cd pipeline_v2/

# Test on first 50 samples
python3 pipeline.py --limit 50 --provider claude

# Full run — 7,030 samples
python3 pipeline.py --provider claude

# Resume after interruption
python3 pipeline.py --resume

# Use a specific config
python3 pipeline.py --config config.yaml --resume
```

### 4. Output Files

| File | Description |
|---|---|
| `predictions.jsonl` | Predictions in Codabench submission format |
| `predictions.debug.jsonl` | Predictions + reasoning + parse errors (debug) |
| `submission.zip` | **Upload this file to Codabench** |
| `.checkpoint.jsonl` | Checkpoint for resuming after interruption |
| `pipeline.log` | Full structured log with stage timings |

---

## ⚙️ Configuration

All settings are managed through [`pipeline_v2/config.yaml`](pipeline_v2/config.yaml):

```yaml
active_provider: "claude"

stage_providers:
  verdict: "claude"           # Stage 1a
  correction: "claude"        # Stage 1b
  category: "claude"          # Stage 2

providers:
  claude:
    base_url: "https://api.anthropic.com/v1"
    model: "claude-opus-4-8"

architecture:
  use_rule_classifier: true    # Diff-based symbolic classifier before LLM
  use_lean_compiler: false     # Lean 4 compiler feedback loop
  category_stages: 1           # 1 = direct, 2 = group→category (legacy)

self_consistency:
  stage1a_n_votes: 3
  stage1a_temperature: 0.5
  verdict_strategy: "any_misaligned"
  stage2_n_votes: 3
  stage2_temperature: 0.5

pipeline:
  max_concurrency: 10

generation:
  on_thinking: true
  temperature: 0.0
  max_tokens: 4096

search_reference:
  enabled: true
  ground_truth_file: "../datasets/search_results.json"
  score_threshold: 0.92
```

Environment variable overrides:
```bash
export ANTHROPIC_API_KEY="sk-xxx"
```

---

## 🧪 Testing

```bash
cd pipeline_v2/

# Unit test parsers (stage 1a, 1b, 2)
python3 parser.py

# Unit test post-processing rules
python3 postprocess.py

# Verify taxonomy normalization
python3 taxonomy.py

# Analyze rule classifier firing patterns
python3 analyze_rules.py
```

---

## 🔗 Links

- 📄 [Camera-Ready Paper (PDF)](assets/icml2026_final.pdf)
- 🥈 [Award Certificate (PDF)](https://ai4math2026.github.io/assets/awards/challenges/track-1-second.pdf)
- 🏆 [Codabench Competition](https://www.codabench.org/competitions/16104/)
- 📦 [FormalRx-Test Dataset](https://huggingface.co/datasets/LARK-Lab/FormalRx-Test)
- 📐 [Lean 4 / Mathlib](https://leanprover-community.github.io/)

---

## 📖 Citation

```bibtex
@inproceedings{formalrx2026,
  title     = {Retrieval-Augmented Hybrid Neuro-Symbolic Framework for Semantic Alignment Diagnosis in Lean Autoformalization},
  booktitle = {ICML 2026 AI4Math Workshop — Track 1: Semantic Alignment Evaluation for Autoformalization (FormalRx)},
  year      = {2026}
}
```
