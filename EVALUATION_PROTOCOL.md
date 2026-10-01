# Evaluation Protocol Document

## 1. Inference Pipeline Description

Our inference pipeline is a fully automated, multi-stage, hybrid neuro-symbolic system designed to maximize accuracy through self-consistency, semantic grounding, and symbolic post-processing. Each test sample flows through the following structured stages:

### Stage 0: Preprocessing & Enrichment
* **Semantic Search:** We use a local FAISS index over the Lean Mathlib corpus (`lean_finder.py`) to retrieve the most semantically similar Lean 4 declarations based on the provided informal mathematical statement. References with a similarity score strictly exceeding `0.92` (`score_threshold`) are dynamically injected into the prompts.
* **Lean AST Parsing:** We extract structural context from the formal statement using a Lean 4 AST parser. This parser leverages pre-processed LSP context when available, or falls back to robust regex-based structural extraction if LSP outputs are unavailable. 

### Stage 1a: Verdict Classification (Self-Consistency Voting)
* To determine if the formal statement is `aligned` or `misaligned`, we query the LLM `3` times in parallel (`stage1a_n_votes: 3`).
* We set the generation temperature to `0.5` (`stage1a_temperature`) to induce diverse reasoning paths across the parallel votes. The max token length is capped at `512` to force concise output.
* We employ an `any_misaligned` voting strategy: if at least `1` out of the 3 votes categorizes the sample as `misaligned`, the final verdict is `misaligned`. This strongly biases the pipeline toward high recall for formalization errors.
* If the consensus verdict is `aligned`, the pipeline halts for this sample and outputs `N/A` for the remaining fields.

### Stage 1b: Correction & Localization (Merged)
* For `misaligned` samples, a single merged LLM call produces both the `corrected_statement` and the exact `error_segment`.
* The base temperature for this call is `0.0` for deterministic generation, with `max_tokens` set to `4096`. 
* **Fallback Strategy:** If the model fails to output a valid correction or segment (e.g. producing `N/A`), the pipeline automatically triggers up to `3` retries. With each retry attempt, the pipeline incrementally scales up the temperature by `0.2` (capped at `0.7`) to encourage the model to break out of failure loops.

### Stage 2: Error Categorization
* **Stage 2a (Symbolic Rules):** We first pass the original formal statement and the newly generated corrected statement into a zero-cost rule-based diff classifier (`architecture.use_rule_classifier: true`). By analyzing explicit diffs, this deterministic script automatically assigns specific taxonomy categories (e.g., if `<` became `≤`, it assigns `Partial Order Errors`; if `sin` became `cos`, it assigns `Function Confusion`).
* **Stage 2c (LLM Classification):** If no symbolic rules are triggered, the LLM handles the 28-class taxonomy categorization directly via a 1-call classification schema (`category_stages: 1`).
  * We also employ self-consistency here: querying the LLM `3` times in parallel (`stage2_n_votes: 3`) at a temperature of `0.5` (`stage2_temperature`). The majority class is selected as the final predicted category.

### Stage 3: Validation and Post-processing
* **Concurrency and Robustness:** The pipeline limits global concurrency to `10` active samples (`max_concurrency`) to respect API rate limits. All LLM calls employ exponential backoff with a `5.0s` base delay and a `2.0` multiplier on failure.
* **Post-processing Heuristics:** Zero-cost logic strictly canonicalizes any hallucinated taxonomy categories back to their nearest valid string, ensures final proof structures have a `sorry` suffix to prevent Lean parser failures, and provides a final structural validation sweep.


## 2. External Tools, APIs, and Auxiliary Systems

* **Language Models:** The pipeline is completely model-agnostic but primarily routes all stages to commercial/open-weight models via REST endpoints. For the primary configuration, it connects to **Claude Opus 4.8** via the Anthropic API. 
* **Model Configuration:** 
  * `on_thinking: true` – We enable implicit chain-of-thought (reasoning tokens) for all calls where supported by the API.
  * `use_few_shot: true` – We supply exactly `2` dynamic in-context examples to the model during each phase.
  * JSON Output Mode – All outputs are coerced into strict JSON structures.
* **Semantic Search Engine (`lean_finder.py`):** An auxiliary local Python system that embeds informal math statements and queries a pre-computed local FAISS index of Mathlib declarations to find mathematical context.
* **Lean 4 Language Server Protocol (LSP):** Utilized offline/pre-inference to parse the AST of the Lean 4 statements, enriching the inputs with a pre-computed JSON structure containing hypotheses and conclusions.


## 3. Challenge Rules Adherence Confirmation

We hereby confirm that this submission strictly adheres to the ICML 2026 AI4Math Workshop Challenge 1 rules:
1. **Independent Prediction:** Each test row was processed and predicted completely independently. No context, state, or information of any kind was carried over or shared between different test samples.
2. **No Test Set Lookup:** The test inputs and the FormalRx dataset were strictly excluded from any retrieval systems. Our semantic search engine exclusively queries the public, canonical Lean 4 Mathlib corpus. It absolutely does not retrieve from, or look up, any samples in the test or training datasets at inference time.
3. **No Manual Annotation:** The entire pipeline operates fully autonomously from start to finish. No test samples were manually annotated, adjusted, or reviewed to produce the final submitted predictions.
