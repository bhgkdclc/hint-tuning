# Hint Tuning baseline walkthrough

This note explains the paper and the pinned checkout as they are. It does not propose or implement a student-aware variant.

## What problem the method addresses

Reasoning models often produce long chains of thought for easy and hard problems alike. Hint Tuning turns the desired amount of reasoning into a data-construction decision: use an instruct model to test how much of a reasoning model's trace is needed to unlock a correct answer, then supervise the reasoning model with a response of the corresponding length.

There are three roles:

| Role | Paper baseline | Responsibility |
| --- | --- | --- |
| Reasoning model | `Qwen/Qwen3-4B-Thinking-2507` | Generates the original full reasoning trace; later receives SFT. |
| Instruct model / capability probe | `Qwen/Qwen3-4B-Instruct-2507` | Attempts the problem after receiving zero or more reasoning episodes as a prefix. |
| Grader | Qwen3-4B-Instruct served through an OpenAI-compatible vLLM endpoint | Compares a generated answer with the gold answer and returns a correctness decision. |

For reasoning episodes `e_1 ... e_N`, the paper defines the Minimum Effective Hint as

```text
K* = smallest k in {0, ..., N} for which
     M_instruct(problem + episodes[1:k]) is graded correct.
```

The resulting target states are:

| State | Paper criterion | Supervised content inside `<think>...</think>` |
| --- | --- | --- |
| No-Hint | `K*=0` | A short directive. |
| Sparse-Hint | `0<K*<N` | The first `K*` reasoning episodes. |
| Full-Hint | No tested prefix succeeds; represented as `K*=N` | The complete reasoning trace. |

Difficulty is therefore not a human label or a continuous score in the data. It is an ordinal, model-dependent probing result. A larger `K*` means the chosen instruct probe needed more of the reasoning trace. Generation sampling and LLM grading also make it an empirical result rather than an intrinsic property of the question.

## What “reasoning budget” means here

Hint Tuning has no per-example hard token-budget field and adds no special loss term. The effective budget is encoded in the length and state of each SFT target. Standard causal-language-model cross-entropy then teaches the reasoning model to imitate short, partial, or full reasoning responses.

The 32,768 values in `construction/config.yaml` are generation/context ceilings. The paper's separate “Budget Prompt” baseline asks a model to reason in approximately 8,000 tokens; that is a comparison method, not Hint Tuning's mechanism. During evaluation, reasoning overhead is measured as the token count inside the `<think>` tags.

## Repository pipeline

```text
data/problems.json
  │
  ├─ pipeline.py --mode think ───────► output/think_results.json
  │                                      │
  │                                      ├─ --mode grade
  │                                      │    └─► output/llm_grading_think.json
  │                                      │
  │                                      └─ --mode instruct
  │                                           └─► output/instruct_results.json
  │
  └─ think failures + grader + instruct probe
         └─ pipeline.py --mode prefix
              └─► output/<probe-name>/k_prefix.json

think + grading + instruct + prefix results
  └─ construction/merge.py
       └─► data/hint_tuning_1k.json (Alpaca format)
            └─► Relax full-parameter SFT
                 └─► Megatron distributed checkpoint
                      └─► HF checkpoint conversion
                           └─► vLLM inference through lighteval
                                └─► benchmark metrics and token counts
```

### 1. Raw problems and full-trace generation

`data/problems.json` contains 1,000 `problem`/`gold_answer` records. `construction/pipeline.py --mode think` loads the reasoning model with vLLM, applies its chat template, samples a response, and records generated-token counts.

```bash
python construction/pipeline.py \
  --mode think \
  --think-model Qwen/Qwen3-4B-Thinking-2507 \
  --dataset data/problems.json \
  --config construction/config.yaml \
  --output-dir output/
```

Expected output: `output/think_results.json`.

### 2. Grade the reasoning responses

The grader server is configured in `construction/instruct_models.yaml`. It must be running for `grade` and `prefix`; the README's example launches it on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --port 8001 --served-model-name grader
```

Contrary to the README comment at this point, `--mode grade` reads and grades `think_results.json`.

```bash
python construction/pipeline.py \
  --mode grade \
  --think-results output/think_results.json \
  --instruct-models-config construction/instruct_models.yaml \
  --output-dir output/
```

Expected output: `output/llm_grading_think.json`. In the checkout, only records graded as incorrect here are sent into prefix search.

### 3. Generate the no-prefix instruct responses

```bash
python construction/pipeline.py \
  --mode instruct \
  --instruct-model Qwen/Qwen3-4B-Instruct-2507 \
  --think-results output/think_results.json \
  --config construction/config.yaml \
  --output-dir output/
```

Expected output: `output/instruct_results.json`. The implementation does not grade this file before using responses for its No-Hint branch.

### 4. Segment traces and search for a prefix

`TextProcessor.split_by_reflection` splits a trace at case-insensitive reflection keywords. `create_cumulative_prefixes` creates `""`, `e_1`, `e_1+e_2`, and so on. Both local and API paths test prefixes in ascending `k` and stop at the first answer accepted by the grader. This is a linear early-stopping search, despite “binary search” wording in the paper pseudocode comment.

```bash
python construction/pipeline.py \
  --mode prefix \
  --think-results output/think_results.json \
  --think-grading output/llm_grading_think.json \
  --instruct-models-config construction/instruct_models.yaml \
  --config construction/config.yaml \
  --output-dir output/
```

Expected output for the configured local probe: `output/qwen3-4b-instruct/k_prefix.json`, plus resumable and slim JSON/JSONL files in the same directory. **Do not run this command with the checked-in `instruct_models.yaml` unchanged:** it also lists a `gpt-4o` API entry with a placeholder key, and the loop will try that model as well. For a formal Qwen-only run, use a copy of the config containing only the Qwen local entry and the local grader. The README's top-level `output/k_prefix.json` path does not match the writer.

The current code truncates the list to the first 40 segments before probing. Its configured keyword list omits `hmm`, although the paper appendix includes it. Preserve these behaviors for a checkout-faithful baseline and record them as source discrepancies.

### 5. Merge into SFT records

Use the real nested prefix path:

```bash
python construction/merge.py \
  --think output/think_results.json \
  --grading output/llm_grading_think.json \
  --instruct output/instruct_results.json \
  --prefix output/qwen3-4b-instruct/k_prefix.json \
  --output data/hint_tuning_1k.json
```

Each result is `{instruction, input, output}`. `input` is empty and `output` consists of a `<think>` block followed by the answer. `data/dataset_info.json` maps those fields to prompt/query/response for training.

### 6. SFT and checkpoint conversion

Training code lives in Relax rather than this repository. At pinned Relax commit `5f86ed96786022640c48143bddcb465eb4679682`, the paper-scale entry is:

```bash
bash examples/cot_compression/run_relax_qwen3_4b_thinking_sft_8xGPU.sh
```

The script expects `MODEL_DIR`, `PROMPT_DATA`, `RELAX`, and `TMP_DIR`; uses the Alpaca `instruction` and `output` fields; and launches eight actor GPUs through Ray. Its 32K BF16, full-parameter path produces a Megatron distributed checkpoint. The pinned converter is invoked as:

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/relax_checkpoint \
  --output-dir /path/to/exported_hf_checkpoint \
  --origin-hf-dir /path/to/original_hf_model
```

The last argument supplies the original model configuration and tokenizer. This is an A800/H20-class reproduction path, not the RTX 3060 smoke path.

### 7. Inference and evaluation

`evaluation/eval.sh` creates a lighteval vLLM model configuration and loads `evaluation/custom_tasks.py`. It evaluates AIME24, AIME25, HMMT25, and MATH-500 with this suffix:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

```bash
bash evaluation/eval.sh /path/to/hf_checkpoint output/eval_results
```

The checked-in script requests four-way tensor parallelism and a 32K context/output ceiling. Its intended artifacts are lighteval aggregate results and saved sample details; its custom metric counts generated tokens. The paper evaluates 16 sampled responses per problem at temperature 0.6 and top-p 0.95, while some checked-in task definitions request `n=1` or `n=8`, so a formal reproduction must record the exact evaluation revision and invocation.

## Paper-to-checkout discrepancies that must remain visible

| Topic | Paper/README description | Pinned implementation or released artifact |
| --- | --- | --- |
| No-Hint gate | Instruct model succeeds with no hint (`K*=0`). | `merge.py` assigns No-Hint when the reasoning model was graded correct, then uses an ungraded instruct response. |
| Successful prefix at `k=0` | No-Hint. | The successful-prefix branch emits the Sparse-Hint marker even for `k=0`. |
| Successful prefix at `k=N` | Paper's `0<K*<N` rule places this outside Sparse-Hint. | `merge.py` sends every successful prefix, including `k=N`, to Sparse-Hint. |
| Search input | Probe every problem to find `K*`. | `--think-grading` filters prefix search to reasoning-model failures. |
| Search wording | Pseudocode comment says binary search; loop is `k=0..N`. | Code performs a linear ascending scan with early stopping. |
| Prefix ceiling | Paper highlights that most successes occur within 25 episodes. | Code keeps at most 40 segments. |
| Reflection keywords | Appendix includes `hmm`. | `config.yaml` and `_DEFAULT_KW` omit it. |
| Prefix result path | README merge command uses `output/k_prefix.json`. | Writer uses `output/<probe-name>/k_prefix.json`. |
| Data size/state counts | Appendix reports 1,003 rows: 357/264/382. | Released pinned JSON has 1,000 rows: 322 No-Hint, 278 Sparse-Hint, 400 Full-Hint. |
| Training epochs | Paper describes ten epochs in its setup text. | Pinned 4B Relax script sets six epochs. |
| Evaluation samples | Paper states 16 responses per problem. | Checked-in custom tasks contain a mixture of `n=1` and `n=8`. |
| Reasoning-token metric | Paper counts tokens inside `<think>` tags. | The checked-in custom `output_tokens` metric counts the complete generated text. |

These are provenance findings, not requests to “fix” the algorithm. A checkout-faithful run must keep them. A paper-faithful reimplementation would be a separately named experiment after the official baseline works.

## Safe future modification boundary

For the later research phase, the narrowest conceptual entry is the capability-probing and state-assignment boundary: `_find_minimal_k_batch` / `_find_minimal_k_sequential` produce the per-problem probe result, and `merge_example` turns it into an SFT target. That observation only identifies the boundary. No change belongs there until the released-data smoke run and the official baseline are stable and recorded.
