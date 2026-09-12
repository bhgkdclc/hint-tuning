# Hint Tuning baseline: source and environment audit

Audit date: 2026-09-12. This note records facts observed before installing packages or running training. It makes no changes to the upstream method.

## Sources and Git state

- Paper: [Hint Tuning: Less Data Makes Better Reasoners, arXiv:2605.08665v2](https://arxiv.org/html/2605.08665v2).
- Official code: [redai-studio/hint-tuning](https://github.com/redai-studio/hint-tuning). The URL printed in the paper, `redai-infra/hint-tuning`, redirects to it.
- Local upstream commit: `6aac81943b2649af76ce5c2bb3be6f63347a2cf4` (2026-07-08). `main` is the unmodified official commit; `reproduce` starts at the same commit.
- The checkout was obtained directly from upstream because GitHub was not signed in and `bhgkdclc/hint-tuning` was not publicly available. The official remote is named `upstream`; `origin` will be assigned to the user's fork after it exists.
- Local `core.autocrlf=false` and a fresh checkout give LF shell scripts. This was necessary because the original Windows Git checkout used CRLF and `bash -n evaluation/eval.sh` failed under WSL. No tracked content changed.
- The user's fork now exists at [bhgkdclc/hint-tuning](https://github.com/bhgkdclc/hint-tuning). It is configured as `origin`; `origin/main`, local `main`, and `upstream/main` all point to `6aac819`. `origin/reproduce` contains this audit history.

## Local environment

| Item | Observed state | Consequence |
| --- | --- | --- |
| WSL | Ubuntu 22.04.5, WSL2, kernel 6.18.33.2 | Suitable Linux target; no reinstall indicated. |
| Python | WSL system Python 3.10.12; `venv` present; `pip` and `torch` absent | Build an isolated environment only after choosing compatible wheels and cache location. |
| GPU | RTX 3060 Laptop, compute capability 8.6, 6144 MiB VRAM; about 5.3 GiB free at inspection | Cannot use the paper's multi-GPU, 32K-token commands directly. |
| CUDA | WSL `nvidia-smi` works; driver reports CUDA UMD 13.4; `nvcc` absent | GPU passthrough works. `nvcc` is not needed to begin with prebuilt PyTorch wheels. PyTorch CUDA support is still untested. |
| Memory | WSL 7.7 GiB RAM and 2 GiB swap | Keep local jobs small; avoid concurrent models. |
| Disk | E: ~91 GiB free; C: ~7.3 GiB free; WSL `/` reports ~944 GiB free on a virtual filesystem | Check where the WSL VHDX and Hugging Face/pip caches actually reside before downloading weights. Do not infer 944 GiB physical free space. |

The Ubuntu VHDX is physically at `E:\WSL\Ubuntu-22.04\ext4.vhdx` (about 15.1 GB allocated during inspection), so its growth consumes E: rather than C:. WSL currently runs as `root`; `/root/.cache/pip` already uses about 2.8 GB. Cache paths should still be made explicit for reproducible cleanup.

The initial sandboxed `wsl` query returned access denied; the same read-only queries succeeded with elevated execution. This was an execution-permission issue, not evidence of a broken WSL installation.

## Repository pipeline, as implemented

1. `data/problems.json`: 1,000 records with `problem` and `gold_answer`, from s1K. `construction/pipeline.py --mode think` converts these to question/solution records, runs the thinking model through vLLM, and writes `output/think_results.json`.
2. `construction/pipeline.py --mode grade`: a separately hosted OpenAI-compatible grader checks the *thinking-model* responses against gold answers and writes `output/llm_grading_think.json`.
3. `construction/pipeline.py --mode instruct`: the instruct model answers every problem without a prefix and writes `output/instruct_results.json`.
4. `construction/pipeline.py --mode prefix`: only think-graded failures are probed. `TextProcessor` splits the thinking trace at reflection keywords; cumulative prefixes are tried from k=0 upward with a grader after each answer. Outputs are written under `output/<model-name>/k_prefix.json` (and JSONL companions), not at the top level.
5. `construction/merge.py`: combines the above files into Alpaca `instruction`/`input`/`output` SFT records with `<think>...</think>` and a final answer. `data/dataset_info.json` maps these columns; the constructed dataset itself is absent from Git and is linked from Hugging Face in README.
6. SFT is not implemented inside this repository. README links to [Relax](https://github.com/redai-infra/Relax) and its `examples/cot_compression/` 8-GPU scripts. The paper reports full-parameter BF16 SFT at 32,768 tokens on 8 H20-141GB GPUs. Training and checkpoint commands must be audited in that separate project before a formal run.
7. `evaluation/eval.sh` invokes lighteval/vLLM with `evaluation/custom_tasks.py` for AIME24, AIME25, HMMT25, and MATH-500. The script hardcodes tensor parallel size 4 and a 32,768-token context/output, so it is not a 6 GiB smoke-test command. The paper also reports AMO-Bench and cross-domain analyses that this script does not cover.

## Released SFT data audit

- Hugging Face repository: `redai-infra/hint-tuning-1k`, revision `b4c3eac6b9b4880b6670f8d13b28b80f2cd1b302`.
- Actual source filename: `k1.15_llmgrading.json`; README's local target name is `data/hint_tuning_1k.json`.
- Size: 34,584,454 bytes. SHA-256/LFS OID: `f50883c17a5e4f296e0e585d66566df678d7a2446ccc38cd4d708513ae927807`.
- The downloaded file has exactly 1,000 unique `instruction` values, empty `input` values, non-empty `output` values, and paired `<think>...</think>` tags in every record.
- All 1,000 records align in order with `data/problems.json` after whitespace normalization. There are 999 unique normalized prompts because one prompt is duplicated in the raw corpus.
- Output size is highly skewed: median 17,390 characters, 95th percentile 119,217, maximum 204,525. A local smoke subset must be selected after tokenizer-based length measurement; choosing the first N rows is not a safe proxy for memory use.

## Relax training audit

The linked Relax repository was inspected at the exact README commit, `5f86ed96786022640c48143bddcb465eb4679682`. Its Qwen3 script:

- reads `--prompt-data`, with `--input-key instruction` and `--label-key output`; the loader appends the label as a learnable assistant message;
- uses 32,768-token sequences, global batch 32, dynamic batching, `truncate_right`, tensor parallel 2, BF16, FlashAttention, activation recomputation, and six epochs;
- writes a Megatron distributed checkpoint, with a separate `scripts/tools/convert_torch_dist_to_hf_bridge.py` conversion step needed for a Hugging Face/vLLM-consumable directory;
- launches a Ray job with eight actor GPUs and assumes Megatron/Transformer Engine infrastructure.

This confirms Relax is the formal-paper path and is unsuitable as the first 6 GiB local training path. A small Hugging Face PEFT/LoRA SFT runner can validate the same published Alpaca examples and causal-language-model objective locally. That is a reproduction harness adaptation, not a change to Hint Tuning's difficulty/hint construction.

## Reproduction questions to resolve before training

- README describes No-Hint as an instruct-model success; the actual `merge.py` assigns it when the *thinking model* was graded correct and then uses the instruct response without grading that response. A successful k=0 in prefix search is handled by the Sparse-Hint branch. Record this as an upstream implementation/documentation discrepancy; do not silently change it in the baseline.
- The README merge example points to `output/k_prefix.json`, while the script writes `output/<model-name>/k_prefix.json`. Use the real nested path for any future run.
- `construction/config.yaml` and `evaluation/eval.sh` default to 4-way tensor parallelism and 32K tokens. The repository has no pinned `requirements.txt`, `pyproject.toml`, or lockfile.
- The paper appendix reports 1,003 constructed samples while this checkout has 1,000 raw problems. Verify the released SFT dataset separately rather than assuming counts match.
- The released file has 1,000 rows, confirming the paper appendix's 1,003 total is inconsistent with both the repository input and released data.
- At the current implementation boundary, `merge.py` treats `prefix_ex is None` (think-model graded correct) as No-Hint. A successful prefix probe with `k_value=0` receives the Sparse-Hint marker. A direct function check confirms this behavior. Preserve it in baseline records and report it as upstream behavior.

## Current WSL issue

WSL initially started and passed GPU queries. Later launches failed with `CreateVm/HCS/0x800705aa`, while Windows had only about 2.1–2.4 GB free of 16 GB RAM. This code means insufficient system resources; it is not evidence that Ubuntu, CUDA passthrough, or the VHDX needs reinstalling. Freeing host memory or rebooting is the minimum corrective action before package installation. Do not change `.wslconfig` until WSL can start again and actual PyTorch memory behavior is measured.

## First execution plan

1. Keep `origin` pointed at the user's fork and `upstream` at the official repo. Preserve `main` at the upstream commit and commit baseline documentation/scripts only on `reproduce`. Create `student-aware` only after baseline validation.
2. Free enough Windows memory for WSL to start. Then create an isolated WSL environment; use explicit cache directories on the E:-backed Ubuntu VHDX, install only the versions needed by the chosen local workflow, and run a tiny PyTorch `torch.cuda.is_available()` and allocation check before loading a model.
3. Inspect the released SFT dataset and Relax training format, then choose and pin a small Qwen3 smoke model. Qwen3-0.6B is a candidate because its official model card supports thinking and non-thinking modes; it is a *workflow test*, not a paper-equivalent model pair. Keep data to a handful of examples, sequence/output limits short, batch size 1, and use LoRA only if memory inspection requires it.
4. Before any training, record the exact command, model revision, sample IDs, dependency versions, estimated weight/activation/optimizer VRAM, target checkpoint path, and expected output files. A local smoke test should verify data loading, one or a few SFT updates, checkpoint reload, generation, and a tiny held-out evaluation. Formal data regeneration with 4B models, 32K full-parameter SFT, and benchmark evaluation belongs on rented multi-GPU hardware.

No packages, model weights, training runs, or evaluations were started during this audit.
