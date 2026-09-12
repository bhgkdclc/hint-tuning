# Hint Tuning baseline: source and environment audit

Audit date: 2026-09-12. This note records facts observed before installing packages or running training. It makes no changes to the upstream method.

## Sources and Git state

- Paper: [Hint Tuning: Less Data Makes Better Reasoners, arXiv:2605.08665v2](https://arxiv.org/html/2605.08665v2).
- Official code: [redai-studio/hint-tuning](https://github.com/redai-studio/hint-tuning). The URL printed in the paper, `redai-infra/hint-tuning`, redirects to it.
- Local upstream commit: `6aac81943b2649af76ce5c2bb3be6f63347a2cf4` (2026-07-08). `main` is the unmodified official commit; `reproduce` starts at the same commit.
- The checkout was obtained directly from upstream because GitHub was not signed in and `bhgkdclc/hint-tuning` was not publicly available. The official remote is named `upstream`; `origin` will be assigned to the user's fork after it exists.
- Local `core.autocrlf=false` and a fresh checkout give LF shell scripts. This was necessary because the original Windows Git checkout used CRLF and `bash -n evaluation/eval.sh` failed under WSL. No tracked content changed.

## Local environment

| Item | Observed state | Consequence |
| --- | --- | --- |
| WSL | Ubuntu 22.04.5, WSL2, kernel 6.18.33.2 | Suitable Linux target; no reinstall indicated. |
| Python | WSL system Python 3.10.12; `venv` present; `pip` and `torch` absent | Build an isolated environment only after choosing compatible wheels and cache location. |
| GPU | RTX 3060 Laptop, compute capability 8.6, 6144 MiB VRAM; about 5.3 GiB free at inspection | Cannot use the paper's multi-GPU, 32K-token commands directly. |
| CUDA | WSL `nvidia-smi` works; driver reports CUDA UMD 13.4; `nvcc` absent | GPU passthrough works. `nvcc` is not needed to begin with prebuilt PyTorch wheels. PyTorch CUDA support is still untested. |
| Memory | WSL 7.7 GiB RAM and 2 GiB swap | Keep local jobs small; avoid concurrent models. |
| Disk | E: ~91 GiB free; C: ~7.3 GiB free; WSL `/` reports ~944 GiB free on a virtual filesystem | Check where the WSL VHDX and Hugging Face/pip caches actually reside before downloading weights. Do not infer 944 GiB physical free space. |

The initial sandboxed `wsl` query returned access denied; the same read-only queries succeeded with elevated execution. This was an execution-permission issue, not evidence of a broken WSL installation.

## Repository pipeline, as implemented

1. `data/problems.json`: 1,000 records with `problem` and `gold_answer`, from s1K. `construction/pipeline.py --mode think` converts these to question/solution records, runs the thinking model through vLLM, and writes `output/think_results.json`.
2. `construction/pipeline.py --mode grade`: a separately hosted OpenAI-compatible grader checks the *thinking-model* responses against gold answers and writes `output/llm_grading_think.json`.
3. `construction/pipeline.py --mode instruct`: the instruct model answers every problem without a prefix and writes `output/instruct_results.json`.
4. `construction/pipeline.py --mode prefix`: only think-graded failures are probed. `TextProcessor` splits the thinking trace at reflection keywords; cumulative prefixes are tried from k=0 upward with a grader after each answer. Outputs are written under `output/<model-name>/k_prefix.json` (and JSONL companions), not at the top level.
5. `construction/merge.py`: combines the above files into Alpaca `instruction`/`input`/`output` SFT records with `<think>...</think>` and a final answer. `data/dataset_info.json` maps these columns; the constructed dataset itself is absent from Git and is linked from Hugging Face in README.
6. SFT is not implemented inside this repository. README links to [Relax](https://github.com/redai-infra/Relax) and its `examples/cot_compression/` 8-GPU scripts. The paper reports full-parameter BF16 SFT at 32,768 tokens on 8 H20-141GB GPUs. Training and checkpoint commands must be audited in that separate project before a formal run.
7. `evaluation/eval.sh` invokes lighteval/vLLM with `evaluation/custom_tasks.py` for AIME24, AIME25, HMMT25, and MATH-500. The script hardcodes tensor parallel size 4 and a 32,768-token context/output, so it is not a 6 GiB smoke-test command. The paper also reports AMO-Bench and cross-domain analyses that this script does not cover.

## Reproduction questions to resolve before training

- README describes No-Hint as an instruct-model success; the actual `merge.py` assigns it when the *thinking model* was graded correct and then uses the instruct response without grading that response. A successful k=0 in prefix search is handled by the Sparse-Hint branch. Record this as an upstream implementation/documentation discrepancy; do not silently change it in the baseline.
- The README merge example points to `output/k_prefix.json`, while the script writes `output/<model-name>/k_prefix.json`. Use the real nested path for any future run.
- `construction/config.yaml` and `evaluation/eval.sh` default to 4-way tensor parallelism and 32K tokens. The repository has no pinned `requirements.txt`, `pyproject.toml`, or lockfile.
- The paper appendix reports 1,003 constructed samples while this checkout has 1,000 raw problems. Verify the released SFT dataset separately rather than assuming counts match.

## First execution plan

1. Sign in to GitHub as `bhgkdclc` and create a fork of `redai-studio/hint-tuning`. Configure it as `origin`, keeping `upstream` for the official repo. Preserve `main` at the upstream commit and commit baseline documentation/scripts only on `reproduce`. Create `student-aware` only after baseline validation.
2. Confirm the physical location of the WSL VHDX and choose model/pip cache paths away from the nearly full C: drive. Create an isolated WSL environment; install only the versions needed by the chosen local workflow. Run a tiny PyTorch `torch.cuda.is_available()` and allocation check before loading a model.
3. Inspect the released SFT dataset and Relax training format, then choose and pin a small Qwen3 smoke model. Qwen3-0.6B is a candidate because its official model card supports thinking and non-thinking modes; it is a *workflow test*, not a paper-equivalent model pair. Keep data to a handful of examples, sequence/output limits short, batch size 1, and use LoRA only if memory inspection requires it.
4. Before any training, record the exact command, model revision, sample IDs, dependency versions, estimated weight/activation/optimizer VRAM, target checkpoint path, and expected output files. A local smoke test should verify data loading, one or a few SFT updates, checkpoint reload, generation, and a tiny held-out evaluation. Formal data regeneration with 4B models, 32K full-parameter SFT, and benchmark evaluation belongs on rented multi-GPU hardware.

No packages, model weights, training runs, or evaluations were started during this audit.
