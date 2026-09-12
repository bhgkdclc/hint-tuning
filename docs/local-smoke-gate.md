# Local baseline smoke gate (data slice verified; training not yet run)

This is a workflow check for the RTX 3060 Laptop GPU (6 GiB), not a reproduction of paper metrics. It uses the authors' released SFT responses without editing their difficulty labels or hint text. Keep `main` at the official commit and make any runnable harness changes only on `reproduce`.

## Fixed inputs

- SFT source: `redai-infra/hint-tuning-1k`, revision `b4c3eac6b9b4880b6670f8d13b28b80f2cd1b302`, file `k1.15_llmgrading.json`, SHA-256 `f50883c17a5e4f296e0e585d66566df678d7a2446ccc38cd4d708513ae927807`.
- Model candidate: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), revision `c1899de289a04d12100db370d81485cdf75e47ca`. The model supports thinking and non-thinking modes, but it is **not** the paper's Qwen3-4B-Thinking/Instruct pair. Treat any output as an engineering smoke result only.
- The model's BF16 `model.safetensors` is 1,503,300,328 bytes on disk; the tokenizer JSON is 11,422,654 bytes. These sizes are not a full training-VRAM estimate.
- Token counts below were computed using that revision's `tokenizer.json` and `tokenizer_config.json` chat template with one user `instruction` message and one assistant `output` message, with thinking enabled. Recheck with the eventual `transformers` version before training.

| Split | Source row index (zero-based) | Published state marker | Full chat tokens |
| --- | ---: | --- | ---: |
| Train | 188 | No-Hint | 684 |
| Train | 279 | No-Hint | 662 |
| Train | 473 | No-Hint | 570 |
| Train | 3 | Sparse-Hint | 1,227 |
| Train | 679 | Sparse-Hint | 698 |
| Train | 758 | Sparse-Hint | 730 |
| Train | 281 | Full-Hint | 2,672 |
| Eval | 0 | No-Hint | 2,144 |
| Eval | 92 | Sparse-Hint | 2,691 |

The nine selected prompts are distinct after whitespace normalization. The Full-Hint sample is longer, so set a 3,072-token **upper bound** and batch size 1. Reject a sample if the complete chat-formatted sequence exceeds the bound; do not truncate away `</think>` or the final answer. Use a short generation cap for the two held-out questions. This tiny set verifies plumbing; its accuracy cannot support a research claim.

The selection is implemented in [`scripts/prepare_local_smoke.py`](../scripts/prepare_local_smoke.py). The raw `data/problems.json` is tracked in Git. From the repository root in a Linux shell, download the released SFT file and the model's two tokenizer files at their fixed revisions:

```bash
mkdir -p output
curl -fL --retry 3 "https://huggingface.co/datasets/redai-infra/hint-tuning-1k/resolve/b4c3eac6b9b4880b6670f8d13b28b80f2cd1b302/k1.15_llmgrading.json" -o data/hint_tuning_1k.json
curl -fL --retry 3 "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json" -o output/qwen3-0.6b-tokenizer.json
curl -fL --retry 3 "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json" -o output/qwen3-0.6b-tokenizer_config.json
```

With `tokenizers` and `jinja2` installed in an isolated Python environment, run:

```bash
python scripts/prepare_local_smoke.py
```

It writes platform-independent LF JSON and verifies the SFT and tokenizer SHA-256 hashes, raw-problem alignment, state markers, complete chat-template sequence lengths, supervised assistant span, and split disjointness. It writes ignored `output/smoke/train.json`, `eval.json`, and `manifest.json`. The current Windows and WSL runs both produced 7 train and 2 eval rows, with maximum complete sequence 2,691 tokens, train SHA-256 `cebc3133c0ac6f748e56ac5207b09103529f9b741c428c544006cb145a13b128`, and eval SHA-256 `c9ae3fcae58592fc9c349b608b7603678c3564e0da7d1f6f4164366b8f396b83`.

## Checks required before the first optimizer step

1. WSL starts consistently; Windows has enough available RAM. The recent `CreateVm/HCS/0x800705aa` failure must be resolved before any install or GPU run.
2. An isolated WSL environment is active. Record `python`, `torch`, `transformers`, `peft`, `tokenizers`, and `safetensors` versions, CUDA runtime, model/data revisions, and the exact command in a run manifest. Confirm `torch.cuda.is_available()` and a small GPU allocation. Do not install the CUDA toolkit merely because `nvcc` is absent.
3. Recompute chat token counts with the installed tokenizer. Verify every train target has at least one supervised assistant token, each complete sequence is at most 3,072 tokens, and train/eval prompt overlap is zero.
4. Start with BF16 (if the device reports support), LoRA rank 8, gradient checkpointing, microbatch 1, accumulation 1, and only 2 optimizer steps. Use no QLoRA initially; the 0.6B model is small enough that 4-bit quantization adds complexity without a proven need. If memory preflight fails, first lower the sequence cap and reselect *complete* examples; consider QLoRA only after measuring the failure.
5. Before loading the model, require at least 4.5 GiB free VRAM and leave about 1 GiB headroom for the display and allocator. Measure peak allocated/reserved VRAM during a single forward/backward dry run; if it cannot complete, do not start the two-step training. This is an operational gate, not a guarantee derived from the weight file size.
6. Expected artifacts: a selected-row manifest with SHA-256s, the LoRA adapter checkpoint, optimizer-step/loss log, a reloaded adapter generating responses for the two held-out prompts, a JSONL of predictions, and an evaluation JSON with answer correctness plus generated-token counts. Reload from disk in a fresh process to prove checkpoint usability.

The official `evaluation/eval.sh` and Relax scripts must **not** be used for this local run: both assume multiple GPUs and 32K-token contexts. The local evaluation should reuse the same `Please reason step by step, and put your final answer within \\boxed{}.` prompt style while limiting itself to the two held-out records. Any resulting numbers must be labeled `smoke`, not Hint Tuning benchmark results.

Formal reproduction on rented GPUs can later run the pinned Qwen3-4B pair, the full construction pipeline, full-parameter Relax SFT, checkpoint export, and lighteval benchmarks. The paper's 8-H20, 32K configuration is a separate experiment from this local gate.

## Gated smoke commands

The local harness is pinned in `configs/local-smoke.json` and `requirements/local-smoke.txt`. Run it from WSL with the repository at `/mnt/e/Note/HintTuning` and the isolated environment at `/root/.venvs/hint-tuning-reproduce`:

```bash
cd /mnt/e/Note/HintTuning
source /root/.venvs/hint-tuning-reproduce/bin/activate

# Longest selected train record: load model + LoRA + forward/backward only.
python scripts/local_smoke.py preflight

# Refuses to run unless the current config and runner passed preflight.
python scripts/local_smoke.py train

# Run as a separate command/process to prove the saved adapter reloads.
python scripts/local_smoke.py infer

# Deterministic two-record plumbing metric; not a paper benchmark.
python scripts/local_smoke.py evaluate
```

`preflight` uses source row 758, the longer of the two records that will actually enter the optimizer (730 complete chat tokens). It performs no optimizer update, records peak CUDA memory, and must leave at least 512 MiB free. `train` uses source rows 473 and 758, exactly one optimizer step per record. It uses BF16 Qwen3-0.6B, LoRA rank 8 on `q_proj`/`v_proj`, batch size 1, no accumulation, no quantization, and SDPA. `infer` reloads the adapter from disk and greedily generates at most 128 new tokens for each held-out prompt. `evaluate` extracts the last complete `\boxed{...}` expression and applies a strict normalized match; this is intentionally a smoke-only parser, not the official lighteval/LLM judge.

The commands create `output/local-smoke/preflight.json`, `train_metrics.jsonl`, `train_manifest.json`, `adapter/`, `predictions.jsonl`, `inference_manifest.json`, and `evaluation.json`. Each stage checks hashes of the preceding artifacts. Use `--force` only when intentionally replacing an existing artifact from that stage.

The isolated environment was created successfully and currently pins PyTorch 2.13.0+cu130, Transformers 4.57.6, PEFT 0.18.0, Accelerate 1.12.0, Tokenizers 0.22.2, Safetensors 0.8.0, and NumPy 2.2.6. `pip check`, Qwen3 class import, `torch.cuda.is_available()`, BF16 support, and a small BF16 CUDA matrix multiplication passed. The model loader now requires the complete fixed-revision model in `output/models/qwen3-0.6b-c1899de` and verifies every required file's SHA-256 before loading with `local_files_only=True`. The model-weight preflight and all four commands remain unrun because WSL startup became intermittent again before the 1.5 GB model download completed.

The pinned model assets can be downloaded from either Windows or WSL. The helper verifies all small files directly, fetches the 1.5 GB weight through resumable 8 MiB byte ranges, and accepts the assembled file only when its size and LFS SHA-256 match the config:

```bash
python scripts/download_local_smoke_model.py
```

Interrupted runs reuse finished chunks and the existing `model.safetensors.part` prefix. If final verification reports that an old partial prefix is corrupt, run the helper once with `--restart` to discard incomplete download artifacts. Model files and partial chunks stay under ignored `output/models/`; they are never committed.

If Hugging Face's storage endpoint is slow from the current network, use Qwen's official ModelScope copy as a byte source:

```bash
python scripts/download_local_smoke_model.py --source modelscope
```

The alternate source does not weaken the pin: the helper still rejects any file whose exact byte size or SHA-256 differs from the fixed Hugging Face revision manifest.

### Measured 6 GiB boundary

An initial dry run deliberately exercised source row 281, the shortest released Full-Hint record at 2,672 complete chat tokens. Its forward and backward passes completed, but PyTorch reported 6,728 MiB peak allocated, 6,878 MiB peak reserved, and 0 MiB free, so the 512 MiB safety gate correctly failed before any optimizer step. There is no shorter complete Full-Hint record in the released 1K SFT file. The local gate therefore dry-runs the longest record that actually participates in the two-step optimizer smoke (row 758, 730 tokens). Row 281 remains in the prepared data so the published Full-Hint format is still validated, but it is not trained on this 6 GiB machine. Full-state training belongs on the later formal-GPU run.

## Completed local engineering smoke

The pinned model download completed with the expected 1,503,300,328-byte size and SHA-256. WSL2 Ubuntu 22.04, Python 3.10.12, PyTorch 2.13.0+cu130, CUDA availability, BF16 support, and `pip check` passed on the RTX 3060 Laptop GPU. The 730-token preflight passed with 2,676 MiB peak allocated, 2,722 MiB peak reserved, and 2,354 MiB free after backward. Two LoRA optimizer steps then ran on original released rows 473 and 758 with losses 1.2940 and 0.5382; peak reserved memory in training was 3,730 MiB. The saved adapter was reloaded in a new process and generated 128 tokens for each of two held-out prompts.

The strict smoke evaluator reported 0/2. Both generations exhausted the deliberately short 128-token cap while still in `<think>` and contained no `\boxed{}` final answer. This confirms the data, optimization, checkpoint, reload, inference, and evaluation machinery; it does **not** measure paper-level answer accuracy or prove that the Hint Tuning method improves reasoning. The hash-linked local artifacts are under ignored `output/local-smoke/`.
