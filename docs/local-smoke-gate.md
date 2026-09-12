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

It verifies the SFT and tokenizer SHA-256 hashes, raw-problem alignment, state markers, complete chat-template sequence lengths, supervised assistant span, and split disjointness. It writes ignored `output/smoke/train.json`, `eval.json`, and `manifest.json`. The current run produced 7 train and 2 eval rows, with maximum complete sequence 2,691 tokens, train SHA-256 `c86e07ad91d9f97affbf2689c7369fd2be969d55620f191c4a729cf232f02529`, and eval SHA-256 `efb89e3f88f989e6d59d81545f2d5aa6c8e051f40256566113cb4a9a48d33c29`.

## Checks required before the first optimizer step

1. WSL starts consistently; Windows has enough available RAM. The recent `CreateVm/HCS/0x800705aa` failure must be resolved before any install or GPU run.
2. An isolated WSL environment is active. Record `python`, `torch`, `transformers`, `peft`, `tokenizers`, and `safetensors` versions, CUDA runtime, model/data revisions, and the exact command in a run manifest. Confirm `torch.cuda.is_available()` and a small GPU allocation. Do not install the CUDA toolkit merely because `nvcc` is absent.
3. Recompute chat token counts with the installed tokenizer. Verify every train target has at least one supervised assistant token, each complete sequence is at most 3,072 tokens, and train/eval prompt overlap is zero.
4. Start with BF16 (if the device reports support), LoRA rank 8, gradient checkpointing, microbatch 1, accumulation 1, and only 2 optimizer steps. Use no QLoRA initially; the 0.6B model is small enough that 4-bit quantization adds complexity without a proven need. If memory preflight fails, first lower the sequence cap and reselect *complete* examples; consider QLoRA only after measuring the failure.
5. Before loading the model, require at least 4.5 GiB free VRAM and leave about 1 GiB headroom for the display and allocator. Measure peak allocated/reserved VRAM during a single forward/backward dry run; if it cannot complete, do not start the two-step training. This is an operational gate, not a guarantee derived from the weight file size.
6. Expected artifacts: a selected-row manifest with SHA-256s, the LoRA adapter checkpoint, optimizer-step/loss log, a reloaded adapter generating responses for the two held-out prompts, a JSONL of predictions, and an evaluation JSON with answer correctness plus generated-token counts. Reload from disk in a fresh process to prove checkpoint usability.

The official `evaluation/eval.sh` and Relax scripts must **not** be used for this local run: both assume multiple GPUs and 32K-token contexts. The local evaluation should reuse the same `Please reason step by step, and put your final answer within \\boxed{}.` prompt style while limiting itself to the two held-out records. Any resulting numbers must be labeled `smoke`, not Hint Tuning benchmark results.

Formal reproduction on rented GPUs can later run the pinned Qwen3-4B pair, the full construction pipeline, full-parameter Relax SFT, checkpoint export, and lighteval benchmarks. The paper's 8-H20, 32K configuration is a separate experiment from this local gate.
