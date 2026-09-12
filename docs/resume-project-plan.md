# Resume-project route after the local Hint Tuning smoke

This project should be presented as a reproducible, controlled reasoning-efficiency experiment. The first completed milestone is an engineering smoke on one RTX 3060; it verifies the pipeline, not the paper's accuracy or token savings. The official `main` checkout stays untouched. Baseline work remains on `reproduce`; no student-aware method is implemented in this phase.

## What counts as a defensible project result

1. A pinned, runnable baseline with a real held-out evaluation. Report the base model, the released Hint Tuning SFT adaptation, and eventually the proposed variant under the **same** model initialization, training data eligibility rule, optimization budget, prompts, generation settings, answer checker, and hardware class.
2. At least answer accuracy, mean/median/p90 tokens *inside* `<think>`, and the fraction of generations that hit the output cap without a final answer. Keep raw per-problem predictions so paired differences and failure cases can be inspected. Report uncertainty or seed variation when the sample is small; never treat the present 0/2 smoke score as evidence of quality.
3. A fixed, disjoint held-out benchmark slice (prefer the repository's published evaluation tasks, with prompt-overlap checks against the training set), one clearly stated answer-grading rule, and identical generation budget for each compared model. Start with a manageable subset for debugging; run the full selected benchmark before making a performance claim. Document any changed context or output limit instead of calling an adapted result a paper reproduction.
4. Versioned configurations, exact model/data revisions and hashes, training and evaluation commands, GPU-memory preflight, checkpoint identity, and a result table plus a small error analysis. Generated weights and large datasets may remain outside Git with verified download instructions.

## Milestones and compute gates

| Milestone | Work and exit criterion | Hardware |
| --- | --- | --- |
| M0: engineering smoke | **Done:** released SFT slice, two LoRA steps, saved/reloaded adapter, two inference records, strict evaluation artifact. This establishes plumbing only. | Local RTX 3060 6 GiB. |
| M1: meaningful baseline | Pin the paper's 4B Thinking model and released 1K SFT data. First compute the *complete* token-length distribution and the retained sample count under a proposed training cap. Preflight the longest retained sample with the intended LoRA configuration, then train a fixed baseline subset without truncating away the final answer. Save manifests and a usable checkpoint. | Candidate: one full A800 80GB GPU; actual feasibility of the proposed sequence length is a measured gate, not a promise. Develop scripts locally before renting. |
| M2: baseline evaluation | Run the unchanged base and the M1 checkpoint through identical held-out prompts. Raise the output limit enough that completed answers can be observed, record cap-hit rate, inspect judge errors and duplicate prompts, and publish accuracy/token tradeoffs. Freeze this baseline and its evaluation protocol before any method change. | One GPU at a time; benchmark duration and memory depend on output length. |
| M3: research comparison | Only after M2 is repeatable, create `student-aware` from the frozen `reproduce` commit. Specify the student-capability measurement, construct the new targets without leaking evaluation answers, and compare against M2 under the same data eligibility, compute, and evaluation conditions. Include an ablation that isolates the proposed change. | Start with a bounded subset on one GPU; scale only after the comparison works. |
| M4: portfolio release | Public README with a pipeline diagram, environment, commands, exact limitations, results and plots, sample predictions, ablations, and a concise account of what changed from the official method. Only write a resume bullet using measured numbers. | No additional GPU requirement. |

The one-GPU M1/M2 path is an **adapted baseline** (LoRA and potentially a reduced sequence/data slice), not a claim to reproduce the paper's full-parameter 32,768-token result. The [paper's implementation details](https://arxiv.org/html/2605.08665v2) report 8×H20-141GB for that setup. An 8-GPU rental becomes relevant only if matching those published conditions and metrics is the objective. The separate data-construction probe, Relax training, and official evaluation still need their own preflight if that objective is chosen later.

## Immediate next work, before any rental

- Make the local evaluator treat a response that ends at `max_new_tokens` without a final answer as **capped**, not simply wrong; retain the raw answer and reason-token count when available.
- Pin a benchmark version and verify prompt overlap with the released 1K training file. Define the paired base-versus-tuned evaluation table and minimum output cap before training a larger model.
- Build a 4B training-data length/eligibility manifest and estimate the longest retained sample's memory with a short GPU preflight. Do not rent for a full run until model download, dependency versions, data paths, checkpoint writing, and the exact short test command are ready.

The existing [baseline walkthrough](baseline-walkthrough.md) describes the official construction and the future conceptual modification boundary. This roadmap changes neither the official algorithm nor its released labels.
