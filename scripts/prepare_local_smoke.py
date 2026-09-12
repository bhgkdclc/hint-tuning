#!/usr/bin/env python3
"""Prepare a tiny, provenance-checked slice of the released Hint Tuning SFT data.

This is a reproduction harness only. It does not regenerate or relabel hints.
Requires ``tokenizers`` and ``jinja2``. All output stays under gitignored output/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from jinja2 import Environment
from tokenizers import Tokenizer


SOURCE_SHA256 = "f50883c17a5e4f296e0e585d66566df678d7a2446ccc38cd4d708513ae927807"
DATASET_REVISION = "b4c3eac6b9b4880b6670f8d13b28b80f2cd1b302"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_ID = "Qwen/Qwen3-0.6B"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
MAX_CHAT_TOKENS = 3072
TRAIN_INDICES = (188, 279, 473, 3, 679, 758, 281)
EVAL_INDICES = (0, 92)
EXPECTED_STATES = {
    188: "no_hint", 279: "no_hint", 473: "no_hint",
    3: "sparse_hint", 679: "sparse_hint", 758: "sparse_hint",
    281: "full_hint", 0: "no_hint", 92: "sparse_hint",
}
FULL_PREFIX = (
    "This is a complex or challenging question, and it is difficult to provide "
    "a direct answer. I need to deep think about it."
)
SPARSE_PREFIX = "I may need some deep thinking."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(output: str) -> str:
    thinking = output.split("</think>", 1)[0].removeprefix("<think>").strip()
    if thinking.startswith(FULL_PREFIX):
        return "full_hint"
    if thinking.startswith(SPARSE_PREFIX):
        return "sparse_hint"
    return "no_hint"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/hint_tuning_1k.json"))
    parser.add_argument("--problems", type=Path, default=Path("data/problems.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("output/qwen3-0.6b-tokenizer.json"))
    parser.add_argument("--tokenizer-config", type=Path, default=Path("output/qwen3-0.6b-tokenizer_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/smoke"))
    args = parser.parse_args()

    actual_hash = sha256(args.source)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"SFT SHA-256 mismatch: {actual_hash} != {SOURCE_SHA256}")
    tokenizer_hash = sha256(args.tokenizer)
    tokenizer_config_hash = sha256(args.tokenizer_config)
    if tokenizer_hash != TOKENIZER_SHA256 or tokenizer_config_hash != TOKENIZER_CONFIG_SHA256:
        raise ValueError("Tokenizer files do not match the pinned Qwen3-0.6B revision")

    sft = json.loads(args.source.read_text(encoding="utf-8"))
    problems = json.loads(args.problems.read_text(encoding="utf-8"))
    if len(sft) != 1000 or len(problems) != 1000:
        raise ValueError(f"Expected 1000 aligned rows; got SFT={len(sft)}, problems={len(problems)}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_cfg = json.loads(args.tokenizer_config.read_text(encoding="utf-8"))
    template = Environment().from_string(tokenizer_cfg["chat_template"])
    selected = TRAIN_INDICES + EVAL_INDICES
    if len(selected) != len(set(selected)):
        raise ValueError("Train/eval source indices overlap")

    train, evaluation, rows = [], [], []
    for index in selected:
        example, original = sft[index], problems[index]
        if set(example) != {"instruction", "input", "output"} or example["input"]:
            raise ValueError(f"Unexpected Alpaca schema or nonempty input at row {index}")
        if normalize(example["instruction"]) != normalize(original["problem"]):
            raise ValueError(f"Source problem mismatch at row {index}")
        output = example["output"]
        if not output.startswith("<think>") or "</think>" not in output:
            raise ValueError(f"Missing think tags at row {index}")
        if not output.split("</think>", 1)[1].strip():
            raise ValueError(f"Missing final answer at row {index}")
        observed_state = state(output)
        if observed_state != EXPECTED_STATES[index]:
            raise ValueError(f"State marker changed at row {index}: {observed_state}")

        prompt = template.render(
            messages=[{"role": "user", "content": example["instruction"]}],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full = template.render(
            messages=[
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": output},
            ],
            add_generation_prompt=False,
            enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(prompt).ids
        full_ids = tokenizer.encode(full).ids
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat-template prompt tokens are not a prefix at row {index}")
        prompt_tokens = len(prompt_ids)
        chat_tokens = len(full_ids)
        if chat_tokens > MAX_CHAT_TOKENS or chat_tokens <= prompt_tokens:
            raise ValueError(f"Row {index} violates token gate: prompt={prompt_tokens}, full={chat_tokens}")

        split = "train" if index in TRAIN_INDICES else "eval"
        rows.append({
            "source_index": index,
            "split": split,
            "published_state_marker": observed_state,
            "prompt_tokens": prompt_tokens,
            "full_chat_tokens": chat_tokens,
        })
        if split == "train":
            train.append({"instruction": example["instruction"], "input": "", "output": output})
        else:
            evaluation.append({
                "source_index": index,
                "instruction": example["instruction"],
                "gold_answer": original["gold_answer"],
            })

    train_prompts = {normalize(item["instruction"]) for item in train}
    eval_prompts = {normalize(item["instruction"]) for item in evaluation}
    if train_prompts & eval_prompts:
        raise ValueError("Normalized train/eval prompts overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train.json", train)
    write_json(args.output_dir / "eval.json", evaluation)
    manifest = {
        "source_dataset": "redai-infra/hint-tuning-1k",
        "source_revision": DATASET_REVISION,
        "source_sha256": actual_hash,
        "problems_sha256": sha256(args.problems),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_config_sha256": tokenizer_config_hash,
        "max_chat_tokens": MAX_CHAT_TOKENS,
        "rows": rows,
        "train_sha256": sha256(args.output_dir / "train.json"),
        "eval_sha256": sha256(args.output_dir / "eval.json"),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Prepared {len(train)} train + {len(evaluation)} eval rows in {args.output_dir}")
    print(f"Max full chat tokens: {max(row['full_chat_tokens'] for row in rows)}")
    print(f"Train SHA-256: {manifest['train_sha256']}")
    print(f"Eval SHA-256:  {manifest['eval_sha256']}")


if __name__ == "__main__":
    main()
