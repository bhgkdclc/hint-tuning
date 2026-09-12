#!/usr/bin/env python3
"""Run the gated RTX 3060 Hint Tuning engineering smoke workflow.

This harness consumes the authors' released SFT records. It does not construct,
relabel, or alter Hint Tuning difficulty/hint states.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_config(path: Path) -> dict[str, Any]:
    cfg = read_json(path)
    if cfg.get("kind") != "engineering-smoke-only":
        raise ValueError("Refusing a config not marked engineering-smoke-only")
    if cfg["max_steps"] > 2:
        raise ValueError("Local smoke is capped at two optimizer steps")
    if cfg["max_sequence_tokens"] > 3072:
        raise ValueError("Local smoke is capped at 3072 complete chat tokens")
    if cfg["max_steps"] != len(cfg["train_source_indices"]):
        raise ValueError("Each configured optimizer step needs exactly one source index")
    return cfg


def package_versions(cfg: dict[str, Any], require_exact: bool = True) -> dict[str, str]:
    versions: dict[str, str] = {}
    errors = []
    for package, expected in cfg["required_packages"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        versions[package] = actual
        if require_exact and actual.split("+")[0] != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
    if errors:
        raise RuntimeError("Package version gate failed: " + "; ".join(errors))
    return versions


def validate_model_files(cfg: dict[str, Any]) -> Path:
    model_dir = resolve(cfg["model_dir"])
    expected = cfg["model_files_sha256"]
    expected_sizes = cfg["model_files_size"]
    if set(expected) != set(expected_sizes):
        raise ValueError("Model hash and size manifests contain different files")
    for filename, expected_hash in expected.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned local model file: {path}")
        if path.stat().st_size != expected_sizes[filename]:
            raise ValueError(
                f"Local model file size mismatch: {filename}: {path.stat().st_size}"
            )
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Local model file hash mismatch: {filename}: {actual_hash}")
    return model_dir


def validate_data(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path = resolve(cfg["train_data"])
    eval_path = resolve(cfg["eval_data"])
    manifest_path = resolve(cfg["data_manifest"])
    for path in (train_path, eval_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared smoke input: {path}")
    actual_train = sha256(train_path)
    actual_eval = sha256(eval_path)
    if actual_train != cfg["expected_train_sha256"]:
        raise ValueError(f"Train data hash mismatch: {actual_train}")
    if actual_eval != cfg["expected_eval_sha256"]:
        raise ValueError(f"Eval data hash mismatch: {actual_eval}")
    train = read_json(train_path)
    evaluation = read_json(eval_path)
    manifest = read_json(manifest_path)
    train_rows = [row for row in manifest["rows"] if row["split"] == "train"]
    eval_rows = [row for row in manifest["rows"] if row["split"] == "eval"]
    if len(train) != 7 or len(evaluation) != 2 or len(train_rows) != 7 or len(eval_rows) != 2:
        raise ValueError("Expected the pinned 7-train/2-eval smoke split")
    if manifest["train_sha256"] != actual_train or manifest["eval_sha256"] != actual_eval:
        raise ValueError("Prepared data and manifest hashes disagree")
    if max(row["full_chat_tokens"] for row in manifest["rows"]) > cfg["max_sequence_tokens"]:
        raise ValueError("Prepared data exceeds the configured complete-sequence cap")
    return train, evaluation, manifest


def config_identity(config_path: Path) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "git_head": git_head(),
    }


def require_cuda(cfg: dict[str, Any]):
    import torch

    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("GPU smoke phases must run inside WSL2 Linux")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if cfg["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 is not supported by this device/runtime")
    free, total = torch.cuda.mem_get_info()
    free_mib, total_mib = free // 1048576, total // 1048576
    if free_mib < cfg["minimum_free_vram_mib"]:
        raise RuntimeError(
            f"Free VRAM gate failed: {free_mib} MiB < {cfg['minimum_free_vram_mib']} MiB"
        )
    return torch, free_mib, total_mib


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    model_dir = validate_model_files(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: dict[str, Any], torch):
    from transformers import AutoModelForCausalLM

    model_dir = validate_model_files(cfg)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        attn_implementation=cfg["attention_implementation"],
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
        device_map={"": 0},
    )


def add_lora(model, cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    lora = cfg["lora"]
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def encode_training_example(tokenizer, example: dict[str, str], limit: int, torch) -> dict[str, Any]:
    user = [{"role": "user", "content": example["instruction"]}]
    full_messages = user + [{"role": "assistant", "content": example["output"]}]
    prompt = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    full = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat-template prompt is not a token prefix of the full example")
    if len(full_ids) > limit:
        raise ValueError(f"Complete sequence has {len(full_ids)} tokens, above {limit}")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Example has no supervised assistant tokens")
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "sequence_tokens": len(full_ids),
        "supervised_tokens": len(full_ids) - len(prompt_ids),
    }


def source_example(
    source_index: int, train: list[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in manifest["rows"] if row["split"] == "train"]
    by_source = {row["source_index"]: (train[pos], row) for pos, row in enumerate(rows)}
    if source_index not in by_source:
        raise ValueError(f"Train source index {source_index} is absent from the pinned split")
    return by_source[source_index]


def memory_mib(torch) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated() // 1048576,
        "reserved": torch.cuda.memory_reserved() // 1048576,
        "peak_allocated": torch.cuda.max_memory_allocated() // 1048576,
        "peak_reserved": torch.cuda.max_memory_reserved() // 1048576,
        "free": free // 1048576,
        "total": total // 1048576,
    }


def base_record(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "kind": cfg["kind"],
        **config_identity(config_path),
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "packages": package_versions(cfg),
        "python": sys.version,
        "platform": platform.platform(),
    }


def cmd_preflight(cfg: dict[str, Any], config_path: Path, force: bool) -> None:
    torch, free_before, total = require_cuda(cfg)
    train, _, manifest = validate_data(cfg)
    out_dir = resolve(cfg["output_dir"])
    artifact = out_dir / "preflight.json"
    if artifact.exists() and not force:
        raise FileExistsError(f"Preflight already exists: {artifact}; use --force to replace it")
    seed_everything(cfg["seed"], torch)
    record = base_record(cfg, config_path)
    record["status"] = "running"
    write_json(artifact, record)
    started = time.perf_counter()
    try:
        tokenizer = load_tokenizer(cfg)
        example, row = source_example(cfg["dry_run_source_index"], train, manifest)
        batch = encode_training_example(
            tokenizer, example, cfg["max_sequence_tokens"], torch
        )
        model = add_lora(load_base_model(cfg, torch), cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        model.train()
        device = next(model.parameters()).device
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        output.loss.backward()
        dry_memory = memory_mib(torch)
        if dry_memory["free"] < cfg["minimum_post_dry_vram_mib"]:
            raise RuntimeError(
                f"Longest-example dry run leaves only {dry_memory['free']} MiB free; "
                f"need {cfg['minimum_post_dry_vram_mib']} MiB"
            )
        record.update(
            {
                "status": "pass",
                "cuda": {
                    "device": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "runtime": torch.version.cuda,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "free_before_mib": free_before,
                    "total_mib": total,
                },
                "dry_run": {
                    "source_index": row["source_index"],
                    "state": row["published_state_marker"],
                    "sequence_tokens": batch["sequence_tokens"],
                    "supervised_tokens": batch["supervised_tokens"],
                    "loss": float(output.loss.detach().cpu()),
                    "trainable_parameters": trainable,
                    "total_parameters_with_adapter": total_params,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "memory_mib": dry_memory,
                },
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        if torch.cuda.is_available():
            record["memory_mib"] = memory_mib(torch)
        write_json(artifact, record)
        raise
    write_json(artifact, record)
    print(json.dumps(record["dry_run"], indent=2))
    print(f"PASS: {artifact}")


def check_preflight(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    artifact = resolve(cfg["output_dir"]) / "preflight.json"
    if not artifact.is_file():
        raise RuntimeError("Missing preflight.json; run the preflight command first")
    record = read_json(artifact)
    identity = config_identity(config_path)
    if record.get("status") != "pass":
        raise RuntimeError("Latest model dry-run preflight did not pass")
    for key in ("config_sha256", "runner_sha256"):
        if record.get(key) != identity[key]:
            raise RuntimeError(f"Preflight is stale because {key} changed")
    return record


def cmd_train(cfg: dict[str, Any], config_path: Path, force: bool) -> None:
    check_preflight(cfg, config_path)
    torch, _, _ = require_cuda(cfg)
    train, _, manifest = validate_data(cfg)
    out_dir = resolve(cfg["output_dir"])
    adapter_dir = out_dir / "adapter"
    metrics_path = out_dir / "train_metrics.jsonl"
    manifest_path = out_dir / "train_manifest.json"
    if any(path.exists() for path in (adapter_dir, metrics_path, manifest_path)) and not force:
        raise FileExistsError("Training outputs already exist; use --force to replace them")
    if force:
        import shutil

        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        metrics_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    seed_everything(cfg["seed"], torch)
    tokenizer = load_tokenizer(cfg)
    model = add_lora(load_base_model(cfg, torch), cfg)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"]
    )
    started = time.perf_counter()
    model.train()
    for step, source_index in enumerate(cfg["train_source_indices"], start=1):
        example, row = source_example(source_index, train, manifest)
        batch = encode_training_example(
            tokenizer, example, cfg["max_sequence_tokens"], torch
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"]
        )
        optimizer.step()
        metric = {
            "step": step,
            "source_index": source_index,
            "state": row["published_state_marker"],
            "sequence_tokens": batch["sequence_tokens"],
            "supervised_tokens": batch["supervised_tokens"],
            "loss": float(output.loss.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "elapsed_seconds": round(time.perf_counter() - step_started, 3),
            "memory_mib": memory_mib(torch),
        }
        append_jsonl(metrics_path, metric)
        print(json.dumps(metric))
    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    record = base_record(cfg, config_path)
    record.update(
        {
            "status": "pass",
            "steps": cfg["max_steps"],
            "source_indices": cfg["train_source_indices"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "adapter_dir": str(adapter_dir),
            "adapter_model_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
            "metrics_sha256": sha256(metrics_path),
        }
    )
    write_json(manifest_path, record)
    print(f"PASS: {manifest_path}")


def check_training(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    manifest_path = resolve(cfg["output_dir"]) / "train_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Missing train_manifest.json; run two-step training first")
    record = read_json(manifest_path)
    identity = config_identity(config_path)
    if record.get("status") != "pass":
        raise RuntimeError("Training manifest is not marked pass")
    for key in ("config_sha256", "runner_sha256"):
        if record.get(key) != identity[key]:
            raise RuntimeError(f"Training artifact is stale because {key} changed")
    adapter = resolve(cfg["output_dir"]) / "adapter" / "adapter_model.safetensors"
    if sha256(adapter) != record["adapter_model_sha256"]:
        raise RuntimeError("Adapter hash does not match the training manifest")
    metrics = resolve(cfg["output_dir"]) / "train_metrics.jsonl"
    if sha256(metrics) != record["metrics_sha256"]:
        raise RuntimeError("Optimizer metrics hash does not match the training manifest")
    return record


def cmd_infer(cfg: dict[str, Any], config_path: Path, force: bool) -> None:
    check_training(cfg, config_path)
    torch, _, _ = require_cuda(cfg)
    _, evaluation, manifest = validate_data(cfg)
    out_dir = resolve(cfg["output_dir"])
    predictions_path = out_dir / "predictions.jsonl"
    inference_path = out_dir / "inference_manifest.json"
    if any(path.exists() for path in (predictions_path, inference_path)) and not force:
        raise FileExistsError("Inference outputs exist; use --force to replace them")
    if force:
        predictions_path.unlink(missing_ok=True)
        inference_path.unlink(missing_ok=True)

    from peft import PeftModel

    seed_everything(cfg["seed"], torch)
    tokenizer = load_tokenizer(cfg)
    base = load_base_model(cfg, torch)
    model = PeftModel.from_pretrained(base, out_dir / "adapter", is_trainable=False)
    model.config.use_cache = True
    model.eval()
    device = next(model.parameters()).device
    eval_rows = [row for row in manifest["rows"] if row["split"] == "eval"]
    started = time.perf_counter()
    with torch.inference_mode():
        for example, row in zip(evaluation, eval_rows):
            content = example["instruction"] + "\n\n" + cfg["evaluation_prompt_suffix"]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            generated = model.generate(
                **inputs,
                do_sample=cfg["generation"]["do_sample"],
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = generated[0, inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
            think_tokens = (
                len(tokenizer.encode(think_match.group(1), add_special_tokens=False))
                if think_match
                else None
            )
            append_jsonl(
                predictions_path,
                {
                    "source_index": row["source_index"],
                    "instruction": example["instruction"],
                    "gold_answer": example["gold_answer"],
                    "prediction": text,
                    "generated_tokens": int(new_ids.numel()),
                    "think_tokens": think_tokens,
                },
            )
            print(f"generated source_index={row['source_index']} tokens={new_ids.numel()}")
    record = base_record(cfg, config_path)
    record.update(
        {
            "status": "pass",
            "fresh_process_adapter_reload": True,
            "num_predictions": len(evaluation),
            "predictions_sha256": sha256(predictions_path),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "memory_mib": memory_mib(torch),
        }
    )
    write_json(inference_path, record)
    print(f"PASS: {inference_path}")


def extract_boxed(text: str) -> str | None:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:pos]
    return None


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    value = text.strip().strip("$.")
    value = value.replace(",", "").replace(" ", "")
    return value


def strict_match(prediction: str | None, gold: str) -> bool:
    pred, expected = normalize_answer(prediction), normalize_answer(gold)
    if pred is None:
        return False
    if pred == expected:
        return True
    try:
        return Decimal(pred) == Decimal(expected)
    except InvalidOperation:
        return False


def cmd_evaluate(cfg: dict[str, Any], config_path: Path, force: bool) -> None:
    check_training(cfg, config_path)
    package_versions(cfg)
    out_dir = resolve(cfg["output_dir"])
    predictions_path = out_dir / "predictions.jsonl"
    inference_path = out_dir / "inference_manifest.json"
    evaluation_path = out_dir / "evaluation.json"
    if not predictions_path.is_file() or not inference_path.is_file():
        raise RuntimeError("Missing fresh-process inference artifacts")
    if evaluation_path.exists() and not force:
        raise FileExistsError("Evaluation output exists; use --force to replace it")
    inference = read_json(inference_path)
    identity = config_identity(config_path)
    if inference.get("status") != "pass" or not inference.get("fresh_process_adapter_reload"):
        raise RuntimeError("Inference manifest is not marked as a fresh-process pass")
    for key in ("config_sha256", "runner_sha256"):
        if inference.get(key) != identity[key]:
            raise RuntimeError(f"Inference artifact is stale because {key} changed")
    if sha256(predictions_path) != inference["predictions_sha256"]:
        raise RuntimeError("Prediction hash does not match inference manifest")
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    _, evaluation, _ = validate_data(cfg)
    expected_ids = {row["source_index"] for row in evaluation}
    observed_ids = {row["source_index"] for row in predictions}
    if len(predictions) != len(evaluation) or observed_ids != expected_ids:
        raise RuntimeError("Predictions are incomplete or contain unexpected source indices")
    records = []
    for item in predictions:
        answer = extract_boxed(item["prediction"])
        records.append(
            {
                "source_index": item["source_index"],
                "gold_answer": item["gold_answer"],
                "extracted_boxed_answer": answer,
                "strict_match": strict_match(answer, item["gold_answer"]),
                "generated_tokens": item["generated_tokens"],
                "think_tokens": item["think_tokens"],
            }
        )
    correct = sum(item["strict_match"] for item in records)
    result = {
        "timestamp_utc": utc_now(),
        "kind": cfg["kind"],
        **config_identity(config_path),
        "metric": "strict normalized boxed-answer match (smoke only)",
        "num_examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "mean_generated_tokens": sum(item["generated_tokens"] for item in records) / len(records),
        "records": records,
    }
    write_json(evaluation_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PASS: {evaluation_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "train", "infer", "evaluate")
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "local-smoke.json"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    commands = {
        "preflight": cmd_preflight,
        "train": cmd_train,
        "infer": cmd_infer,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](cfg, config_path, args.force)


if __name__ == "__main__":
    main()
