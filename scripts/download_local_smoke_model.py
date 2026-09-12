#!/usr/bin/env python3
"""Download the pinned local smoke model with resumable HTTP range requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "local-smoke.json"
USER_AGENT = "hint-tuning-reproduce/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_url(model_id: str, revision: str, filename: str) -> str:
    quoted_name = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{quoted_name}?download=true"


def request(url: str, start: int | None = None, end: int | None = None):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.Request(url, headers=headers)


def fetch_whole(
    url: str, destination: Path, expected_size: int, expected_hash: str, retries: int
) -> None:
    temporary = destination.with_name(destination.name + ".download")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(request(url), timeout=90) as response:
                with temporary.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(f"Wrong size for {destination.name}")
            actual_hash = sha256(temporary)
            if actual_hash != expected_hash:
                raise RuntimeError(f"Wrong SHA256 for {destination.name}: {actual_hash}")
            os.replace(temporary, destination)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Failed to download {destination.name}: {last_error}")


def fetch_range(
    url: str,
    destination: Path,
    start: int,
    end: int,
    total_size: int,
    retries: int,
) -> Path:
    expected_size = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected_size:
        return destination
    destination.unlink(missing_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    expected_content_range = f"bytes {start}-{end}/{total_size}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(request(url, start, end), timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {response.status}")
                actual_range = response.headers.get("Content-Range")
                if actual_range != expected_content_range:
                    raise RuntimeError(
                        f"Expected Content-Range {expected_content_range!r}, got {actual_range!r}"
                    )
                with temporary.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Expected {expected_size} bytes, got {temporary.stat().st_size}"
                )
            os.replace(temporary, destination)
            return destination
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Range {start}-{end} failed after {retries} attempts: {last_error}")


def download_large_file(
    url: str,
    destination: Path,
    expected_size: int,
    expected_hash: str,
    chunk_size: int,
    workers: int,
    retries: int,
    restart: bool,
) -> None:
    prefix = destination.with_name(destination.name + ".part")
    chunks_dir = destination.with_name("." + destination.name + ".chunks")
    assembling = destination.with_name(destination.name + ".assembling")

    if restart:
        prefix.unlink(missing_ok=True)
        assembling.unlink(missing_ok=True)
        if chunks_dir.is_dir():
            for path in chunks_dir.iterdir():
                if path.is_file():
                    path.unlink()
            chunks_dir.rmdir()

    prefix_size = prefix.stat().st_size if prefix.is_file() else 0
    if prefix_size > expected_size:
        raise RuntimeError(f"Partial prefix is larger than {destination.name}")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    for start in range(prefix_size, expected_size, chunk_size):
        end = min(start + chunk_size, expected_size) - 1
        chunk_path = chunks_dir / f"{start:012d}-{end:012d}.part"
        ranges.append((start, end, chunk_path))

    print(
        f"Downloading {destination.name}: {prefix_size}/{expected_size} bytes already present; "
        f"{len(ranges)} chunks, {workers} workers"
    )
    completed = 0
    next_report = 0.05
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_range, url, chunk_path, start, end, expected_size, retries
            ): (start, end)
            for start, end, chunk_path in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            fraction = completed / max(len(ranges), 1)
            if fraction >= next_report or completed == len(ranges):
                print(f"  chunks complete: {completed}/{len(ranges)} ({fraction:.0%})")
                next_report += 0.05

    digest = hashlib.sha256()
    with assembling.open("wb") as out:
        if prefix_size:
            with prefix.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
        for _start, _end, chunk_path in ranges:
            with chunk_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
    if assembling.stat().st_size != expected_size:
        assembling.unlink(missing_ok=True)
        raise RuntimeError(f"Assembled size mismatch for {destination.name}")
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        assembling.unlink(missing_ok=True)
        raise RuntimeError(
            f"Assembled SHA256 mismatch for {destination.name}: {actual_hash}. "
            "Re-run with --restart to discard the saved prefix and chunks."
        )
    os.replace(assembling, destination)
    prefix.unlink(missing_ok=True)
    for _start, _end, chunk_path in ranges:
        chunk_path.unlink(missing_ok=True)
    if chunks_dir.is_dir() and not any(chunks_dir.iterdir()):
        chunks_dir.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard incomplete weight chunks before downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_mib < 1 or not 1 <= args.workers <= 8 or args.retries < 1:
        raise ValueError("Require chunk-mib >= 1, 1 <= workers <= 8, and retries >= 1")
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model_dir = Path(cfg["model_dir"])
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    model_dir = model_dir.resolve()
    if not model_dir.is_relative_to((ROOT / "output" / "models").resolve()):
        raise ValueError("Model directory must stay within output/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    hashes = cfg["model_files_sha256"]
    sizes = cfg["model_files_size"]
    if set(hashes) != set(sizes):
        raise ValueError("Model hash and size manifests contain different files")
    for filename in hashes:
        destination = model_dir / filename
        expected_size = sizes[filename]
        expected_hash = hashes[filename]
        if (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and sha256(destination) == expected_hash
        ):
            print(f"Verified existing {filename}")
            continue
        url = model_url(cfg["model_id"], cfg["model_revision"], filename)
        if filename == "model.safetensors":
            download_large_file(
                url,
                destination,
                expected_size,
                expected_hash,
                args.chunk_mib * 1024 * 1024,
                args.workers,
                args.retries,
                args.restart,
            )
        else:
            print(f"Downloading {filename}")
            fetch_whole(url, destination, expected_size, expected_hash, args.retries)

    for filename, expected_hash in hashes.items():
        destination = model_dir / filename
        actual_hash = sha256(destination)
        if destination.stat().st_size != sizes[filename] or actual_hash != expected_hash:
            raise RuntimeError(f"Final verification failed for {filename}: {actual_hash}")
    print(f"PASS: pinned model assets verified in {model_dir}")


if __name__ == "__main__":
    main()
