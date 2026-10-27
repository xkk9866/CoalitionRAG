"""Download the HotpotQA abstract JSONL in parallel HTTP ranges.

The Stanford bzip2 mirror was too slow. This file is the same introductory
paragraphs, including text_with_links, published as JSON lines.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/LovinYou/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts/resolve/main/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tsv"
)
SIZE = 11_513_319_796
PARTS = 8
OUT = Path(r"data/wiki/hf_parts")


def ranges(size: int, parts: int) -> list[tuple[int, int]]:
    chunk = (size + parts - 1) // parts
    out = []
    for i in range(parts):
        start = i * chunk
        if start >= size:
            break
        end = min(size - 1, (i + 1) * chunk - 1)
        out.append((start, end))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spans = ranges(SIZE, PARTS)
    procs = []
    for i, (start, end) in enumerate(spans):
        dest = OUT / f"part_{i:02d}.jsonl"
        expected = end - start + 1
        if dest.exists() and dest.stat().st_size == expected:
            print(f"part {i} already complete", flush=True)
            continue
        cmd = [
            "curl.exe",
            "-L",
            "--retry",
            "5",
            "--retry-all-errors",
            "-C",
            "-",
            "-r",
            f"{start}-{end}",
            "-o",
            str(dest),
            URL,
        ]
        print(f"start part {i} bytes {start}-{end}", flush=True)
        procs.append(subprocess.Popen(cmd))
    while procs:
        time.sleep(20)
        alive = []
        total = 0
        for path in sorted(OUT.glob("part_*.jsonl")):
            total += path.stat().st_size
        print(f"downloaded {total / 1e9:.2f} / {SIZE / 1e9:.2f} GB", flush=True)
        for proc in procs:
            if proc.poll() is None:
                alive.append(proc)
        procs = alive
    missing = []
    for i, (start, end) in enumerate(spans):
        dest = OUT / f"part_{i:02d}.jsonl"
        expected = end - start + 1
        got = dest.stat().st_size if dest.exists() else 0
        if got != expected:
            missing.append((i, got, expected))
    if missing:
        raise SystemExit(f"incomplete parts: {missing}")
    print("all parts complete", flush=True)


if __name__ == "__main__":
    main()
