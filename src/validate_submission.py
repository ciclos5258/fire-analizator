"""Проверка submission.csv на соответствие требованиям робота."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rle import rle_decode  # noqa: E402


def validate(sub_path: Path, sample_path: Path | None = None) -> None:
    with open(sub_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == ["chip_id", "class_id", "rle"], f"Bad header: {header}"

    seen = set()
    for i, row in enumerate(rows, start=2):
        assert len(row) == 3, f"Row {i}: expected 3 cols, got {len(row)}"
        cid, cid_class, rle = row
        assert cid, f"Row {i}: empty chip_id"
        assert cid_class in {"1", "2", "3"}, f"Row {i}: bad class_id={cid_class}"
        # RLE: пустая строка или пары start length
        if rle.strip():
            parts = rle.split()
            assert len(parts) % 2 == 0, f"Row {i}: odd RLE tokens"
            starts = [int(x) for x in parts[0::2]]
            lengths = [int(x) for x in parts[1::2]]
            assert all(s >= 1 for s in starts), f"Row {i}: start<1"
            assert all(l >= 1 for l in lengths), f"Row {i}: length<1"
            # Проверяем неубывание и непересечение
            ends = [s + l - 1 for s, l in zip(starts, lengths)]
            assert all(ends[k] < starts[k + 1] for k in range(len(starts) - 1)), \
                f"Row {i}: overlapping RLE runs"
        key = (cid, cid_class)
        assert key not in seen, f"Duplicate row: {key}"
        seen.add(key)

    # Сверка с sample_submission.csv
    if sample_path and sample_path.exists():
        with open(sample_path, encoding="utf-8", newline="") as f:
            sr = csv.reader(f)
            next(sr)
            expected = {(r[0], r[1]) for r in sr}
        missing = expected - seen
        extra = seen - expected
        assert not missing, f"Missing {len(missing)} rows, e.g. {list(missing)[:3]}"
        assert not extra, f"Extra {len(extra)} rows, e.g. {list(extra)[:3]}"

    print(f"✅ Submission валиден: {len(rows)} строк данных")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    p.add_argument("--sample", type=Path, default=None)
    args = p.parse_args()
    validate(args.submission, args.sample)