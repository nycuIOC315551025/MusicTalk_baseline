"""Inspect the official OpenMIC archive layout without changing dataset files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "0_dataset" / "processed" / "openmic-2018",
    )
    args = parser.parse_args()
    root = args.dataset_dir
    npz_path = root / "openmic-2018.npz"
    if not npz_path.is_file():
        raise SystemExit(f"Missing NPZ file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as dataset:
        print(f"NPZ: {npz_path}")
        print("Keys and shapes:")
        for key in dataset.files:
            value = dataset[key]
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if value.ndim == 1 and value.size and value.dtype.kind in "USO":
                print(f"    first values: {value[:3]!r}")

    for name in ("train01.txt", "test01.txt"):
        path = root / "partitions" / name
        if path.is_file():
            ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            print(f"{name}: {len(ids)} IDs; first={ids[:3]}")

    labels_path = root / "openmic-2018-aggregated-labels.csv"
    if labels_path.is_file():
        with labels_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            print(f"Aggregated-label CSV header: {next(reader, [])}")
            print(f"Aggregated-label CSV first row: {next(reader, [])}")


if __name__ == "__main__":
    main()
