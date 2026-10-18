"""One-seed, one-epoch OpenMIC pipeline smoke run on official VGGish features.

This intentionally small MLP is a data/loss/checkpoint smoke baseline. It is
not the complete MusicTalk PaSST + BCP-WP + CNN ensemble reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "0_dataset" / "processed" / "openmic-2018",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "smoke")
    parser.add_argument("--x-key", default="X")
    parser.add_argument("--y-key", default=None, help="label key; default auto-detects Y or Y_true")
    parser.add_argument("--mask-key", default="Y_mask")
    parser.add_argument("--sample-key", default="sample_key")
    parser.add_argument("--max-train", type=int, default=1024)
    parser.add_argument("--max-val", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def decode_ids(values: np.ndarray) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def load_split_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Official split file not found: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def find_split_file(dataset_dir: Path, split: str) -> Path:
    candidates = {
        "train": ("train01.txt", "split01_train.csv", "train01.csv"),
        "test": ("test01.txt", "split01_test.csv", "test01.csv"),
    }[split]
    for name in candidates:
        path = dataset_dir / "partitions" / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not locate the official OpenMIC {split} split; tried: "
        + ", ".join(str(dataset_dir / "partitions" / name) for name in candidates)
    )


def validation_metrics(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Compute OpenMIC macro ROC-AUC, average precision, and ROC-tuned F1.

    Missing labels are excluded class by class. F1 uses the threshold whose
    ROC point is closest to (0, 1), matching the paper's stated rule.
    """
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    auc_values: list[float] = []
    ap_values: list[float] = []
    f1_values: list[float] = []

    for class_index in range(targets.shape[1]):
        observed = mask[:, class_index].astype(bool)
        truth = targets[observed, class_index].astype(bool)
        scores = probabilities[observed, class_index]
        positives = int(truth.sum())
        negatives = int(truth.size - positives)
        if positives == 0 or negatives == 0:
            continue

        ascending = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[ascending]
        sorted_truth = truth[ascending]
        ranks = np.empty(truth.size, dtype=np.float64)
        start = 0
        while start < truth.size:
            end = start + 1
            while end < truth.size and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[start:end] = ((start + 1) + end) / 2.0
            start = end
        positive_rank_sum = ranks[sorted_truth].sum()
        auc_values.append(float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)))

        descending = np.argsort(-scores, kind="mergesort")
        sorted_positive = truth[descending].astype(np.float64)
        precision_at_k = np.cumsum(sorted_positive) / np.arange(1, truth.size + 1)
        ap_values.append(float((precision_at_k * sorted_positive).sum() / positives))

        candidates = np.concatenate(([np.nextafter(scores.max(), np.inf)], np.unique(scores)))
        best_distance = float("inf")
        best_f1 = 0.0
        for threshold in candidates:
            predicted = scores >= threshold
            tp = int(np.logical_and(predicted, truth).sum())
            fp = int(np.logical_and(predicted, ~truth).sum())
            fn = positives - tp
            tpr = tp / positives
            fpr = fp / negatives
            distance = fpr * fpr + (1.0 - tpr) * (1.0 - tpr)
            precision = tp / max(tp + fp, 1)
            recall = tpr
            f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
            if distance < best_distance:
                best_distance = distance
                best_f1 = f1
        f1_values.append(best_f1)

    def macro(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "val_macro_roc_auc": macro(auc_values),
        "val_map": macro(ap_values),
        "val_macro_f1_roc_tuned": macro(f1_values),
        "val_metric_classes": len(f1_values),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    npz_path = args.dataset_dir / "openmic-2018.npz"
    if not npz_path.is_file():
        raise SystemExit(f"Dataset not found: {npz_path}; extract and inspect OpenMIC first.")
    labels_key = args.y_key
    with np.load(npz_path, allow_pickle=True) as archive:
        if labels_key is None:
            labels_key = next((key for key in ("Y", "Y_true") if key in archive.files), None)
        if labels_key is None:
            raise SystemExit(
                f"Could not auto-detect label key ('Y' or 'Y_true'). Available keys: {archive.files}; pass --y-key explicitly."
            )
        missing = [key for key in (args.x_key, labels_key, args.sample_key) if key not in archive.files]
        if missing:
            raise SystemExit(f"NPZ keys missing: {missing}. Available keys: {archive.files}")
        features = np.asarray(archive[args.x_key])
        raw_targets = np.asarray(archive[labels_key])
        sample_ids = decode_ids(np.asarray(archive[args.sample_key]))
        if args.mask_key in archive.files:
            raw_mask = np.asarray(archive[args.mask_key]).astype(bool)
            resolved_mask_key = args.mask_key
        elif raw_targets.ndim == 2 and raw_targets.shape[1] == 40:
            # Some OpenMIC loaders concatenate the 20 relevance scores and
            # their 20 confidence/observation mask columns into one matrix.
            raw_mask = raw_targets[:, 20:] > 0.5
            raw_targets = raw_targets[:, :20]
            resolved_mask_key = f"{labels_key}[:, 20:] (concatenated mask)"
        else:
            finite_mask = np.isfinite(raw_targets)
            if not finite_mask.all():
                raw_mask = finite_mask
                resolved_mask_key = "finite-target fallback"
            elif np.isclose(raw_targets, 0.5, rtol=0.0, atol=1e-7).any():
                # OpenMIC encodes unobserved target/class pairs as neutral 0.5
                # in several commonly used label matrices.
                raw_mask = ~np.isclose(raw_targets, 0.5, rtol=0.0, atol=1e-7)
                resolved_mask_key = "neutral-0.5 sentinel fallback"
            else:
                raise SystemExit(
                    f"NPZ has no {args.mask_key!r}, no concatenated mask, and no NaN/neutral-0.5 markers; refusing to treat weak-label unknowns as negatives."
                )

    if features.shape[0] != raw_targets.shape[0] or len(sample_ids) != features.shape[0]:
        raise SystemExit(
            f"Misaligned arrays: X={features.shape}, Y={raw_targets.shape}, IDs={len(sample_ids)}"
        )
    if raw_targets.ndim != 2:
        raise SystemExit(f"Expected a 2-D multi-label target matrix, got {raw_targets.shape}")
    if raw_mask.shape != raw_targets.shape:
        raise SystemExit(f"Mask shape {raw_mask.shape} does not match targets {raw_targets.shape}")
    if not np.isfinite(raw_targets[raw_mask]).all():
        raise SystemExit("Non-finite values remain at observed target positions; inspect label encoding.")
    if np.any((raw_targets[raw_mask] < 0) | (raw_targets[raw_mask] > 1)):
        raise SystemExit("Observed targets are outside [0, 1]; inspect official target encoding.")
    # The paper defines targets as binary labels. OpenMIC's aggregated
    # relevance scores are thresholded at 0.5 in the commonly used loaders.
    raw_targets = (raw_targets > 0.5).astype(np.float32)

    train_split_path = find_split_file(args.dataset_dir, "train")
    test_split_path = find_split_file(args.dataset_dir, "test")
    train_ids = load_split_ids(train_split_path)
    test_ids = load_split_ids(test_split_path)
    if train_ids & test_ids:
        raise SystemExit("Official OpenMIC train and test partition files overlap.")
    index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    train_idx = np.array([index[sample_id] for sample_id in sorted(train_ids) if sample_id in index], dtype=np.int64)
    if train_idx.size == 0:
        raise SystemExit("No NPZ sample IDs matched official train01 partition.")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_idx)
    n_val_holdout = max(1, int(round(0.15 * len(train_idx))))
    val_idx = train_idx[:n_val_holdout][: args.max_val]
    train_idx = train_idx[n_val_holdout : n_val_holdout + args.max_train]
    if not train_idx.size or not val_idx.size:
        raise SystemExit("Not enough train-partition samples for a train/validation smoke split.")

    # OpenMIC stores postprocessed VGGish embeddings quantized to uint8 [0,255].
    # Invert the official VGGish quantizer's [-2,2] mapping before flattening.
    x = (np.asarray(features, dtype=np.float32) * (4.0 / 255.0) - 2.0).reshape(features.shape[0], -1)
    y = np.nan_to_num(raw_targets.astype(np.float32), nan=0.0)
    mask = raw_mask.astype(np.float32)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    mask_tensor = torch.from_numpy(mask)
    train_observed = mask[train_idx]
    train_labels = y[train_idx]
    observed_per_class = train_observed.sum(axis=0)
    positive_per_class = (train_labels * train_observed).sum(axis=0)
    negative_per_class = ((1.0 - train_labels) * train_observed).sum(axis=0)
    alpha = observed_per_class / (2.0 * np.maximum(positive_per_class, 1.0))
    beta = observed_per_class / (2.0 * np.maximum(negative_per_class, 1.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_tensor = torch.from_numpy(alpha.astype(np.float32)).to(device)
    beta_tensor = torch.from_numpy(beta.astype(np.float32)).to(device)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, y.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx], mask_tensor[train_idx])
    val_x, val_y, val_mask = x_tensor[val_idx], y_tensor[val_idx], mask_tensor[val_idx]
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "purpose": "pipeline smoke baseline; not complete MusicTalk architecture",
        "dataset": "OpenMIC-2018 official VGGish NPZ features",
        "official_partition": "train01 only; official test01 is not touched",
        "train_partition_file": train_split_path.name,
        "test_partition_file_verified_disjoint": test_split_path.name,
        "split": "deterministic 85/15 holdout within train01, then cap train and validation independently",
        "x_key": args.x_key,
        "feature_decode": "VGGish uint8 [0,255] dequantized linearly to [-2,2]",
        "y_key": labels_key,
        "mask_key": resolved_mask_key,
        "class_weighting": "inverse observed positive/negative class frequency, computed on capped train subset",
        "label_threshold": 0.5,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "num_instruments": int(y.shape[1]),
        "input_shape_per_clip": list(features.shape[1:]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(device),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = args.output_dir / "metrics.csv"

    with log_path.open("w", newline="", encoding="utf-8") as log_stream:
        writer = csv.DictWriter(
            log_stream,
            fieldnames=[
                "epoch",
                "train_masked_weighted_bce",
                "val_masked_weighted_bce",
                "val_macro_roc_auc",
                "val_map",
                "val_macro_f1_roc_tuned",
                "val_metric_classes",
                "seconds",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            loss_sum = 0.0
            observed_sum = 0.0
            for batch_x, batch_y, batch_mask in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                element_loss = (
                    alpha_tensor * batch_y * F.softplus(-logits)
                    + beta_tensor * (1.0 - batch_y) * F.softplus(logits)
                ) * batch_mask
                denominator = batch_mask.sum().clamp_min(1.0)
                loss = element_loss.sum() / denominator
                loss.backward()
                optimizer.step()
                loss_sum += float(element_loss.detach().sum().cpu())
                observed_sum += float(batch_mask.detach().sum().cpu())

            model.eval()
            with torch.no_grad():
                vx = val_x.to(device)
                vy = val_y.to(device)
                vm = val_mask.to(device)
                val_logits = model(vx)
                val_element = (
                    alpha_tensor * vy * F.softplus(-val_logits)
                    + beta_tensor * (1.0 - vy) * F.softplus(val_logits)
                ) * vm
                val_loss = float(val_element.sum().cpu() / vm.sum().clamp_min(1.0).cpu())
                val_metrics = validation_metrics(
                    val_logits.cpu().numpy(),
                    vy.cpu().numpy(),
                    vm.cpu().numpy(),
                )
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "train_masked_weighted_bce": loss_sum / max(observed_sum, 1.0),
                "val_masked_weighted_bce": val_loss,
                **val_metrics,
                "seconds": elapsed,
            }
            writer.writerow(row)
            log_stream.flush()
            print(json.dumps(row), flush=True)

    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        args.output_dir / "checkpoint.pt",
    )
    print(f"Saved smoke artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
