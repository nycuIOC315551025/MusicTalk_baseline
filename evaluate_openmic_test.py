"""Evaluate a saved one-epoch MusicTalk checkpoint on OpenMIC split01 test.

This is inference only: it never trains or tunes on test labels. Macro F1 uses
the per-class ROC-nearest thresholds saved from the validation split; ROC-AUC
and average precision are threshold-free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_dual_branch_smoke import (
    MusicTalkBranches,
    OpenMICAudio,
    compute_openmic_metrics,
    load_openmic_arrays,
    load_panns_model,
    load_passt_model,
)
from models.preprocess import AugmentMelSTFT


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "0_dataset" / "processed" / "openmic-2018")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / "official_split_one_epoch" / "checkpoint_heads.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "official_test_one_epoch",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def f1_at_validation_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    thresholds: list[float | None],
) -> tuple[float | None, int, list[float | None]]:
    per_class: list[float | None] = []
    for class_index, threshold in enumerate(thresholds):
        if threshold is None:
            per_class.append(None)
            continue
        observed = masks[:, class_index].astype(bool)
        truth = targets[observed, class_index].astype(bool)
        if not len(truth):
            per_class.append(None)
            continue
        predicted = probabilities[observed, class_index] >= float(threshold)
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        denom = 2 * tp + fp + fn
        per_class.append(float(2 * tp / denom) if denom else 0.0)
    valid = [value for value in per_class if value is not None]
    return (float(np.mean(valid)) if valid else None, len(valid), per_class)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the selected PyTorch environment.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    # This project's PyTorch 2.10 checkpoints store torch.__version__ as a
    # TorchVersion object in the metadata. Keep weights_only loading enabled,
    # while explicitly allowlisting that inert metadata type.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    is_full_finetune_checkpoint = "model_state_dict" in checkpoint
    if is_full_finetune_checkpoint:
        saved_metrics = checkpoint["validation_metrics"]
        split_path = args.checkpoint.parent / "split_ids.json"
        split_ids = json.loads(split_path.read_text(encoding="utf-8"))
        checkpoint_train_ids = set(split_ids["train_ids"])
        validation_ids = set(split_ids["validation_ids"])
        run_config = checkpoint["config"]
        seed = run_config["seed"]
        training_epochs = checkpoint["epoch"]
        training_samples = run_config["train_samples"]
        validation_samples = run_config["validation_samples"]
        bcp_drop = run_config["bcp_dropped_patches"]
    else:
        saved_metrics = checkpoint["metrics"]
        checkpoint_train_ids = set(saved_metrics["train_ids"])
        validation_ids = set(saved_metrics["validation_ids"])
        seed = checkpoint["seed"]
        training_epochs = saved_metrics["epochs"]
        training_samples = saved_metrics["train_samples"]
        validation_samples = saved_metrics["validation_samples"]
        bcp_drop = 3
    run_summary_path = args.checkpoint.parent / "metrics.json"
    if is_full_finetune_checkpoint and run_summary_path.is_file():
        training_epochs = json.loads(run_summary_path.read_text(encoding="utf-8")).get(
            "epochs_run", training_epochs
        )
    prior_test_metrics_path = ROOT / "runs" / "official_test_one_epoch" / "metrics.json"
    prior_test_metrics = (
        json.loads(prior_test_metrics_path.read_text(encoding="utf-8"))
        if prior_test_metrics_path.is_file()
        else None
    )
    thresholds = saved_metrics["validation_class_thresholds_roc_nearest"]
    if len(thresholds) != 20:
        raise ValueError(f"Expected 20 validation thresholds, found {len(thresholds)}")

    all_ids, targets, masks, index = load_openmic_arrays(args.dataset_dir)
    partition_dir = args.dataset_dir / "partitions"
    official_train_ids = [
        line.strip()
        for line in (partition_dir / "split01_train.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_ids = [
        line.strip()
        for line in (partition_dir / "split01_test.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_id_set = set(test_ids)
    if len(test_id_set) != len(test_ids):
        raise ValueError("Official test partition contains duplicate IDs")
    if len(test_ids) != 5085:
        raise ValueError(f"Expected 5,085 official test IDs, found {len(test_ids)}")
    if checkpoint_train_ids & test_id_set or validation_ids & test_id_set:
        raise ValueError("Checkpoint train/validation IDs overlap the official test partition")
    if not checkpoint_train_ids.issubset(set(official_train_ids)) or not validation_ids.issubset(set(official_train_ids)):
        raise ValueError("Checkpoint train/validation IDs are not contained in official split01_train")
    missing = [sample_id for sample_id in test_ids if sample_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} official test IDs are missing from OpenMIC NPZ; first={missing[:3]}")

    dataset = OpenMICAudio(args.dataset_dir, test_ids, targets, masks, index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("LOADING_PASST", flush=True)
    passt, passt_missing = load_passt_model(args.weights_dir)
    print("LOADING_PANNS", flush=True)
    panns = load_panns_model(args.weights_dir, device)
    frontend = AugmentMelSTFT(
        n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
        freqm=0, timem=0, htk=False, fmin=0.0, fmax=15000.0, norm=1,
        fmin_aug_range=10, fmax_aug_range=2000,
    )
    model = MusicTalkBranches(passt, panns, frontend, bcp_drop=bcp_drop).to(device)
    if is_full_finetune_checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.passt.head[1].load_state_dict(checkpoint["passt_openmic_head"], strict=True)
        model.panns.fc_audioset.load_state_dict(checkpoint["panns_openmic_head"], strict=True)
        model.fusion.load_state_dict(checkpoint["fusion"], strict=True)
    model.eval()

    probability_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (waveform, batch_targets, batch_mask, _) in enumerate(loader, start=1):
            waveform = waveform.to(device, non_blocking=True)
            logits, _ = model(waveform)
            probability_parts.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
            target_parts.append(batch_targets.numpy())
            mask_parts.append(batch_mask.numpy().astype(bool))
            if batch_index % args.progress_every == 0 or batch_index == len(loader):
                print(
                    f"TEST_FORWARD batch={batch_index}/{len(loader)} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    probabilities = np.concatenate(probability_parts)
    test_targets = np.concatenate(target_parts)
    test_masks = np.concatenate(mask_parts)
    threshold_free = compute_openmic_metrics(probabilities, test_targets, test_masks)
    f1_macro, f1_classes, f1_per_class = f1_at_validation_thresholds(
        probabilities, test_targets, test_masks, thresholds
    )
    scored_classes = int(threshold_free["validation_scored_classes"])
    result = {
        "status": "completed",
        "dataset": "OpenMIC-2018 official split01_test, raw OGG audio",
        "official_test_evaluated": True,
        "test_samples": len(test_ids),
        "test_observed_labels": int(test_masks.sum()),
        "test_scored_classes_roc_auc_and_ap": scored_classes,
        "test_macro_roc_auc": threshold_free["validation_macro_roc_auc"],
        "test_macro_average_precision": threshold_free["validation_macro_average_precision"],
        "test_macro_f1_at_validation_roc_nearest_thresholds": f1_macro,
        "test_f1_scored_classes": f1_classes,
        "test_f1_per_class_at_validation_thresholds": f1_per_class,
        "f1_threshold_source": "Per-class ROC-nearest thresholds selected on the checkpoint's validation split and saved in the training checkpoint.",
        "test_thresholds_tuned_on_test": False,
        "official_test_evaluated_before_for_another_checkpoint": prior_test_metrics is not None,
        "prior_official_test_checkpoint": prior_test_metrics.get("checkpoint") if prior_test_metrics else None,
        "validation_macro_roc_auc_reference": saved_metrics["validation_macro_roc_auc"],
        "validation_macro_average_precision_reference": saved_metrics["validation_macro_average_precision"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": seed,
        "training_epochs": training_epochs,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(device),
        "inference_seconds": elapsed,
        "examples_per_second": len(test_ids) / elapsed,
        "passt_checkpoint_missing_head_keys": passt_missing,
        "note": "Test was evaluated once for a different one-epoch checkpoint earlier in this project. This result uses validation-selected thresholds and must be reported with that prior test exposure disclosed; it is not an independent untouched final test.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "test_macro_roc_auc",
            "test_macro_average_precision",
            "test_macro_f1_at_validation_roc_nearest_thresholds",
            "test_scored_classes_roc_auc_and_ap",
            "test_f1_scored_classes",
            "test_observed_labels",
            "inference_seconds",
            "examples_per_second",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in fields})
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        sample_ids=np.asarray(test_ids),
        probabilities=probabilities,
        targets=test_targets,
        masks=test_masks,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
