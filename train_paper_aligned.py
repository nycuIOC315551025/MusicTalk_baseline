"""Configurable OpenMIC MusicTalk training with paper-stated early stopping.

This is a paper-guided reimplementation built on the local PaSST/PANNs
snapshots. The paper specifies the data split, weighted masked BCE, and
patience=10, but not the optimizer, learning rate, random seed, or every
frontend detail. Those choices are written to config.json for each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from train_dual_branch_smoke import (
    MusicTalkBranches,
    OpenMICAudio,
    compute_openmic_metrics,
    get_capped_split,
    load_openmic_arrays,
    load_panns_model,
    load_passt_model,
    weighted_masked_bce,
)
from models.preprocess import AugmentMelSTFT
from bcp_wp import bcp_wp_keep_mask, patch_brightness


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


class NoSingletonBatchSampler(Sampler[list[int]]):
    """Merge a final one-item batch into its predecessor for BatchNorm safety."""

    def __init__(self, sample_count: int, batch_size: int, generator: torch.Generator):
        self.sample_count = sample_count
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        indices = torch.randperm(self.sample_count, generator=self.generator).tolist()
        batches = [indices[start : start + self.batch_size] for start in range(0, self.sample_count, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self) -> int:
        return (self.sample_count + self.batch_size - 1) // self.batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "0_dataset" / "processed" / "openmic-2018")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=None,
                        help="Start a clearly labeled fine-tuning branch from a model-only checkpoint; optimizer/RNG are reinitialized")
    parser.add_argument("--split-ids", type=Path, default=None,
                        help="Reuse a saved train/validation split JSON (required with --initial-checkpoint)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap; 0 uses the full paper split")
    parser.add_argument("--max-val", type=int, default=0, help="Optional cap; 0 uses the full paper split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3, help="Development cap; increase after reviewing runtime and metrics")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Also save a model-only checkpoint every N completed epochs; 0 disables periodic snapshots")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Backbone learning rate; not specified by the paper")
    parser.add_argument("--head-learning-rate", type=float, default=3e-4, help="New OpenMIC/fusion heads; not specified by the paper")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Not specified by the paper")
    parser.add_argument("--bcp-drop", type=int, default=3)
    parser.add_argument("--freeze-backbones", action="store_true", help="Train only branch heads and fusion for a diagnostic comparison")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class PaperTrainingBranches(MusicTalkBranches):
    """MusicTalk branch/fusion model with optional end-to-end fine-tuning."""

    def __init__(self, passt: nn.Module, panns: nn.Module, frontend: nn.Module, bcp_drop: int, freeze_backbones: bool):
        super().__init__(passt, panns, frontend, bcp_drop)
        self.freeze_backbones = freeze_backbones

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.mel_frontend.eval()
        if self.freeze_backbones:
            self.passt.eval()
            self.panns.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        mel = self.mel_frontend(waveform).unsqueeze(1)
        keep_mask = None
        if self.training:
            # PaSST's 128-bin input uses 12 overlapping 16-bin frequency
            # regions (stride 10); preserve 9 by brightness-weighted sampling.
            mel_power = torch.exp(mel.squeeze(1) * 5.0 - 4.5)
            bands = [slice(i * 10, min(i * 10 + 16, mel_power.shape[1])) for i in range(12)]
            brightness = patch_brightness(mel_power, bands, frequency_dim=1)
            keep_mask = bcp_wp_keep_mask(brightness, self.bcp_drop, training=True)

        if self.freeze_backbones:
            with torch.no_grad():
                passt_features = self.passt.forward_features(mel, bcp_keep_mask=keep_mask)
                panns_outputs = self.panns(waveform)
        else:
            passt_features = self.passt.forward_features(mel, bcp_keep_mask=keep_mask)
            panns_outputs = self.panns(waveform)

        if isinstance(passt_features, tuple):
            passt_features = (passt_features[0] + passt_features[1]) / 2.0
        passt_logits = self.passt.head(passt_features)
        panns_logits = self.panns.fc_audioset(panns_outputs["embedding"])
        branch_probabilities = torch.cat((torch.sigmoid(passt_logits), torch.sigmoid(panns_logits)), dim=1)
        return self.fusion(branch_probabilities), keep_mask


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("epochs, patience, and batch-size must be positive")
    if args.checkpoint_every < 0:
        raise SystemExit("checkpoint-every must be zero (disabled) or positive")
    if not args.freeze_backbones and args.batch_size < 2:
        raise SystemExit("Trainable PANNs BatchNorm needs batch-size >= 2; use --freeze-backbones for batch-size 1")
    if args.max_train < 0 or args.max_val < 0:
        raise SystemExit("max-train and max-val must be zero (full split) or positive")
    if args.initial_checkpoint is not None and args.split_ids is None:
        raise SystemExit("--split-ids is required with --initial-checkpoint to preserve the original split")
    if args.initial_checkpoint is None and args.split_ids is not None:
        raise SystemExit("--split-ids is only supported with --initial-checkpoint")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the selected PyTorch environment")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {args.output_dir}")
    if not (args.dataset_dir / "openmic-2018.npz").is_file():
        raise SystemExit(f"OpenMIC NPZ not found under {args.dataset_dir}")
    if not (args.weights_dir / "passt-s-kd-ap.486.pt").is_file() or not (args.weights_dir / "ResNet38_mAP_0.434.pth").is_file():
        raise SystemExit(f"Required AudioSet pretrained checkpoints not found under {args.weights_dir}")

    initial_checkpoint = None
    initial_checkpoint_epoch = 0
    initial_validation_metrics = None
    initial_run_epochs = None
    if args.initial_checkpoint is not None:
        if not args.initial_checkpoint.is_file() or not args.split_ids.is_file():
            raise SystemExit("Initial checkpoint or split-ids JSON does not exist")
        # Keep restricted loading enabled. This local checkpoint stores the
        # inert TorchVersion metadata object from torch.__version__.
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        initial_checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" not in initial_checkpoint or "epoch" not in initial_checkpoint:
            raise SystemExit("Initial checkpoint must contain a full model_state_dict and source epoch")
        initial_checkpoint_epoch = int(initial_checkpoint["epoch"])
        if args.epochs <= initial_checkpoint_epoch:
            raise SystemExit("--epochs must exceed the initial checkpoint epoch")
        source_config = initial_checkpoint.get("config", {})
        expected_config = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "backbones_frozen": args.freeze_backbones,
            "backbone_learning_rate": args.learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "bcp_dropped_patches": args.bcp_drop,
        }
        mismatches = {
            key: (source_config.get(key), value)
            for key, value in expected_config.items()
            if source_config.get(key) != value
        }
        if mismatches:
            raise SystemExit(f"Continuation settings differ from initial checkpoint: {mismatches}")
        source_summary_path = args.initial_checkpoint.parent / "metrics.json"
        if source_summary_path.is_file():
            initial_run_epochs = json.loads(source_summary_path.read_text(encoding="utf-8")).get("epochs_run")
        initial_validation_metrics = initial_checkpoint.get("validation_metrics")
        if not isinstance(initial_validation_metrics, dict):
            raise SystemExit("Initial checkpoint is missing validation metrics")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda")

    stage_start = time.perf_counter()
    _, targets, masks, index = load_openmic_arrays(args.dataset_dir)
    cap_train = args.max_train or len(index)
    cap_val = args.max_val or len(index)
    if args.split_ids is not None:
        saved_split = json.loads(args.split_ids.read_text(encoding="utf-8"))
        train_ids = list(saved_split["train_ids"])
        val_ids = list(saved_split["validation_ids"])
        train_set, val_set = set(train_ids), set(val_ids)
        official_train_ids = {
            line.strip()
            for line in (args.dataset_dir / "partitions" / "split01_train.csv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if len(train_set) != len(train_ids) or len(val_set) != len(val_ids) or train_set & val_set:
            raise SystemExit("Saved continuation split has duplicate or overlapping sample IDs")
        if not train_set.issubset(official_train_ids) or not val_set.issubset(official_train_ids):
            raise SystemExit("Saved continuation IDs must remain inside official split01_train")
        if not train_set.issubset(index) or not val_set.issubset(index):
            raise SystemExit("Saved continuation split contains sample IDs missing from the OpenMIC NPZ")
        source_config = initial_checkpoint["config"]
        if len(train_ids) != int(source_config["train_samples"]) or len(val_ids) != int(source_config["validation_samples"]):
            raise SystemExit("Saved continuation split counts differ from the initial checkpoint")
    else:
        train_ids, val_ids = get_capped_split(args.dataset_dir, args.seed, cap_train, cap_val)
    print(f"DATA_READY train={len(train_ids)} val={len(val_ids)} elapsed_s={time.perf_counter()-stage_start:.2f}", flush=True)

    train_rows = np.asarray([index[sample_id] for sample_id in train_ids], dtype=np.int64)
    observed = masks[train_rows].astype(np.float32)
    labels = targets[train_rows]
    n_observed = observed.sum(axis=0)
    positives = (observed * labels).sum(axis=0)
    negatives = (observed * (1.0 - labels)).sum(axis=0)
    alpha = torch.as_tensor(n_observed / (2.0 * np.maximum(positives, 1.0)), device=device)
    beta = torch.as_tensor(n_observed / (2.0 * np.maximum(negatives, 1.0)), device=device)

    train_dataset = OpenMICAudio(args.dataset_dir, train_ids, targets, masks, index)
    val_dataset = OpenMICAudio(args.dataset_dir, val_ids, targets, masks, index)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_batch_sampler = NoSingletonBatchSampler(len(train_dataset), args.batch_size, loader_generator)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    stage_start = time.perf_counter()
    print("LOADING_PASST", flush=True)
    passt, passt_missing = load_passt_model(args.weights_dir)
    print(f"PASST_READY elapsed_s={time.perf_counter()-stage_start:.2f}", flush=True)
    stage_start = time.perf_counter()
    print("LOADING_PANNS", flush=True)
    panns = load_panns_model(args.weights_dir, device)
    print(f"PANNS_READY elapsed_s={time.perf_counter()-stage_start:.2f}", flush=True)
    frontend = AugmentMelSTFT(
        n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
        freqm=0, timem=0, htk=False, fmin=0.0, fmax=15000.0, norm=1,
        fmin_aug_range=10, fmax_aug_range=2000,
    )
    model = PaperTrainingBranches(passt, panns, frontend, args.bcp_drop, args.freeze_backbones).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if args.freeze_backbones:
        for parameter in model.passt.parameters():
            parameter.requires_grad_(False)
        for parameter in model.panns.parameters():
            parameter.requires_grad_(False)
    for parameter in model.passt.head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.panns.fc_audioset.parameters():
        parameter.requires_grad_(True)
    for parameter in model.fusion.parameters():
        parameter.requires_grad_(True)

    head_parameters = list(model.passt.head.parameters()) + list(model.panns.fc_audioset.parameters()) + list(model.fusion.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids]
    parameter_groups = [{"params": head_parameters, "lr": args.head_learning_rate}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    config = {
        "paper": "Lin, Cheng, Chiu, IEEE OJCS 2024, DOI 10.1109/OJCS.2024.3476416",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "weights_dir": str(args.weights_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "train_samples": len(train_ids),
        "validation_samples": len(val_ids),
        "official_test_evaluated": False,
        "batch_size": args.batch_size,
        "train_batch_policy": "shuffle; merge a trailing singleton into the preceding batch to avoid a 1-sample PANNs BatchNorm batch",
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "checkpoint_every_epochs": args.checkpoint_every,
        "backbones_frozen": args.freeze_backbones,
        "backbone_learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "bcp_frequency_patches": 12,
        "bcp_dropped_patches": args.bcp_drop,
        "audio_seconds": 10,
        "sample_rate_hz": 32000,
        "batch_norm_behavior": "updated during trainable-backbone training; fixed only with --freeze-backbones",
        "paper_specified": ["OpenMIC split counts", "15% training holdout for validation", "inverse-frequency weighted BCE", "mask unknown labels", "early stopping patience 10", "save lowest validation-loss setting"],
        "reimplementation_choices_not_fully_specified_by_paper": ["seed", "optimizer AdamW", "learning rates", "weight decay", "batch size", "ten-second input selection", "PaSST/PANNs initialization checkpoint variants", "exact validation sample IDs"],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "passt_missing_openmic_head_keys": passt_missing,
    }
    continuation_metadata = None
    if initial_checkpoint is not None:
        source_checkpoint_path = args.initial_checkpoint.resolve()
        source_checkpoint_sha256 = hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
        continuation_metadata = {
            "kind": "best_checkpoint_branch_with_optimizer_and_rng_reset",
            "source_checkpoint": str(source_checkpoint_path),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_checkpoint_epoch": initial_checkpoint_epoch,
            "source_run_epochs_completed": initial_run_epochs,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "optimizer": "new AdamW state initialized from the same recorded hyperparameters",
            "rng": "reseeded from the same seed; generator position is not resumed",
            "interpretation": "fine-tuning branch from the best saved epoch, not an exact resume from the source run's final epoch",
        }
        config["continuation"] = continuation_metadata
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(args.output_dir / "split_ids.json", {"train_ids": train_ids, "validation_ids": val_ids})
    print(f"MODEL_READY device={config['device']} frozen={args.freeze_backbones} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    history: list[dict[str, object]] = []
    best_loss = float(initial_checkpoint["validation_loss_weighted_masked_bce"]) if initial_checkpoint is not None else float("inf")
    best_epoch = initial_checkpoint_epoch if initial_checkpoint is not None else 0
    best_validation_metrics = initial_validation_metrics
    epochs_without_improvement = 0
    stop_reason = "epoch_cap_reached"

    if initial_checkpoint is not None:
        initial_best_payload = {
            "epoch": best_epoch,
            "validation_loss_weighted_masked_bce": best_loss,
            "validation_metrics": best_validation_metrics,
            "model_state_dict": model.state_dict(),
            "config": config,
            "continuation": continuation_metadata,
        }
        checkpoint_path = args.output_dir / "checkpoint_best.pt"
        checkpoint_temp = args.output_dir / "checkpoint_best.tmp"
        torch.save(initial_best_payload, checkpoint_temp)
        checkpoint_temp.replace(checkpoint_path)
        if args.checkpoint_every and best_epoch % args.checkpoint_every == 0:
            periodic_path = args.output_dir / f"checkpoint_epoch_{best_epoch:03d}.pt"
            periodic_temp = periodic_path.with_suffix(".tmp")
            torch.save(
                {
                    **initial_best_payload,
                    "checkpoint_kind": "source_model_snapshot_without_optimizer_state",
                },
                periodic_temp,
            )
            periodic_temp.replace(periodic_path)

    first_epoch = initial_checkpoint_epoch + 1 if initial_checkpoint is not None else 1
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train(True)
        train_loss_sum = 0.0
        train_observed = 0.0
        for batch_index, (waveform, batch_targets, batch_mask, _) in enumerate(train_loader, start=1):
            waveform = waveform.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            logits, _ = model(waveform)
            loss = weighted_masked_bce(logits, batch_targets, batch_mask, alpha, beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n_valid = float(batch_mask.sum().item())
            train_loss_sum += float(loss.detach().item()) * n_valid
            train_observed += n_valid
            if batch_index == 1 or batch_index % 250 == 0:
                print(f"TRAIN epoch={epoch}/{args.epochs} batch={batch_index}/{len(train_loader)} loss={loss.item():.5f}", flush=True)

        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_start
        model.eval()
        val_loss_sum = 0.0
        val_observed = 0.0
        val_probabilities: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
        val_masks: list[np.ndarray] = []
        with torch.no_grad():
            for waveform, batch_targets, batch_mask, _ in val_loader:
                waveform = waveform.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                logits, _ = model(waveform)
                loss = weighted_masked_bce(logits, batch_targets, batch_mask, alpha, beta)
                probabilities = torch.sigmoid(logits)
                val_probabilities.append(probabilities.cpu().numpy())
                val_targets.append(batch_targets.cpu().numpy())
                val_masks.append(batch_mask.cpu().numpy().astype(bool))
                n_valid = float(batch_mask.sum().item())
                val_loss_sum += float(loss.item()) * n_valid
                val_observed += n_valid
        torch.cuda.synchronize()
        validation_metrics = compute_openmic_metrics(
            np.concatenate(val_probabilities),
            np.concatenate(val_targets),
            np.concatenate(val_masks),
        )
        val_loss = val_loss_sum / max(val_observed, 1.0)
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path = args.output_dir / "checkpoint_best.pt"
            checkpoint_temp = args.output_dir / "checkpoint_best.tmp"
            torch.save(
                {
                    "epoch": epoch,
                    "validation_loss_weighted_masked_bce": val_loss,
                    "validation_metrics": validation_metrics,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "continuation": continuation_metadata,
                },
                checkpoint_temp,
            )
            checkpoint_temp.replace(checkpoint_path)
        else:
            epochs_without_improvement += 1

        epoch_result = {
            "epoch": epoch,
            "training_loss_weighted_masked_bce": train_loss_sum / max(train_observed, 1.0),
            "validation_loss_weighted_masked_bce": val_loss,
            "validation_observed_labels": int(val_observed),
            "train_seconds": train_seconds,
            "validation_metrics": validation_metrics,
            "improved_best_validation_loss": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        history.append(epoch_result)
        atomic_json(args.output_dir / "history.json", history)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            periodic_path = args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            periodic_temp = periodic_path.with_suffix(".tmp")
            torch.save(
                {
                    "epoch": epoch,
                    "training_loss_weighted_masked_bce": epoch_result["training_loss_weighted_masked_bce"],
                    "validation_loss_weighted_masked_bce": val_loss,
                    "validation_metrics": validation_metrics,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "checkpoint_kind": "periodic_model_snapshot_without_optimizer_state",
                },
                periodic_temp,
            )
            periodic_temp.replace(periodic_path)
            print(f"CHECKPOINT_SAVED epoch={epoch} path={periodic_path}", flush=True)
        print(
            f"EPOCH_DONE epoch={epoch} train_loss={epoch_result['training_loss_weighted_masked_bce']:.5f} "
            f"val_loss={val_loss:.5f} val_auc={validation_metrics['validation_macro_roc_auc']} "
            f"val_mAP={validation_metrics['validation_macro_average_precision']} "
            f"val_f1={validation_metrics['validation_macro_f1_roc_nearest']} "
            f"best={best_epoch} time_s={train_seconds:.1f}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            stop_reason = "early_stopping_patience_reached"
            print(f"EARLY_STOP epoch={epoch} patience={args.patience}", flush=True)
            break

    summary = {
        "status": "early_stopped" if stop_reason == "early_stopping_patience_reached" else "epoch_cap_reached",
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_validation_loss_weighted_masked_bce": best_loss,
        "epochs_run": len(history),
        "first_epoch_index_this_stage": first_epoch if history else None,
        "last_epoch_index": history[-1]["epoch"] if history else initial_checkpoint_epoch or None,
        "source_checkpoint_epoch": initial_checkpoint_epoch if initial_checkpoint is not None else None,
        "source_run_epochs_completed": initial_run_epochs,
        "epochs_run_this_stage": len(history),
        "requested_epoch_cap": args.epochs,
        "early_stopping_patience": args.patience,
        "checkpoint_every_epochs": args.checkpoint_every,
        "backbones_frozen": args.freeze_backbones,
        "official_test_evaluated": False,
        "checkpoint": str((args.output_dir / "checkpoint_best.pt").resolve()),
        "best_validation_metrics": next((row["validation_metrics"] for row in history if row["epoch"] == best_epoch), best_validation_metrics),
        "continuation": continuation_metadata,
        "note": (
            "Best-checkpoint fine-tuning branch with optimizer and RNG reset; not an exact resume of the source run."
            if initial_checkpoint is not None
            else "Paper-guided OpenMIC run. If status is epoch_cap_reached, this is a capped development reproduction, not a patience-converged paper-length run."
        ),
    }
    atomic_json(args.output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
