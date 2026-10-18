"""One-epoch, feature-frozen MusicTalk branch/fusion smoke on OpenMIC audio.

This checks the PaSST + BCP-WP and ResNet38 paths on real 10-second clips.
Only the two 20-class branch heads and the final fusion layer are trained;
this is deliberately a short integration run, not a full paper reproduction.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEPS = ROOT / "vendor" / "python_deps"
PASST_SOURCE = ROOT / "vendor" / "PaSST"
PANNS_SOURCE = ROOT / "vendor" / "PANNs" / "pytorch"

if os.name == "nt":
    # The staged package target contains Windows wheels, so keep it first on Windows.
    sys.path.insert(0, str(DEPS))
    sys.path.insert(1, str(PASST_SOURCE))
else:
    # WSL/Linux must prefer its native site-packages over those Windows wheels.
    sys.path.insert(1, str(PASST_SOURCE))
    sys.path.append(str(DEPS))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bcp_wp import bcp_wp_keep_mask, patch_brightness
import models.passt as passt_module
from models.passt import get_model
from models.preprocess import AugmentMelSTFT

# Upstream PaSST prints large tensor traces during its first forward by default.
# Keep the vendored model behavior but silence that diagnostic global here.
passt_module.first_RUN = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "0_dataset" / "processed" / "openmic-2018")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "dual_branch_smoke")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-train", type=int, default=12)
    parser.add_argument("--max-val", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--bcp-drop", type=int, default=3)
    return parser.parse_args()


def decode_ids(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def get_capped_split(dataset_dir: Path, seed: int, max_train: int, max_val: int) -> tuple[list[str], list[str]]:
    with np.load(dataset_dir / "openmic-2018.npz", allow_pickle=True) as archive:
        ids = decode_ids(np.asarray(archive["sample_key"]))
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    train_file = dataset_dir / "partitions" / "split01_train.csv"
    official_train_ids = [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [sample_id for sample_id in official_train_ids if sample_id not in index]
    if missing:
        raise ValueError(f"{len(missing)} official training IDs are missing from the NPZ; first={missing[:3]}")
    rng = np.random.default_rng(seed)
    rng.shuffle(official_train_ids)
    held_out = max(1, math.ceil(0.15 * len(official_train_ids)))
    val_ids = official_train_ids[: min(max_val, held_out)]
    train_ids = official_train_ids[held_out : held_out + max_train]
    if len(train_ids) < 2 or not val_ids:
        raise ValueError("Need at least two capped training samples and one validation sample")
    return train_ids, val_ids


class OpenMICAudio(Dataset):
    def __init__(self, dataset_dir: Path, sample_ids: list[str], targets: np.ndarray, masks: np.ndarray, index: dict[str, int]):
        self.dataset_dir = dataset_dir
        self.sample_ids = sample_ids
        self.targets = targets
        self.masks = masks
        self.index = index

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, item: int):
        sample_id = self.sample_ids[item]
        audio_path = self.dataset_dir / "audio" / sample_id[:3] / f"{sample_id}.ogg"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        wave, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        if sample_rate != 32000:
            wave = soxr.resample(wave, sample_rate, 32000, quality="HQ")
        clip_samples = 32000 * 10
        if wave.shape[0] < clip_samples:
            wave = np.pad(wave, (0, clip_samples - wave.shape[0]))
        else:
            wave = wave[:clip_samples]
        row = self.index[sample_id]
        return (
            torch.from_numpy(wave.copy()),
            torch.from_numpy(self.targets[row].copy()),
            torch.from_numpy(self.masks[row].astype(np.float32, copy=True)),
            sample_id,
        )


def load_openmic_arrays(dataset_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    with np.load(dataset_dir / "openmic-2018.npz", allow_pickle=True) as archive:
        ids = decode_ids(np.asarray(archive["sample_key"]))
        raw_targets = np.asarray(archive["Y_true"], dtype=np.float32)
        masks = np.asarray(archive["Y_mask"], dtype=bool)
    targets = (raw_targets > 0.5).astype(np.float32)
    if targets.shape != masks.shape or targets.shape[1] != 20:
        raise ValueError(f"Unexpected OpenMIC labels: targets={targets.shape}, masks={masks.shape}")
    return ids, targets, masks, {sample_id: i for i, sample_id in enumerate(ids)}


def load_panns_model(weights_dir: Path, device: torch.device) -> nn.Module:
    if str(PANNS_SOURCE) not in sys.path:
        sys.path.insert(1, str(PANNS_SOURCE))
    module_path = PANNS_SOURCE / "models.py"
    spec = importlib.util.spec_from_file_location("music_talk_panns_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PANNs model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("PANNS_MODEL_INIT_START", flush=True)
    # The AudioSet checkpoint overwrites every pretrained parameter. Skip
    # random initialization during construction to avoid spending minutes
    # generating 80M throwaway CPU weights before loading the checkpoint.
    init_names = ("kaiming_uniform_", "uniform_", "xavier_uniform_", "constant_", "ones_", "zeros_")
    original_inits = {name: getattr(nn.init, name) for name in init_names}
    original_module_init_layer = module.init_layer
    original_module_init_bn = module.init_bn
    try:
        for name in init_names:
            setattr(nn.init, name, lambda tensor, *args, **kwargs: tensor)
        module.init_layer = lambda layer: None
        module.init_bn = lambda bn: None
        model = module.ResNet38(
            sample_rate=32000, window_size=1024, hop_size=320,
            mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        )
    finally:
        for name, function in original_inits.items():
            setattr(nn.init, name, function)
        module.init_layer = original_module_init_layer
        module.init_bn = original_module_init_bn
    print("PANNS_MODEL_INIT_DONE", flush=True)
    print("PANNS_CHECKPOINT_LOAD_START", flush=True)
    state = torch.load(weights_dir / "ResNet38_mAP_0.434.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    print("PANNS_CHECKPOINT_LOAD_DONE", flush=True)
    model.fc_audioset = nn.Linear(2048, 20)
    nn.init.xavier_uniform_(model.fc_audioset.weight)
    nn.init.zeros_(model.fc_audioset.bias)
    return model


def load_passt_model(weights_dir: Path) -> tuple[nn.Module, list[str]]:
    output = open(os.devnull, "w", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(output):
            model = get_model(
                arch="passt_s_kd_p16_128_ap486", pretrained=False, n_classes=20,
                in_channels=1, fstride=10, tstride=10,
                # Keep the checkpoint's 99 time positions. The ten-second
                # frontend yields one additional patch, which upstream crops.
                input_fdim=128, input_tdim=998,
                u_patchout=0, s_patchout_t=40, s_patchout_f=0,
            )
    finally:
        output.close()
    state = torch.load(weights_dir / "passt-s-kd-ap.486.pt", map_location="cpu", weights_only=True)
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and value.shape == current[key].shape}
    result = model.load_state_dict(compatible, strict=False)
    expected_head = {"head.1.weight", "head.1.bias", "head_dist.weight", "head_dist.bias"}
    missing = set(result.missing_keys)
    if missing != expected_head or result.unexpected_keys:
        raise RuntimeError(f"Unexpected PaSST checkpoint mismatch: missing={sorted(missing)}, unexpected={result.unexpected_keys}")
    return model, sorted(missing)


def weighted_masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    per_label = -(
        alpha * targets * F.logsigmoid(logits)
        + beta * (1.0 - targets) * F.logsigmoid(-logits)
    )
    observed = mask.sum().clamp_min(1.0)
    return (per_label * mask).sum() / observed


def compute_openmic_metrics(probabilities: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> dict[str, object]:
    """Paper-style macro ROC-AUC, AP, and ROC-nearest-point F1.

    Only observed labels are used. Classes missing either polarity in the
    capped validation subset are excluded and their count is reported.
    """
    auc_values: list[float] = []
    ap_values: list[float] = []
    f1_values: list[float] = []
    thresholds: list[float | None] = []
    for class_index in range(targets.shape[1]):
        observed = masks[:, class_index].astype(bool)
        truth = targets[observed, class_index].astype(np.int64)
        score = probabilities[observed, class_index].astype(np.float64)
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)
        if positives == 0 or negatives == 0:
            thresholds.append(None)
            continue

        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        sorted_truth = truth[order]
        true_positive = np.cumsum(sorted_truth)
        false_positive = np.cumsum(1 - sorted_truth)
        group_ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(sorted_score) - 1]
        tp = np.r_[0, true_positive[group_ends]].astype(np.float64)
        fp = np.r_[0, false_positive[group_ends]].astype(np.float64)
        roc_tpr = tp / positives
        roc_fpr = fp / negatives
        auc_values.append(float(np.trapezoid(roc_tpr, roc_fpr)))

        precision = tp[1:] / np.maximum(tp[1:] + fp[1:], 1.0)
        recall = roc_tpr[1:]
        ap_values.append(float(np.sum(np.diff(np.r_[0.0, recall]) * precision)))

        closest = int(np.argmin((1.0 - roc_tpr[1:]) ** 2 + roc_fpr[1:] ** 2))
        selected = group_ends[closest]
        threshold = float(sorted_score[selected])
        predicted_positive = sorted_score[: selected + 1]
        selected_tp = int(sorted_truth[: selected + 1].sum())
        selected_fp = int(len(predicted_positive) - selected_tp)
        selected_fn = positives - selected_tp
        denominator = 2 * selected_tp + selected_fp + selected_fn
        f1_values.append(float(2 * selected_tp / denominator) if denominator else 0.0)
        thresholds.append(threshold)

    def macro_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "validation_macro_roc_auc": macro_or_none(auc_values),
        "validation_macro_average_precision": macro_or_none(ap_values),
        "validation_macro_f1_roc_nearest": macro_or_none(f1_values),
        "validation_scored_classes": len(auc_values),
        "validation_class_thresholds_roc_nearest": thresholds,
    }


class MusicTalkBranches(nn.Module):
    def __init__(self, passt: nn.Module, panns: nn.Module, mel_frontend: nn.Module, bcp_drop: int):
        super().__init__()
        self.passt = passt
        self.panns = panns
        self.mel_frontend = mel_frontend
        self.bcp_drop = bcp_drop
        self.fusion = nn.Linear(40, 20)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # The smoke uses fixed pretrained feature extractors; update only
        # the OpenMIC heads and final fusion layer.
        self.mel_frontend.eval()
        self.panns.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        mel = self.mel_frontend(waveform).unsqueeze(1)
        keep_mask = None
        if self.training:
            # The 12 PaSST frequency tokens use overlapping 16-bin kernels
            # with stride 10. Recover positive Mel power from PaSST's log scale
            # and use the paper's weighted-without-replacement BCP-WP rule.
            mel_power = torch.exp(mel.squeeze(1) * 5.0 - 4.5)
            bands = [slice(i * 10, min(i * 10 + 16, mel_power.shape[1])) for i in range(12)]
            brightness = patch_brightness(mel_power, bands, frequency_dim=1)
            keep_mask = bcp_wp_keep_mask(brightness, self.bcp_drop, training=True)

        with torch.no_grad():
            passt_features = self.passt.forward_features(mel, bcp_keep_mask=keep_mask)
            if isinstance(passt_features, tuple):
                passt_features = (passt_features[0] + passt_features[1]) / 2.0
            panns_outputs = self.panns(waveform)
            panns_embedding = panns_outputs["embedding"]

        passt_logits = self.passt.head(passt_features)
        panns_logits = self.panns.fc_audioset(panns_embedding)
        branch_probabilities = torch.cat((torch.sigmoid(passt_logits), torch.sigmoid(panns_logits)), dim=1)
        return self.fusion(branch_probabilities), keep_mask


def main() -> None:
    args = parse_args()
    if args.epochs != 1:
        raise SystemExit("This smoke runner is capped to exactly one epoch; increase only after reviewing its result.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the selected PyTorch environment.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda")

    stage_start = time.perf_counter()
    all_ids, targets, masks, index = load_openmic_arrays(args.dataset_dir)
    train_ids, val_ids = get_capped_split(args.dataset_dir, args.seed, args.max_train, args.max_val)
    print(f"DATA_READY n_train={len(train_ids)} n_val={len(val_ids)} elapsed={time.perf_counter()-stage_start:.2f}s", flush=True)
    train_dataset = OpenMICAudio(args.dataset_dir, train_ids, targets, masks, index)
    val_dataset = OpenMICAudio(args.dataset_dir, val_ids, targets, masks, index)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stage_start = time.perf_counter()
    print("LOADING_PASST", flush=True)
    passt, passt_missing = load_passt_model(args.weights_dir)
    print(f"PASST_READY elapsed={time.perf_counter()-stage_start:.2f}s", flush=True)
    stage_start = time.perf_counter()
    print("LOADING_PANNS", flush=True)
    panns = load_panns_model(args.weights_dir, device)
    print(f"PANNS_READY elapsed={time.perf_counter()-stage_start:.2f}s", flush=True)
    stage_start = time.perf_counter()
    print("BUILDING_FRONTEND_AND_HEADS", flush=True)
    frontend = AugmentMelSTFT(
        n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
        freqm=0, timem=0, htk=False, fmin=0.0, fmax=15000.0, norm=1,
        fmin_aug_range=10, fmax_aug_range=2000,
    )
    model = MusicTalkBranches(passt, panns, frontend, args.bcp_drop).to(device)
    print(f"MODELS_ON_DEVICE elapsed={time.perf_counter()-stage_start:.2f}s", flush=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.passt.head[1].parameters():
        parameter.requires_grad_(True)
    for parameter in model.panns.fc_audioset.parameters():
        parameter.requires_grad_(True)
    for parameter in model.fusion.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-4,
    )

    train_rows = np.asarray([index[sample_id] for sample_id in train_ids], dtype=np.int64)
    observed = masks[train_rows].astype(np.float32)
    labels = targets[train_rows]
    n_observed = observed.sum(axis=0)
    positives = (observed * labels).sum(axis=0)
    negatives = (observed * (1.0 - labels)).sum(axis=0)
    alpha = torch.as_tensor(n_observed / (2.0 * np.maximum(positives, 1.0)), device=device)
    beta = torch.as_tensor(n_observed / (2.0 * np.maximum(negatives, 1.0)), device=device)

    train_start = time.perf_counter()
    model.train()
    train_loss_sum = 0.0
    train_observed = 0.0
    for batch_index, (waveform, batch_targets, batch_mask, _) in enumerate(train_loader, start=1):
        batch_start = time.perf_counter()
        waveform = waveform.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)
        logits, keep_mask = model(waveform)
        print(f"TRAIN_FORWARD batch={batch_index}/{len(train_loader)} elapsed={time.perf_counter()-batch_start:.2f}s", flush=True)
        loss = weighted_masked_bce(logits, batch_targets, batch_mask, alpha, beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        n_valid = float(batch_mask.sum().item())
        train_loss_sum += float(loss.detach().item()) * n_valid
        train_observed += n_valid
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    print("VALIDATING", flush=True)
    model.eval()
    val_loss_sum = 0.0
    val_observed = 0.0
    val_prob_min = float("inf")
    val_prob_max = float("-inf")
    val_probabilities: list[np.ndarray] = []
    val_targets: list[np.ndarray] = []
    val_masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, (waveform, batch_targets, batch_mask, _) in enumerate(val_loader, start=1):
            batch_start = time.perf_counter()
            waveform = waveform.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            logits, _ = model(waveform)
            print(f"VAL_FORWARD batch={batch_index}/{len(val_loader)} elapsed={time.perf_counter()-batch_start:.2f}s", flush=True)
            loss = weighted_masked_bce(logits, batch_targets, batch_mask, alpha, beta)
            probabilities = torch.sigmoid(logits)
            val_probabilities.append(probabilities.cpu().numpy())
            val_targets.append(batch_targets.cpu().numpy())
            val_masks.append(batch_mask.cpu().numpy().astype(bool))
            n_valid = float(batch_mask.sum().item())
            val_loss_sum += float(loss.item()) * n_valid
            val_observed += n_valid
            val_prob_min = min(val_prob_min, float(probabilities.min().item()))
            val_prob_max = max(val_prob_max, float(probabilities.max().item()))
    torch.cuda.synchronize()
    validation_metrics = compute_openmic_metrics(
        np.concatenate(val_probabilities),
        np.concatenate(val_targets),
        np.concatenate(val_masks),
    )

    result = {
        "status": "completed",
        "dataset": "OpenMIC-2018 official split01_train, raw OGG audio",
        "seed": args.seed,
        "epochs": 1,
        "train_samples": len(train_ids),
        "validation_samples": len(val_ids),
        "batch_size": args.batch_size,
        "training_loss_weighted_masked_bce": train_loss_sum / max(train_observed, 1.0),
        "validation_loss_weighted_masked_bce": val_loss_sum / max(val_observed, 1.0),
        "validation_observed_labels": int(val_observed),
        "validation_probability_range": [val_prob_min, val_prob_max],
        **validation_metrics,
        "train_seconds": train_seconds,
        "device": torch.cuda.get_device_name(device),
        "passt_checkpoint_missing_head_keys": passt_missing,
        "bcp_wp_frequency_patches": 12,
        "bcp_wp_dropped_patches": args.bcp_drop,
        "frozen_backbones": True,
        "official_test_evaluated": False,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "note": "One-epoch integration smoke only; frozen AudioSet backbones and capped subset are not the paper's full training protocol.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args) | {"output_dir": str(args.output_dir)}, default=str, indent=2), encoding="utf-8")
    torch.save(
        {
            "seed": args.seed,
            "passt_openmic_head": model.passt.head[1].state_dict(),
            "panns_openmic_head": model.panns.fc_audioset.state_dict(),
            "fusion": model.fusion.state_dict(),
            "metrics": result,
        },
        args.output_dir / "checkpoint_heads.pt",
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["training_loss_weighted_masked_bce", "validation_loss_weighted_masked_bce", "train_seconds"])
        writer.writeheader()
        writer.writerow({key: result[key] for key in writer.fieldnames})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
