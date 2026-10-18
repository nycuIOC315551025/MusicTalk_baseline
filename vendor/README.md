# MusicTalk upstream implementation sources

These are source snapshots of the upstream model implementations used as references. The source files are included directly so a clone of this repository contains the code without needing nested Git repositories. No pretrained weights are included in these snapshots.

| Component | Local path | Upstream commit | Role |
|---|---|---|---|
| PaSST | `PaSST/` | `2a5c818afcc2a215b2a1aaf1ed8be71f89d43201` | Patchout Vision Transformer; official OpenMIC example uses 20 instrument tags and returns embeddings. |
| PANNs | `PANNs/` | `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4` | ResNet38 spectrogram branch; returns clipwise outputs and a 2048-dimensional embedding. |

The paper-specific BCP-WP patch retention, PaSST/ResNet38 fusion, and training protocol remain MusicTalk work and are not supplied by these upstream projects.

The PaSST snapshot includes the two local runtime edits described below. PANNs source is unchanged. Dataset archives, audio samples, and AudioSet ground-truth CSVs are not included.

## Local runtime additions

The short dual-branch runner uses an isolated package target at `python_deps/` (including `timm==0.4.12`, `torchlibrosa==0.0.4`, and `librosa==0.10.2.post1`) with the existing CUDA PyTorch environment. The package target keeps the shared thesis Python environment unchanged.

Two small compatibility adjustments are applied to the local copies only:

- PaSST disables optional Sacred Git/source metadata collection and accepts the per-example BCP-WP frequency keep mask in `forward_features`. Patchout and attention computations remain upstream.
- `torchlibrosa/stft.py` builds the periodic Hann window directly with SciPy and the Slaney-normalized mel filter bank with torchaudio. The corresponding librosa lazy wrappers stalled in this Windows environment. The filter-bank shape is `(n_fft // 2 + 1, n_mels)`; the frontend smoke verified output shapes against the PANN path.

The dual-branch run trains only the replaced 20-class branch heads and the 40-to-20 fusion layer; pretrained PaSST and ResNet38 feature backbones stay frozen. It is an execution pilot, not a full reproduction of the paper's optimization schedule.
