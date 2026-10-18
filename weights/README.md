# MusicTalk pretrained weights

The weights are kept separate from source and datasets.

| Checkpoint | Source | Expected integrity |
|---|---|---|
| `ResNet38_mAP_0.434.pth` | [PANNs Zenodo v3](https://zenodo.org/records/3987831), file `ResNet38_mAP=0.434.pth` | MD5 `bf12f36aaabac4e0855e22d3c3239c1b`; 299.6 MB |
| `passt-s-kd-ap.486.pt` | [PaSST v.0.0.9 release](https://github.com/kkoutini/PaSST/releases/tag/v.0.0.9) | SHA-256 `5fce4c0ba7769f5ffa4e70b010309da73c86f689b786b480d7ed4f6f7e7f5c33`; 344,665,670 bytes. |

Both checkpoints are AudioSet-pretrained initialization for the paper's two branches; they are not MusicTalk-trained checkpoints. A reproduction run must record whether the classifier heads are replaced or fine-tuned.

## Integrity / structure checks

- PaSST SHA-256 verified; loaded with PyTorch `weights_only=True`: 159 state-dict entries, mono patch-projection shape `(768, 1, 16, 16)`, and the AudioSet head has 527 outputs. Its classifier must be replaced or adapted for OpenMIC's 20 tags.
- PANNs MD5 verified as `bf12f36aaabac4e0855e22d3c3239c1b`; loaded with PyTorch `weights_only=True`: 246 state-dict entries, ResNet38 AudioSet head `(527, 2048)`, and input stem `(64, 1, 3, 3)`. The 527-class head must be adapted for the 20 OpenMIC tags.
- OpenMIC's stored VGGish embeddings are 8-bit quantized to `[0,255]`; the official [TensorFlow VGGish postprocessor](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_postprocess.py) maps clipped embeddings from `[-2,2]` to that range. The smoke baseline reverses this affine quantization.
