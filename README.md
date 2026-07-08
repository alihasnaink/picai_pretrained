# picai_pretrained

A generic PyTorch interface for loading and running the [PI-CAI Grand Challenge](https://pi-cai.grand-challenge.org/) pre-trained nnU-Net segmentation weights, without needing the full nnU-Net training framework.

The PI-CAI baseline models were trained for detection/segmentation of clinically significant prostate cancer from bpMRI (T2W / ADC / DWI). This repo strips the architecture and inference logic down to a plain `nn.Module` so the 5-fold ensemble can be dropped into any PyTorch pipeline with just a `state_dict` load.

## Features

- **`Generic_UNet`** — a standalone reimplementation of the nnU-Net architecture used by the PI-CAI baseline, so checkpoints load with `strict=True` and no dependency on the original `nnunet` package.
- **`PicaiEnsemble`** — loads all 5 cross-validation folds as a frozen, weight-averaged ensemble and exposes clean inference methods.
- **`verify.py`** — a standalone script to sanity-check that `architecture.py` is shape- and weight-compatible with a given checkpoint.
- Minimal preprocessing helpers (`to_tensor`, `znorm`) matching nnU-Net's default normalization.

## Repository structure

```
picai_pretrained/
├── __init__.py       # public exports: PicaiEnsemble, to_tensor, znorm
├── architecture.py    # Generic_UNet — nnU-Net-compatible architecture
├── ensemble.py         # PicaiEnsemble — loads and runs the 5-fold ensemble
├── classifier.py       # WIP — lesion-level classification head
├── utils.py             # to_tensor, znorm preprocessing helpers
└── verify.py             # CLI script to verify architecture <-> checkpoint compatibility
```

## Installation

```bash
git clone https://github.com/alihasnaink/picai_pretrained.git
cd picai_pretrained
pip install torch numpy
```

No `setup.py` / PyPI package yet — use it as a local module or copy the folder into your project.

### Getting the weights

This repo ships **only** the interface, not the weights themselves. Download the pre-trained PI-CAI baseline checkpoints separately from the official [PI-CAI Grand Challenge](https://pi-cai.grand-challenge.org/) resources, and arrange them as:

```
checkpoints/
├── fold_0/model_best.model
├── fold_1/model_best.model
├── fold_2/model_best.model
├── fold_3/model_best.model
└── fold_4/model_best.model
```

## Usage

### Load the ensemble and run inference

```python
import numpy as np
from picai_pretrained import PicaiEnsemble, to_tensor, znorm

# volume: [3, D, H, W] numpy array — T2W / ADC / DWI channels stacked, already resampled/cropped
volume = np.load("example_scan.npy")

# Preprocess (per-channel z-score over foreground, matching nnU-Net defaults)
volume = znorm(volume)
x = to_tensor(volume, device="cpu")  # -> [1, 3, D, H, W]

# Load all 5 folds as a frozen ensemble
model = PicaiEnsemble(checkpoint_dir="checkpoints", folds=(0, 1, 2, 3, 4), device="cpu")

probs = model(x)                      # [1, 2, D, H, W] softmax probabilities
mask = model.predict_mask(x)          # [1, D, H, W] binary segmentation
lesion_prob = model.predict_lesion_prob(x)  # [1, D, H, W] foreground probability map
```

- `x` must already be preprocessed/resampled to the shape the baseline model expects (default input shape used for verification: `(1, 3, 16, 320, 320)`).
- You can load a subset of folds (e.g. `folds=(0, 2)`) for faster, lower-memory inference at the cost of ensemble accuracy.
- Pass `device="cuda"` to run on GPU if available.

### Verify a checkpoint against the architecture

```bash
python verify.py --ckpt path/to/fold_0/model_best.model
```

This builds `Generic_UNet`, loads the checkpoint with `strict=True`, and runs a dummy forward pass to confirm the output shape matches expectations. If no checkpoint is found at the given path, it falls back to a shape-only sanity check.

## Model details

The default architecture configuration (`PicaiEnsemble.DEFAULT_CONFIG`) matches the PI-CAI baseline:

| Parameter | Value |
|---|---|
| Input channels | 3 (T2W, ADC, DWI) |
| Base features | 32 |
| Output classes | 2 (background, lesion) |
| Pooling stages | 6 |

## Status

- [x] `Generic_UNet` architecture
- [x] 5-fold ensemble inference (`PicaiEnsemble`)
- [x] Preprocessing utilities
- [x] Architecture/checkpoint verification script
- [ ] `classifier.py` — lesion-level classification head (in progress)
- [ ] PyPI packaging

## License

No license file has been added yet — all rights reserved by default until one is specified. Open an issue if you'd like to use this under a specific license.

## Acknowledgements

Built on top of the architecture and pre-trained weights released by the [PI-CAI Grand Challenge](https://pi-cai.grand-challenge.org/) organizers, and the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework their baseline is derived from.
