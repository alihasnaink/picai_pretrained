import torch
import numpy as np


def to_tensor(volume: np.ndarray, device='cpu') -> torch.Tensor:
    """
    Convert [C, D, H, W] numpy array to float32 tensor with batch dim.
    Returns [1, C, D, H, W].
    """
    t = torch.from_numpy(volume.astype(np.float32))
    return t.unsqueeze(0).to(device)


def znorm(volume: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalisation — matches nnUNet's default."""
    out = np.zeros_like(volume, dtype=np.float32)
    for c in range(volume.shape[0]):
        ch = volume[c]
        mask = ch > 0  # normalise over foreground only
        mean, std = ch[mask].mean(), ch[mask].std()
        out[c] = (ch - mean) / (std + 1e-8)
    return out