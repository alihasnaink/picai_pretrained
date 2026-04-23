import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .architecture import Generic_UNet


class PicaiEnsemble(nn.Module):
    """
    Loads all 5 nnUNet folds as a weight-frozen ensemble.
    Inference returns averaged softmax probabilities.
    """

    DEFAULT_CONFIG = dict(
        input_channels       = 3,
        base_num_features    = 32,
        num_classes          = 2,
        num_pool             = 6,
        pool_op_kernel_sizes = [[1,2,2],[1,2,2],[2,2,2],[2,2,2],[1,2,2],[1,2,2]],
        conv_kernel_sizes    = [[1,3,3],[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
    )

    def __init__(self, checkpoint_dir: str, folds=(0,1,2,3,4), device='cpu'):
        """
        Args:
            checkpoint_dir: path to the folder containing fold_0 .. fold_4
            folds: which folds to load (default: all 5)
            device: 'cpu', 'cuda', 'cuda:0', etc.
        """
        super().__init__()
        self.device = torch.device(device)
        self.models = nn.ModuleList()

        for fold in folds:
            ckpt_path = os.path.join(checkpoint_dir, f'fold_{fold}', 'model_best.model')
            model = self._load_fold(ckpt_path)
            self.models.append(model)

        self.to(self.device)
        print(f"Loaded {len(self.models)} fold(s) from {checkpoint_dir}")

    def _load_fold(self, ckpt_path: str) -> Generic_UNet:
        model = Generic_UNet(**self.DEFAULT_CONFIG)
        ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        # Freeze — we never train these weights
        for p in model.parameters():
            p.requires_grad = False
        return model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, D, H, W] — T2/ADC/DWI stack, already preprocessed
        Returns:
            probs: [B, 2, D, H, W] — averaged softmax probabilities across folds
        """
        x = x.to(self.device)
        logits = torch.stack([m(x) for m in self.models], dim=0)  # [F, B, 2, D, H, W]
        avg_logits = logits.mean(dim=0)                            # [B, 2, D, H, W]
        return F.softmax(avg_logits, dim=1)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Returns binary segmentation mask [B, D, H, W]"""
        probs = self.forward(x)
        return probs.argmax(dim=1)

    @torch.no_grad()
    def predict_lesion_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns foreground probability map [B, D, H, W] — useful for PI-RADS scoring"""
        probs = self.forward(x)
        return probs[:, 1]  # channel 1 = foreground