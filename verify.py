import argparse
from pathlib import Path
import torch
from architecture import Generic_UNet

DEFAULT_CKPT = (
    "../models/fold_0/model_best.model"
)

def verify_architecture(
    checkpoint_path: str,
    input_shape=(1, 3, 16, 320, 320),
    expected_output_shape=(1, 2, 16, 320, 320),
) -> None:
    model = Generic_UNet()
    print("Built Generic_UNet from architecture.py")

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        incompatible = model.load_state_dict(state_dict, strict=True)
        print("Missing keys:   ", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)

        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Checkpoint is not strictly compatible with architecture.py")

        print(f"Checkpoint loaded successfully from: {ckpt_path}")
    else:
        print(f"Checkpoint not found at: {ckpt_path}")
        print("Skipping weight-compatibility check and running shape check only.")

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        out = model(dummy)

    print("Output shape:", tuple(out.shape))
    if tuple(out.shape) != tuple(expected_output_shape):
        raise RuntimeError(
            f"Output shape mismatch. Expected {expected_output_shape}, got {tuple(out.shape)}"
        )

    print("Architecture verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify architecture.py against checkpoint and output shape")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to checkpoint file")
    args = parser.parse_args()

    verify_architecture(checkpoint_path=args.ckpt)


if __name__ == "__main__":
    main()