import argparse
import os
from glob import glob
import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


class DummyDDP(torch.nn.Module):
    """Simple stub for DataParallel/DistributedDataParallel wrappers."""

    def __init__(self, module=None, *args, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# monkey patch wrappers so torch.load doesn't require distributed setup
torch.nn.DataParallel = DummyDDP
torch.nn.parallel.DataParallel = DummyDDP
torch.nn.parallel.DistributedDataParallel = DummyDDP


def load_checkpoint(path: str, device: torch.device, model: torch.nn.Module) -> torch.nn.Module:
    """Load checkpoint weights into the provided model."""
    ckpt = torch.load(path, map_location=device)

    # when entire model object saved
    if isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        new_state[key] = v
    model.load_state_dict(new_state, strict=False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ATLAS lesion segmentation inference")
    parser.add_argument("--input_dir", required=True, help="directory with T1w NIfTI files")
    parser.add_argument("--checkpoint", required=True, help="path to finetuned checkpoint")
    parser.add_argument("--output_dir", required=True, help="directory to save masks")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[128, 128, 128], help="inference patch size")
    parser.add_argument("--overlap", type=float, default=0.5, help="sliding window overlap fraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=tuple(args.patch_size),
        in_channels=1,
        out_channels=1,
        feature_size=48,
    ).to(device)
    model.eval()
    model = load_checkpoint(args.checkpoint, device, model)

    files = sorted(glob(os.path.join(args.input_dir, "*.nii")) + glob(os.path.join(args.input_dir, "*.nii.gz")))
    if not files:
        print(f"No NIfTI files found in {args.input_dir}")
        return

    processed = 0
    for fpath in files:
        print(f"Processing {os.path.basename(fpath)}")
        img = nib.load(fpath)
        data = img.get_fdata().astype(np.float32)
        tensor = torch.from_numpy(data)[None, None]
        tensor = tensor.to(device)
        with torch.no_grad():
            probs = torch.sigmoid(
                sliding_window_inference(tensor, args.patch_size, 1, model, overlap=args.overlap)
            )
        mask = (probs > 0.5).float()[0, 0].cpu().numpy().astype(np.uint8)
        out_img = nib.Nifti1Image(mask, img.affine)
        base = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(args.output_dir, f"{base}_mask.nii.gz")
        nib.save(out_img, out_path)
        processed += 1

    print(f"Finished inference on {processed} volume{'s' if processed != 1 else ''}.")


if __name__ == "__main__":
    main()
