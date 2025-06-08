import os
import argparse
import nibabel as nib
import numpy as np
import torch
import monai.transforms
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from torch.utils.data import DataLoader

# allow running this script from the repository root or from within the
# ``downstream/ATLAS`` directory by using an absolute import path.
from dataset.ATLASDataset import ATLASDataset


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on the ATLAS dataset using a finetuned model")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to finetuned model weights")
    # allow the user to pass either --data_dir or --input_dir
    parser.add_argument(
        "--data_dir",
        "--input_dir",
        dest="data_dir",
        default="data/test/",
        type=str,
        help="Directory containing ATLAS test data",
    )
    parser.add_argument(
        "--image",
        "--input_file",
        dest="image",
        default=None,
        type=str,
        help="Path to a single T1-weighted image for inference",
    )
    parser.add_argument(
        "--output",
        "--output_dir",
        dest="output",
        default="atlas_predictions",
        type=str,
        help="Directory to save predictions",
    )
    parser.add_argument("--in_channels", default=1, type=int, help="Number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="Number of output channels")
    parser.add_argument("--feature_size", default=48, type=int, help="Patch embedding feature size")
    parser.add_argument("--depths", nargs=4, type=int, default=[2, 2, 2, 2], help="SwinUNETR depths")
    parser.add_argument("--heads", nargs=4, type=int, default=[3, 6, 12, 24], help="Attention heads by layer")
    parser.add_argument(
        "--roi",
        "--patch_size",
        dest="roi",
        nargs=3,
        type=int,
        default=[96, 96, 96],
        help="Inference window size (x y z)",
    )
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap fraction")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for inference")
    parser.add_argument("--num_workers", default=2, type=int, help="Data loader workers")
    return parser.parse_args()


def load_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load weights from a checkpoint path handling various checkpoint formats."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    elif hasattr(ckpt, "state_dict"):
        state_dict = ckpt.state_dict()
    else:
        raise RuntimeError("Unsupported checkpoint format")

    # remove common prefixes
    new_state = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name.replace("module.", "")
        if name.startswith("swinViT."):
            name = name
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)


def main() -> None:
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    transform = monai.transforms.Compose([
        monai.transforms.ToTensor(),
        monai.transforms.Resize(args.roi),
    ])

    if args.image is None:
        data_entities = [{"subject": "", "session": "", "suffix": "T1w", "space": "MNI152NLin2009aSym"}]
        target_entities = [{"suffix": "mask", "label": "L", "desc": "T1lesion"}]
        data_derivatives_names = ["ATLAS"]
        target_derivatives_names = ["ATLAS"]

        dataset = ATLASDataset(
            data_entities=data_entities,
            target_entities=target_entities,
            data_derivatives_names=data_derivatives_names,
            target_derivatives_names=target_derivatives_names,
            root_dir=args.data_dir,
            transform=transform,
            target_transform=transform,
        )

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        img_nii = nib.load(args.image)
        affine = img_nii.affine
        loader = [(transform(img_nii.get_fdata()), affine)]

    model = SwinUNETR(
        img_size=args.roi,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=True,
        depths=args.depths,
        num_heads=args.heads,
        drop_rate=0.0,
    ).to(device)

    load_weights(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if isinstance(batch, tuple):
                img, affine = batch if len(batch) == 2 else (batch[0], np.eye(4))
            else:
                img, affine = batch, np.eye(4)

            if img.dim() == 4:  # C,H,W,D
                img = img.unsqueeze(0)

            img = img.to(device)
            logits = sliding_window_inference(img, args.roi, 1, model, overlap=args.overlap)
            prob = torch.sigmoid(logits)
            seg = (prob > 0.5).float().cpu().numpy()[0, 0]

            if args.image is not None:
                if os.path.isdir(args.output):
                    base = os.path.splitext(os.path.basename(args.image))[0]
                    out_path = os.path.join(args.output, f"{base}_mask.nii.gz")
                else:
                    out_path = args.output
            else:
                out_path = os.path.join(args.output, f"pred_{idx:03d}.nii.gz")

            nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine), out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
