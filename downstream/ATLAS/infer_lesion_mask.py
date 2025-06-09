vimport torch
import monai
import numpy as np
import nibabel as nib
import argparse
import os

"""
To use this script run:

python infer_lesion_mask.py --checkpoint /path/to/finetune_best_val_loss.pt \
    --image /path/to/patient.nii.gz \
    --output /path/to/predicted_mask.nii.gz \
    --roi 96 96 96

"""

def load_image(image_path, roi):
    img = nib.load(image_path)
    data = img.get_fdata()
    # Add channel dimension if needed
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    # Resize to ROI
    transform = monai.transforms.Compose([
        monai.transforms.ToTensor(),
        monai.transforms.Resize(roi)
    ])
    tensor = transform(data)
    return tensor, img.affine

def save_mask(mask, affine, output_path):
    mask = mask.astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, affine), output_path)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = torch.load(args.checkpoint, map_location=device)
    model.eval()
    # Load and preprocess image
    image_tensor, affine = load_image(args.image, args.roi)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        pred = model(image_tensor)
        pred_mask = torch.sigmoid(pred).cpu().numpy()[0, 0]  # Assuming out_channels=1
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    # Save mask
    save_mask(pred_mask, affine, args.output)
    print(f"Saved predicted lesion mask to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer lesion mask from MRI using finetuned model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to finetuned model checkpoint (.pt)')
    parser.add_argument('--image', type=str, required=True, help='Path to input MRI image (NIfTI)')
    parser.add_argument('--output', type=str, required=True, help='Path to save predicted mask (NIfTI)')
    parser.add_argument('--roi', nargs=3, type=int, default=[96, 96, 96], help='ROI size used during training')
    args = parser.parse_args()
    main(args)