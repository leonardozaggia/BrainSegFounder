import os
import numpy as np
import torch
import monai.networks.nets
from torch.utils.data import DataLoader, Subset
from dataset.ATLASDataset import ATLASDataset
from data.split_data import get_split_indices
import argparse

class ATLASPredictor(torch.nn.Module):
    """Minimal stub used to unpickle old checkpoints."""

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):  # pragma: no cover - legacy only
        if self.model is None:
            raise RuntimeError("ATLASPredictor missing wrapped model")
        return self.model(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference with pretrained model.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained model checkpoint.')
    parser.add_argument('--output', type=str, required=True, help='Directory to save predictions.')
    parser.add_argument('--roi', nargs=3, type=int, default=[96, 96, 96], help='Input dimensions.')
    parser.add_argument('--in_channels', type=int, required=True, help='Number of input channels.')
    parser.add_argument('--out_channels', type=int, required=True, help='Number of output channels.')
    parser.add_argument('--feature_size', type=int, help='Patch embedding features.')
    parser.add_argument('--depths', nargs=4, type=int, help='SSL attention heads by layer.')
    parser.add_argument('--heads', nargs=4, type=int, help='Number of heads.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference.')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers.')
    parser.add_argument('--seed', type=int, help='Seed for data split.')
    return parser.parse_args()

def predict(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup dataset (using validation split as example)
    data_entities=[{'subject': '', 'session': '', 'suffix': 'T1w', 'space': 'MNI152NLin2009aSym'}]
    target_entities=[{'suffix': 'mask', 'label': 'L', 'desc': 'T1lesion'}]
    data_derivatives_names=['ATLAS']
    target_derivatives_names=['ATLAS']
    root_dir = 'data/train/'

    dataset = ATLASDataset(
        data_entities=data_entities, target_entities=target_entities,
        data_derivatives_names=data_derivatives_names,
        target_derivatives_names=target_derivatives_names, root_dir=root_dir,
        transform=monai.transforms.Compose([
            monai.transforms.ToTensor(),
            monai.transforms.Resize(args.roi)
        ]),
        target_transform=monai.transforms.Compose([
            monai.transforms.ToTensor(),
            monai.transforms.Resize(args.roi)
        ])
    )

    # Use only validation indices for prediction
    _, val_indices = get_split_indices(dataset, split_fraction=0.8, seed=args.seed)
    val_subset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load model
    model = monai.networks.nets.SwinUNETR(
        img_size=args.roi,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=True,
        depths=args.depths,
        num_heads=args.heads,
        drop_rate=0.0
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    if "module." in list(state_dict.keys())[0]:
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "swinViT.")] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Inference loop
    with torch.no_grad():
        for i, (image, _) in enumerate(val_loader):
            image = image.to(device)
            pred = model(image)
            pred = torch.sigmoid(pred)  # If using sigmoid in loss
            pred_np = pred.cpu().numpy()
            np.save(os.path.join(args.output, f'prediction_{i}.npy'), pred_np)
            print(f"Saved prediction {i} to {args.output}")

if __name__ == '__main__':
    args = parse_args()
    predict(args)