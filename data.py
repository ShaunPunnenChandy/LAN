import torch
from pathlib import Path
import torchvision

class Dataset(torch.utils.data.Dataset):
    def __init__(self, lq_dir, gt_dir, crop_size=256):
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir)
        self.crop_size = crop_size
        self.lq_paths = sorted(list(self.lq_dir.glob("*.png")))
        self.gt_paths = sorted(list(self.gt_dir.glob("*.png")))
        assert len(self.lq_paths) == len(self.gt_paths)
    
    def __len__(self):
        return len(self.lq_paths)
    
    def __getitem__(self, idx):
        lq_name = self.lq_paths[idx].stem
        gt_name = self.gt_paths[idx].stem
        assert lq_name == gt_name
        lq = torchvision.io.read_image(str(self.lq_paths[idx]))/255.0
        gt = torchvision.io.read_image(str(self.gt_paths[idx]))/255.0
        lq = lq[:, :self.crop_size, :self.crop_size]
        gt = gt[:, :self.crop_size, :self.crop_size]
        return lq, gt