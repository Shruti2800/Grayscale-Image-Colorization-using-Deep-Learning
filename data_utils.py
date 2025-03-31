import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

class ColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None, split='train'):
        self.image_dir = image_dir
        self.split = split
        self.split_dir = os.path.join(image_dir, split)
        self.image_files = [f for f in os.listdir(self.split_dir) 
                            if os.path.isfile(os.path.join(self.split_dir, f)) 
                            and f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.split_dir, self.image_files[idx])
        
        # Load RGB image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Convert to Lab color space
        img_lab = rgb2lab(img.permute(1, 2, 0).numpy())
        
        # Extract L channel (grayscale intensity)
        L = img_lab[:, :, 0] / 50.0 - 1.0  # Normalize to [-1, 1]
        
        # Extract a and b channels (color information)
        ab = img_lab[:, :, 1:] / 110.0  # Normalize to [-1, 1] roughly
        
        return torch.tensor(L).unsqueeze(0).float(), torch.tensor(ab).permute(2, 0, 1).float()

def get_data_loaders(data_dir, batch_size=16, img_size=256, num_workers=4):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = ColorizationDataset(data_dir, transform=transform, split='train')
    val_dataset = ColorizationDataset(data_dir, transform=transform, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader