import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.unet import ColorizationModel
from utils.data_utils import get_data_loaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration
config = {
    'batch_size': 16,
    'img_size': 256,
    'lr': 2e-4,
    'epochs': 30,
    'data_dir': 'data/coco_subset',
    'checkpoints_dir': 'checkpoints',
    'log_interval': 100,
    'val_interval': 1,
    'save_interval': 5
}

def train_model():
    # Create checkpoint directory
    os.makedirs(config['checkpoints_dir'], exist_ok=True)
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        config['data_dir'], 
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # Initialize model
    model = ColorizationModel().to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        model.train()
        epoch_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [Train]") as pbar:
            for i, (L, ab) in enumerate(pbar):
                # Move to device
                L, ab = L.to(device), ab.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                ab_pred = model(L)
                loss = criterion(ab_pred, ab)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Log batch loss
                if (i + 1) % config['log_interval'] == 0:
                    print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Average training loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        if epoch % config['val_interval'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch}/{config['epochs']} [Validate]") as pbar:
                    for L, ab in pbar:
                        L, ab = L.to(device), ab.to(device)
                        ab_pred = model(L)
                        loss = criterion(ab_pred, ab)
                        val_loss += loss.item()
                        pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Average validation loss
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save model checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['checkpoints_dir'], f"colorization_model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if epoch % config['val_interval'] == 0 else None
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(config['checkpoints_dir'], "colorization_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    
    val_epochs = list(range(config['val_interval'], config['epochs'] + 1, config['val_interval']))
    if len(val_losses) > 0:
        plt.plot(val_epochs, val_losses, label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model, final_model_path

if __name__ == "__main__":
    # Check if dataset exists, download if not
    if not os.path.exists('data/coco_subset/train') or not os.path.exists('data/coco_subset/val'):
        print("Dataset not found. Downloading...")
        import sys
        sys.path.append('data')
        from download_dataset import download_coco_subset
        download_coco_subset()
    
    # Train the model
    model, model_path = train_model()
    print(f"Training completed! Model saved at: {model_path}")