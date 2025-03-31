import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms

from models.unet import ColorizationModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Load the trained colorization model."""
    model = ColorizationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def colorize_image(model, image_path, output_path=None, show=True):
    """Colorize a grayscale image using the trained model."""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Open image and convert to RGB (even if grayscale)
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_original = np.array(img)
    img = transform(img)
    
    # Convert to Lab color space
    img_lab = rgb2lab(img.permute(1, 2, 0).numpy())
    L = img_lab[:, :, 0] / 50.0 - 1.0  # Normalize to [-1, 1]
    
    # Create grayscale version for comparison
    img_gray = np.zeros_like(img_original)
    for i in range(3):
        img_gray[:, :, i] = rgb2lab(img_original)[:, :, 0]
    
    # Predict a and b channels
    with torch.no_grad():
        L_tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).to(device)
        ab_pred = model(L_tensor).cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Denormalize and combine with L channel
    ab_pred = ab_pred * 110.0
    
    # Original L channel (not normalized)
    L_original = img_lab[:, :, 0]
    
    # Combine channels
    colorized_lab = np.zeros((L_original.shape[0], L_original.shape[1], 3))
    colorized_lab[:, :, 0] = L_original
    colorized_lab[:, :, 1:] = ab_pred
    
    # Convert back to RGB
    colorized_rgb = lab2rgb(colorized_lab)
    
    # Ensure colorized image is in uint8 format
    colorized_rgb_uint8 = (colorized_rgb * 255).astype(np.uint8)
    
    # Save the colorized image if output path is provided
    if output_path:
        plt.imsave(output_path, colorized_rgb)
        print(f"Colorized image saved to {output_path}")
    
    # Display images
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Grayscale image
        axes[1].imshow(img_gray / 255.0)
        axes[1].set_title('Grayscale')
        axes[1].axis('off')
        
        # Colorized image
        axes[2].imshow(colorized_rgb)
        axes[2].set_title('Colorized')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('comparison.png')
        plt.show()
    
    return colorized_rgb_uint8

def main():
    parser = argparse.ArgumentParser(description='Test colorization model on grayscale images')
    parser.add_argument('--model', type=str, default='checkpoints/colorization_model_final.pth',
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the grayscale image to colorize')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the colorized image')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display the result')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Colorize image
    colorize_image(model, args.image, args.output, show=not args.no_display)

if __name__ == "__main__":
    main()