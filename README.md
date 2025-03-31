Project Overview
This project implements an image-to-image translation model that can add realistic colors to grayscale images. The model is trained to predict the 'a' and 'b' color channels of the LAB color space based on the lightness ('L') channel from grayscale input.
Features

Automatic colorization of black and white photos
U-Net architecture with skip connections for preserving spatial details
Training on a subset of the COCO dataset with automatic download
Custom data loader for working with the LAB color space
Visualization of original, grayscale, and colorized results

Installation
Clone the repository and install the required dependencies:
git clone https://github.com/yourusername/bw-image-colorization.git
cd bw-image-colorization

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch>=1.7.0 torchvision>=0.8.1 numpy>=1.19.2 Pillow>=8.0.1 matplotlib>=3.3.2 scikit-image>=0.17.2 tqdm>=4.51.0 scikit-learn

Project Structure
colorization_project/
├── data/
│   └── download_dataset.py     # Script to download COCO dataset subset
├── models/
│   └── unet.py                 # U-Net architecture definition
├── utils/
│   └── data_utils.py           # Dataset and dataloader utilities
├── checkpoints/                # Model checkpoints saved here
├── train.py                    # Training script
├── test.py                     # Testing and inference script
└── requirements.txt            # Project dependencies

Dataset
The project uses a subset of the COCO dataset, which will be automatically downloaded and prepared when you run the training script for the first time. The dataset is split into training (80%) and validation (20%) sets.
To use your own dataset:

Create directories data/your_dataset/train and data/your_dataset/val
Place your images in the respective directories
Update the data_dir in the config section of train.py

Training
To train the colorization model:
bash
python train.py

The script will:

Check if the dataset exists and download it if needed
Train the U-Net model for the specified number of epochs (default: 30)
Save model checkpoints in the checkpoints directory
Plot and save the training/validation loss curves

Testing and Inference
To colorize a grayscale image using the trained model:
bash
python test.py --image path/to/your/grayscale_image.jpg --output colorized_result.jpg

Options:

--model: Path to the trained model (default: checkpoints/colorization_model_final.pth)
--image: Path to the grayscale image to colorize (required)
--output: Path to save the colorized image (optional)
--no-display: Do not display the result (optional)

Model Architecture
The project implements a U-Net architecture specifically designed for image colorization:

Encoder: A series of convolutional layers that downsample the image and extract features
Decoder: Transposed convolutions that upsample the features back to the original image size
Skip connections: Connect encoder layers to decoder layers to preserve spatial information
Output: Predicts 'a' and 'b' channels of the LAB color space (2 channels)

Results
The model can produce reasonable colorization results after training for about 30 epochs. Results can be improved by:

Training for more epochs (50-100)
Using a larger dataset
Implementing additional techniques like GANs for more realistic colorization

Customization
You can modify training parameters in the train.py file:
python
config = {
    'batch_size': 16,  # Increase/decrease based on your GPU
    'img_size': 256,   # Image resolution
    'lr': 2e-4,        # Learning rate
    'epochs': 30,      # Number of training epochs
    # ...other parameters
}
