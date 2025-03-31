import os
import zipfile
import urllib.request
from tqdm import tqdm

def download_coco_subset():
    """
    Downloads a subset of COCO dataset for image colorization
    and organizes it into train/val folders.
    """
    # Create directories
    os.makedirs('data/coco_subset', exist_ok=True)
    os.makedirs('data/coco_subset/train', exist_ok=True)
    os.makedirs('data/coco_subset/val', exist_ok=True)
    
    # Download COCO 2017 val images (small subset for this example)
    coco_url = 'http://images.cocodataset.org/zips/val2017.zip'
    zip_path = 'data/val2017.zip'
    
    print(f"Downloading COCO validation images from {coco_url}...")
    
    # Download with progress bar
    with tqdm(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urllib.request.urlretrieve(
            coco_url, zip_path,
            reporthook=lambda b, bsize, tsize: t.update(bsize if tsize < 0 else min(bsize, tsize - t.n))
        )
    
    # Extract zip file
    print("Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/')
    
    # Move and split files for train/val
    import shutil
    from sklearn.model_selection import train_test_split
    
    # Get all image files
    all_images = [f for f in os.listdir('data/val2017') 
                if os.path.isfile(os.path.join('data/val2017', f)) 
                and f.endswith('.jpg')]
    
    # Split into train (80%) and validation (20%)
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Copy files to respective directories
    print("Organizing dataset into train/val splits...")
    for img in tqdm(train_images):
        shutil.copy(
            os.path.join('data/val2017', img),
            os.path.join('data/coco_subset/train', img)
        )
        
    for img in tqdm(val_images):
        shutil.copy(
            os.path.join('data/val2017', img),
            os.path.join('data/coco_subset/val', img)
        )
    
    # Clean up
    print("Cleaning up...")
    os.remove(zip_path)
    shutil.rmtree('data/val2017')
    
    print(f"Dataset prepared: {len(train_images)} training images, {len(val_images)} validation images")
    print(f"Dataset location: data/coco_subset/")

if __name__ == "__main__":
    download_coco_subset()
