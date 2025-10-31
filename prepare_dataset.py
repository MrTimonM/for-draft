"""
Download and prepare dataset for training
Downloads RGB images and masks from Carbon Mapper API
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
import time

def download_images(num_samples=100, output_dir='dataset'):
    """
    Download RGB images and masks from Carbon Mapper dataset
    
    Args:
        num_samples: number of samples to download (default 100 for quick start)
        output_dir: output directory
    """
    
    print(f"\n{'='*80}")
    print("CARBON MAPPER DATASET DOWNLOAD")
    print(f"{'='*80}\n")
    
    # Create directories
    img_dir = Path(output_dir) / 'images'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_file = Path('ch4_dataset/all_plumes_metadata.json')
    
    if not metadata_file.exists():
        # Try batch files
        batch_files = sorted(Path('ch4_dataset').glob('plumes_batch_*.json'))
        if not batch_files:
            print("❌ No metadata found. Please run download_ch4_dataset.py first")
            return
        
        print(f"Found {len(batch_files)} batch files")
        all_plumes = []
        for batch_file in batch_files:
            with open(batch_file, 'r') as f:
                all_plumes.extend(json.load(f))
    else:
        with open(metadata_file, 'r') as f:
            all_plumes = json.load(f)
    
    print(f"Total plumes available: {len(all_plumes)}")
    print(f"Downloading first {num_samples} samples...\n")
    
    # Download samples
    downloaded = 0
    failed = 0
    
    for i, plume in enumerate(tqdm(all_plumes[:num_samples], desc="Downloading")):
        plume_id = plume.get('plume_id', f'plume_{i}')
        
        try:
            # Download RGB image
            rgb_url = plume.get('rgb_png')
            if rgb_url:
                response = requests.get(rgb_url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    img = img.convert('RGB')
                    
                    # Save as numpy array
                    img_array = np.array(img) / 255.0  # Normalize
                    np.save(img_dir / f"{plume_id}.npy", img_array.astype(np.float32))
            
            # Download plume mask
            mask_url = plume.get('plume_png') or plume.get('plume_tif')
            if mask_url:
                response = requests.get(mask_url, timeout=10)
                if response.status_code == 200:
                    mask = Image.open(io.BytesIO(response.content))
                    mask = mask.convert('L')  # Grayscale
                    
                    # Save as numpy array
                    mask_array = np.array(mask) / 255.0  # Normalize to [0, 1]
                    np.save(mask_dir / f"{plume_id}.npy", mask_array.astype(np.float32))
            
            downloaded += 1
            time.sleep(0.1)  # Be nice to the API
            
        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first few errors
                print(f"\n  Error downloading {plume_id}: {e}")
    
    print(f"\n{'='*80}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully downloaded: {downloaded}/{num_samples}")
    print(f"Failed: {failed}")
    print(f"Images saved to: {img_dir}")
    print(f"Masks saved to: {mask_dir}")
    print(f"{'='*80}\n")


def create_synthetic_dataset(num_samples=100, output_dir='dataset'):
    """
    Create synthetic dataset for testing without downloading
    Useful for quick development and testing
    """
    
    print(f"\n{'='*80}")
    print("CREATING SYNTHETIC DATASET")
    print(f"{'='*80}\n")
    
    img_dir = Path(output_dir) / 'images'
    mask_dir = Path(output_dir) / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic samples...")
    
    for i in tqdm(range(num_samples), desc="Generating"):
        # Create synthetic RGB image (256x256)
        image = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.3
        
        # Create synthetic mask with plume
        mask = np.zeros((256, 256), dtype=np.float32)
        
        # 70% chance of having a plume
        if np.random.rand() > 0.3:
            # Random plume location and size
            y = np.random.randint(50, 206)
            x = np.random.randint(50, 206)
            size = np.random.randint(20, 60)
            
            # Create blob-like plume
            for dy in range(-size, size):
                for dx in range(-size, size):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < 256 and 0 <= xx < 256:
                        dist = np.sqrt(dy**2 + dx**2)
                        if dist < size:
                            mask[yy, xx] = max(0, 1 - dist/size + np.random.rand()*0.2)
            
            # Make image darker where plume is
            image[mask > 0.3] *= 0.7
        
        # Save
        plume_id = f"synthetic_{i:04d}"
        np.save(img_dir / f"{plume_id}.npy", image)
        np.save(mask_dir / f"{plume_id}.npy", mask)
    
    print(f"\n✓ Synthetic dataset created")
    print(f"  Images: {img_dir}")
    print(f"  Masks: {mask_dir}")
    print(f"{'='*80}\n")


def create_splits(dataset_dir='dataset', train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits
    
    Args:
        dataset_dir: directory containing images and masks
        train_ratio: proportion for training (default 0.7)
        val_ratio: proportion for validation (default 0.15)
        test_ratio: remaining for test
    """
    
    print(f"\n{'='*80}")
    print("CREATING DATA SPLITS")
    print(f"{'='*80}\n")
    
    img_dir = Path(dataset_dir) / 'images'
    
    # Get all image files
    image_files = list(img_dir.glob('*.npy'))
    plume_ids = [f.stem for f in image_files]
    
    if len(plume_ids) == 0:
        print("❌ No images found. Run download or create synthetic dataset first.")
        return
    
    print(f"Total samples: {len(plume_ids)}")
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(plume_ids)
    
    # Split
    n_train = int(len(plume_ids) * train_ratio)
    n_val = int(len(plume_ids) * val_ratio)
    
    train_ids = plume_ids[:n_train]
    val_ids = plume_ids[n_train:n_train + n_val]
    test_ids = plume_ids[n_train + n_val:]
    
    print(f"Training: {len(train_ids)} ({len(train_ids)/len(plume_ids)*100:.1f}%)")
    print(f"Validation: {len(val_ids)} ({len(val_ids)/len(plume_ids)*100:.1f}%)")
    print(f"Test: {len(test_ids)} ({len(test_ids)/len(plume_ids)*100:.1f}%)")
    
    # Save splits
    splits_dir = Path(dataset_dir)
    
    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(splits_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_ids))
    
    with open(splits_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_ids))
    
    print(f"\n✓ Splits saved to {splits_dir}")
    print(f"{'='*80}\n")


def verify_dataset(dataset_dir='dataset'):
    """Verify dataset integrity"""
    
    print(f"\n{'='*80}")
    print("VERIFYING DATASET")
    print(f"{'='*80}\n")
    
    img_dir = Path(dataset_dir) / 'images'
    mask_dir = Path(dataset_dir) / 'masks'
    
    # Check directories exist
    if not img_dir.exists():
        print("❌ Images directory not found")
        return False
    
    if not mask_dir.exists():
        print("❌ Masks directory not found")
        return False
    
    # Count files
    images = list(img_dir.glob('*.npy'))
    masks = list(mask_dir.glob('*.npy'))
    
    print(f"Images found: {len(images)}")
    print(f"Masks found: {len(masks)}")
    
    # Check matching pairs
    image_ids = {f.stem for f in images}
    mask_ids = {f.stem for f in masks}
    
    missing_masks = image_ids - mask_ids
    missing_images = mask_ids - image_ids
    
    if missing_masks:
        print(f"⚠️  {len(missing_masks)} images missing corresponding masks")
    
    if missing_images:
        print(f"⚠️  {len(missing_images)} masks missing corresponding images")
    
    matched = len(image_ids & mask_ids)
    print(f"\n✓ Matched pairs: {matched}")
    
    # Check a sample
    if matched > 0:
        sample_id = list(image_ids & mask_ids)[0]
        img = np.load(img_dir / f"{sample_id}.npy")
        mask = np.load(mask_dir / f"{sample_id}.npy")
        
        print(f"\nSample check ({sample_id}):")
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}, range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # Check splits
    for split_name in ['train', 'val', 'test']:
        split_file = Path(dataset_dir) / f'{split_name}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_ids = f.read().strip().split('\n')
            print(f"\n{split_name}.txt: {len(split_ids)} samples")
        else:
            print(f"\n⚠️  {split_name}.txt not found")
    
    print(f"\n{'='*80}\n")
    
    return matched > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--download', action='store_true', help='Download from Carbon Mapper')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic dataset')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--verify', action='store_true', help='Verify dataset')
    parser.add_argument('--splits', action='store_true', help='Create train/val/test splits')
    
    args = parser.parse_args()
    
    if args.download:
        download_images(num_samples=args.num_samples)
    
    if args.synthetic:
        create_synthetic_dataset(num_samples=args.num_samples)
    
    if args.splits or args.download or args.synthetic:
        create_splits()
    
    if args.verify or args.download or args.synthetic:
        verify_dataset()
    
    if not any([args.download, args.synthetic, args.verify, args.splits]):
        print("\nNo action specified. Use --help for options")
        print("\nQuick start:")
        print("  1. Create synthetic dataset: python prepare_dataset.py --synthetic --num-samples 100")
        print("  2. Or download real data: python prepare_dataset.py --download --num-samples 100")
        print("  3. Verify: python prepare_dataset.py --verify")
