"""
Methane Plume Detection - Training Script
Optimized for AMD 5600U (6-core CPU, integrated graphics)

This script trains both baseline and optimized models with energy measurement.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import argparse
from codecarbon import EmissionsTracker
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET
# ============================================================================

class CH4PlumeDataset(Dataset):
    """Dataset for methane plume detection"""
    
    def __init__(self, plume_ids, img_dir, mask_dir, img_size=256):
        self.plume_ids = plume_ids
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
    
    def __len__(self):
        return len(self.plume_ids)
    
    def __getitem__(self, idx):
        plume_id = self.plume_ids[idx]
        
        # Load image (RGB)
        img_path = self.img_dir / f"{plume_id}.npy"
        if img_path.exists():
            image = np.load(img_path)
        else:
            # Create synthetic data for demo
            image = np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)
        
        # Load mask
        mask_path = self.mask_dir / f"{plume_id}.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
        else:
            # Create synthetic mask for demo
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            # Add a small random plume
            if np.random.rand() > 0.3:
                y, x = np.random.randint(50, self.img_size-50, 2)
                size = np.random.randint(20, 50)
                mask[y:y+size, x:x+size] = 1.0
        
        # Convert to tensors
        image = torch.FloatTensor(image).permute(2, 0, 1)  # (3, H, W)
        mask = torch.FloatTensor(mask).unsqueeze(0)        # (1, H, W)
        
        return image, mask


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SimpleUNet(nn.Module):
    """Baseline U-Net model"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.sigmoid(self.out(dec1))


class OptimizedUNet(nn.Module):
    """Optimized lightweight U-Net for edge deployment"""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(OptimizedUNet, self).__init__()
        
        # Encoder (reduced channels)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        self.out = nn.Conv2d(32, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        bottleneck = self.bottleneck(self.pool(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.sigmoid(self.out(dec1))


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice Loss"""
    
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


# ============================================================================
# METRICS
# ============================================================================

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def calculate_f1(pred, target, threshold=0.5):
    """Calculate F1 score"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.item()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)
        total_f1 += calculate_f1(outputs, masks)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{calculate_iou(outputs, masks):.4f}'
        })
    
    n = len(dataloader)
    return total_loss / n, total_iou / n, total_f1 / n


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
            total_f1 += calculate_f1(outputs, masks)
    
    n = len(dataloader)
    return total_loss / n, total_iou / n, total_f1 / n


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(model_type='baseline', epochs=10, batch_size=4, lr=1e-4):
    """
    Train a model with energy tracking
    
    Args:
        model_type: 'baseline' or 'optimized'
        epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
    """
    
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*80}\n")
    
    # Device
    device = torch.device('cpu')  # AMD 5600U - use CPU
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create directories
    Path('dataset/images').mkdir(parents=True, exist_ok=True)
    Path('dataset/masks').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load or create synthetic data
    print("\n[1/6] Loading dataset...")
    
    # Try to load real plume IDs
    metadata_file = Path('ch4_dataset/all_plumes_metadata.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            all_plumes = json.load(f)
        plume_ids = [p['plume_id'] for p in all_plumes[:100]]  # Use first 100
        print(f"   Loaded {len(plume_ids)} plume IDs from metadata")
    else:
        # Create synthetic IDs
        plume_ids = [f"plume_{i:04d}" for i in range(100)]
        print(f"   Created {len(plume_ids)} synthetic plume IDs")
    
    # Split data
    split_idx = int(0.8 * len(plume_ids))
    train_ids = plume_ids[:split_idx]
    val_ids = plume_ids[split_idx:]
    
    print(f"   Training samples: {len(train_ids)}")
    print(f"   Validation samples: {len(val_ids)}")
    
    # Create datasets
    train_dataset = CH4PlumeDataset(train_ids, 'dataset/images', 'dataset/masks')
    val_dataset = CH4PlumeDataset(val_ids, 'dataset/images', 'dataset/masks')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n[2/6] Creating {model_type} model...")
    if model_type == 'baseline':
        model = SimpleUNet()
    else:
        model = OptimizedUNet()
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Start energy tracking
    print(f"\n[3/6] Starting energy tracking...")
    tracker = EmissionsTracker(
        project_name=f"methane_detection_{model_type}",
        output_dir="results",
        output_file=f"emissions_{model_type}.csv",
        log_level='warning'
    )
    tracker.start()
    
    # Training loop
    print(f"\n[4/6] Training for {epochs} epochs...")
    best_val_iou = 0
    training_start = time.time()
    
    history = {
        'train_loss': [], 'train_iou': [], 'train_f1': [],
        'val_loss': [], 'val_iou': [], 'val_f1': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_iou, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)
        
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_f1': val_f1,
            }, f'models/{model_type}_best.pth')
            print(f"  ✓ Saved best model (IoU: {val_iou:.4f})")
    
    training_time = time.time() - training_start
    
    # Stop energy tracking
    print(f"\n[5/6] Stopping energy tracking...")
    emissions = tracker.stop()
    
    # Save final model
    print(f"\n[6/6] Saving final results...")
    torch.save(model.state_dict(), f'models/{model_type}_final.pth')
    
    # Save training history
    with open(f'results/{model_type}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - {model_type.upper()}")
    print(f"{'='*80}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best validation F1: {max(history['val_f1']):.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Energy consumed: {emissions:.6f} kWh")
    print(f"CO2 emissions: Check results/emissions_{model_type}.csv")
    print(f"Model saved to: models/{model_type}_best.pth")
    print(f"{'='*80}\n")
    
    return model, history, emissions


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train methane plume detection model')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'optimized'],
                      help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--both', action='store_true', help='Train both baseline and optimized')
    
    args = parser.parse_args()
    
    if args.both:
        print("\n🚀 Training BOTH models for comparison...\n")
        
        # Train baseline
        model_base, hist_base, em_base = train_model('baseline', args.epochs, args.batch_size, args.lr)
        
        # Train optimized
        model_opt, hist_opt, em_opt = train_model('optimized', args.epochs, args.batch_size, args.lr)
        
        # Compare
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        print(f"Baseline Model:")
        print(f"  Parameters: {sum(p.numel() for p in model_base.parameters()):,}")
        print(f"  Best Val IoU: {max(hist_base['val_iou']):.4f}")
        print(f"  Energy: {em_base:.6f} kWh")
        print(f"\nOptimized Model:")
        print(f"  Parameters: {sum(p.numel() for p in model_opt.parameters()):,}")
        print(f"  Best Val IoU: {max(hist_opt['val_iou']):.4f}")
        print(f"  Energy: {em_opt:.6f} kWh")
        print(f"\nImprovement:")
        param_reduction = (1 - sum(p.numel() for p in model_opt.parameters()) / sum(p.numel() for p in model_base.parameters())) * 100
        energy_reduction = (1 - em_opt / em_base) * 100 if em_base > 0 else 0
        print(f"  Parameters reduced: {param_reduction:.1f}%")
        print(f"  Energy reduced: {energy_reduction:.1f}%")
        print(f"{'='*80}\n")
        
    else:
        model, history, emissions = train_model(args.model, args.epochs, args.batch_size, args.lr)
