# 🎓 Complete Guide: Training a Methane Plume Detection Model

## 📚 Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [Dataset Overview](#dataset-overview)
3. [Training Pipeline](#training-pipeline)
4. [Model Architecture](#model-architecture)
5. [How Inference Works](#how-inference-works)
6. [Complete Code Examples](#complete-code-examples)

---

## 🎯 Understanding the Problem

### What Are We Detecting?
**Methane (CH4) plumes** are invisible gas emissions that appear in satellite/aerial imagery. While invisible to human eyes, special sensors can detect them.

### The Task
Build an AI model that can:
1. **Input**: Take a satellite/aerial RGB image
2. **Output**: Identify WHERE methane plumes are located (pixel-level masks)
3. **Bonus**: Estimate how much methane is being emitted

### Real-World Application
- Monitor oil & gas facilities for leaks
- Detect landfill emissions
- Track agricultural methane sources
- Regulatory compliance and climate monitoring

---

## 📊 Dataset Overview

### What You Have: 12,000 Methane Plume Detections

Each detection includes:

#### 1. **Input Image** (`rgb_png` URL)
```
Type: RGB satellite/aerial image
Format: PNG
Content: Shows the ground/facility from above
Size: Varies (typically 512x512 to 2048x2048 pixels)
What it shows: Buildings, pipelines, terrain, vegetation
```

#### 2. **Label Mask** (`plume_tif` or `plume_png` URL)
```
Type: Binary or grayscale mask
Format: GeoTIFF or PNG
Content: Shows WHERE the plume is
Values: 
  - 0 (black) = No plume
  - 255 (white) = Plume detected
```

#### 3. **Metadata**
```json
{
  "plume_id": "av320250707t150341-B",
  "emission_auto": 812.78,           // kg/hr emission rate
  "geometry_json": {
    "coordinates": [-81.898, 37.250]  // Lat/Lon location
  },
  "sector": "1B1a",                  // Oil & gas sector
  "wind_speed_avg_auto": 2.59,       // Wind speed m/s
  "instrument": "av3"                // Sensor type
}
```

---

## 🔄 Training Pipeline

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: DATA PREPARATION                                    │
├─────────────────────────────────────────────────────────────┤
│ Download images → Preprocess → Split train/val/test         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: TRAINING                                            │
├─────────────────────────────────────────────────────────────┤
│ Feed images+masks → Model learns patterns → Save weights    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: INFERENCE (DETECTION)                               │
├─────────────────────────────────────────────────────────────┤
│ New image → Trained model → Predicted plume mask            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Model Architecture

### Recommended: U-Net for Semantic Segmentation

**Why U-Net?**
- Excellent for image segmentation tasks
- Works well with limited data
- Can detect small plumes
- Fast inference

**Architecture Diagram:**
```
Input Image (512x512x3)
         ↓
    [Encoder] ← Extracts features (what's in the image?)
         ↓
    [Bottleneck] ← Deepest understanding
         ↓
    [Decoder] ← Reconstructs pixel-level predictions
         ↓
Output Mask (512x512x1) ← Where is the plume?
```

---

## 🚀 How Inference Works (How to Use the Trained Model)

### Simple Explanation

**Training:**
```python
# What the model learns:
Input:  [RGB Image of oil facility]
Label:  [Mask showing plume location]
→ Model learns: "When I see this pattern, plume is here"
```

**Inference (Using the model):**
```python
# After training:
Input:  [NEW RGB image - never seen before]
Output: [Predicted mask showing plume location]
→ Model thinks: "Based on what I learned, plume should be here"
```

### What You Pass to the Model

#### During Training:
```python
model.train(
    images=numpy_array,      # Shape: (batch_size, 512, 512, 3)
    masks=numpy_array        # Shape: (batch_size, 512, 512, 1)
)
```

#### During Inference:
```python
prediction = model.predict(
    new_image=numpy_array    # Shape: (1, 512, 512, 3)
)
# Output: prediction is a mask (1, 512, 512, 1)
```

---

## 💻 Complete Code Examples

### 1️⃣ Download Dataset

```python
"""
download_images.py - Download all RGB images and masks
"""
import requests
import json
from pathlib import Path
from tqdm import tqdm

def download_dataset():
    # Load metadata
    with open('ch4_dataset/all_plumes_metadata.json', 'r') as f:
        plumes = json.load(f)
    
    # Create directories
    rgb_dir = Path('dataset/images')
    mask_dir = Path('dataset/masks')
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    for i, plume in enumerate(tqdm(plumes)):
        plume_id = plume['plume_id']
        
        # Download RGB image
        rgb_url = plume.get('rgb_png')
        if rgb_url:
            response = requests.get(rgb_url)
            with open(rgb_dir / f"{plume_id}.png", 'wb') as f:
                f.write(response.content)
        
        # Download plume mask
        mask_url = plume.get('plume_png')
        if mask_url:
            response = requests.get(mask_url)
            with open(mask_dir / f"{plume_id}.png", 'wb') as f:
                f.write(response.content)

if __name__ == "__main__":
    download_dataset()
```

### 2️⃣ Prepare Dataset for Training

```python
"""
prepare_dataset.py - Create train/val/test splits
"""
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import json

# Load all plume metadata
with open('ch4_dataset/all_plumes_metadata.json', 'r') as f:
    plumes = json.load(f)

# Get plume IDs
plume_ids = [p['plume_id'] for p in plumes]

# Split: 70% train, 15% val, 15% test
train_ids, temp_ids = train_test_split(plume_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

print(f"Training: {len(train_ids)} samples")
print(f"Validation: {len(val_ids)} samples")
print(f"Test: {len(test_ids)} samples")

# Save splits
with open('dataset/train.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open('dataset/val.txt', 'w') as f:
    f.write('\n'.join(val_ids))

with open('dataset/test.txt', 'w') as f:
    f.write('\n'.join(test_ids))
```

### 3️⃣ Create Dataset Loader (PyTorch)

```python
"""
dataset.py - PyTorch Dataset class
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class CH4PlumeDataset(Dataset):
    """
    Dataset for methane plume detection
    
    What this does:
    - Loads RGB image and corresponding mask
    - Preprocesses them for the model
    - Returns paired image-mask samples
    """
    
    def __init__(self, ids_file, img_dir, mask_dir, img_size=512):
        # Load IDs from train.txt/val.txt/test.txt
        with open(ids_file, 'r') as f:
            self.ids = f.read().strip().split('\n')
        
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        plume_id = self.ids[idx]
        
        # Load RGB image
        img_path = self.img_dir / f"{plume_id}.png"
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        
        # Load mask
        mask_path = self.mask_dir / f"{plume_id}.png"
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask = mask.resize((self.img_size, self.img_size))
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]
        
        # Convert to PyTorch tensors
        image = torch.FloatTensor(image).permute(2, 0, 1)  # (3, 512, 512)
        mask = torch.FloatTensor(mask).unsqueeze(0)         # (1, 512, 512)
        
        return image, mask

# Usage:
# train_dataset = CH4PlumeDataset('dataset/train.txt', 'dataset/images', 'dataset/masks')
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

### 4️⃣ Build U-Net Model (PyTorch)

```python
"""
model.py - U-Net architecture for plume detection
"""
import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net for semantic segmentation
    
    Input: RGB image (3 channels)
    Output: Plume probability mask (1 channel)
    """
    
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final output layer
        self.out = nn.Conv2d(64, 1, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
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
        
        # Output
        out = self.sigmoid(self.out(dec1))
        
        return out
```

### 5️⃣ Training Script

```python
"""
train.py - Train the plume detection model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CH4PlumeDataset
from model import UNet

def train_model(epochs=50, batch_size=8, lr=0.001):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = CH4PlumeDataset('dataset/train.txt', 'dataset/images', 'dataset/masks')
    val_dataset = CH4PlumeDataset('dataset/val.txt', 'dataset/images', 'dataset/masks')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = UNet().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ Saved best model!")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model(epochs=50, batch_size=8, lr=0.001)
```

### 6️⃣ Inference Script (HOW TO USE THE MODEL)

```python
"""
detect.py - Use trained model to detect plumes in new images
"""
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import UNet

def detect_plume(image_path, model_path='best_model.pth'):
    """
    Detect methane plume in a new satellite image
    
    Args:
        image_path: Path to RGB satellite image
        model_path: Path to trained model weights
    
    Returns:
        prediction: Binary mask showing plume location
    """
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to model input size
    image_resized = image.resize((512, 512))
    image_array = np.array(image_resized) / 255.0
    
    # Convert to tensor
    image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert prediction to numpy
    prediction = prediction.cpu().squeeze().numpy()
    
    # Threshold to binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8)
    
    # Resize back to original size
    binary_mask_resized = Image.fromarray(binary_mask * 255).resize(original_size)
    
    return np.array(binary_mask_resized)

def visualize_detection(image_path, model_path='best_model.pth'):
    """
    Visualize the detection result
    """
    # Load original image
    original_image = Image.open(image_path)
    
    # Get prediction
    predicted_mask = detect_plume(image_path, model_path)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Input: Satellite Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(predicted_mask, cmap='hot')
    axes[1].set_title('Predicted Plume Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(predicted_mask, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay: Plume Detection')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_result.png')
    plt.show()
    
    # Check if plume detected
    plume_pixels = np.sum(predicted_mask > 0)
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    plume_percentage = (plume_pixels / total_pixels) * 100
    
    print(f"\n🔍 Detection Results:")
    print(f"   Plume detected: {'YES' if plume_pixels > 0 else 'NO'}")
    print(f"   Affected area: {plume_percentage:.2f}% of image")
    print(f"   Plume pixels: {plume_pixels:,} / {total_pixels:,}")
    
    return predicted_mask

# USAGE EXAMPLE:
if __name__ == "__main__":
    # Detect plume in a new image
    new_image = "path/to/new/satellite/image.png"
    mask = visualize_detection(new_image, model_path='best_model.pth')
```

---

## 🎯 Real-World Workflow

### Step-by-Step: From Raw Image to Detection

#### 1. **Training Phase** (You do this once)
```
1. Download 12,000 images + masks
2. Train model for 50 epochs (~4-8 hours on GPU)
3. Save model weights: best_model.pth
```

#### 2. **Deployment Phase** (Use the model)
```
Scenario: You get a new satellite image of an oil facility

1. Load the trained model
   model = load_model('best_model.pth')

2. Pass the new image
   prediction = model.predict(new_image)

3. Get the result
   - Binary mask showing where plume is
   - Calculate plume area
   - Generate alert if plume detected
```

### Example Use Cases

#### Use Case 1: Monitoring Pipeline
```python
# Monitor oil pipeline every day
for day in range(365):
    satellite_image = get_daily_image(pipeline_location)
    mask = detect_plume(satellite_image, 'best_model.pth')
    
    if has_plume(mask):
        send_alert("⚠️ Methane leak detected at pipeline!")
        log_detection(day, mask)
```

#### Use Case 2: Batch Processing
```python
# Process 1000 new images
images = glob.glob('new_images/*.png')

for img_path in images:
    mask = detect_plume(img_path)
    save_result(img_path, mask)
```

---

## 📈 Expected Performance

With 12,000 training samples, you can expect:

- **Accuracy**: 85-95% plume detection rate
- **False Positives**: 5-10% (model sees plume where there isn't one)
- **Inference Speed**: ~50-100 images/second on GPU
- **Model Size**: ~50-100 MB

---

## 🔧 Troubleshooting

### Common Issues

**Q: Model predicts all black (no plumes detected)**
- Check if masks are loaded correctly
- Verify images are normalized (0-1 range)
- Try different learning rate

**Q: Model predicts all white (everything is plume)**
- Class imbalance issue
- Use weighted loss function
- Try data augmentation

**Q: Low accuracy**
- Need more training epochs
- Try transfer learning (start with pretrained weights)
- Increase model complexity

---

## 🚀 Next Steps

1. **Download the dataset** → Run `download_images.py`
2. **Prepare data** → Run `prepare_dataset.py`
3. **Train model** → Run `train.py`
4. **Test on new image** → Run `detect.py`

---

## 📝 Summary

**What goes INTO the model:**
- RGB satellite/aerial images (PNG/JPEG)
- Size: 512x512 pixels
- 3 channels (Red, Green, Blue)

**What comes OUT of the model:**
- Binary mask (same size as input)
- Values: 0 (no plume) or 1 (plume detected)
- Can be visualized as heatmap

**Training teaches the model:**
"When you see these patterns in an image, there's likely a methane plume here"

**Inference uses that knowledge:**
"Given this new image, based on what I learned, plume is at these pixel locations"

---

**You're ready to build a methane detection system! 🎉**
