"""Quick test script for trained model"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from train import OptimizedUNet

def test_model():
    print("Loading model...")
    model = OptimizedUNet()
    checkpoint = torch.load('models/optimized_best.pth', map_location='cpu')
    # Handle both checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Val IoU: {checkpoint.get('val_iou', 0):.4f}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print("Loading test samples...")
    # Load 5 random samples
    dataset_dir = Path('dataset')
    image_files = sorted(list((dataset_dir / 'images').glob('*.npy')))[:5]
    
    fig, axes = plt.subplots(5, 3, figsize=(12, 15))
    fig.suptitle('Model Predictions (First 5 Samples)', fontsize=16)
    
    for i, img_file in enumerate(image_files):
        # Load image and mask
        image = np.load(img_file)
        mask_file = dataset_dir / 'masks' / img_file.name
        mask = np.load(mask_file)
        
        # Prepare for model
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            pred = model(image_tensor)
            pred_mask = pred.squeeze().numpy()
        
        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Input Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction (max={pred_mask.max():.3f})')
        axes[i, 2].axis('off')
        
        # Print stats
        has_plume = mask.max() > 0.1
        pred_plume = pred_mask.max() > 0.5
        print(f"Sample {i+1}: GT={'Plume' if has_plume else 'Empty'}, "
              f"Pred={'Plume' if pred_plume else 'Empty'} "
              f"(max={pred_mask.max():.3f})")
    
    plt.tight_layout()
    plt.savefig('results/model_test_predictions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Predictions saved to results/model_test_predictions.png")
    plt.show()

if __name__ == '__main__':
    test_model()
