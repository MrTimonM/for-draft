"""
Real-Time Methane Plume Detection - Inference Script
Detects plumes in <100ms and generates alerts

Usage:
    python inference.py --image path/to/image.png
    python inference.py --batch path/to/images/ --model models/optimized_best.pth
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
import argparse
from pathlib import Path
import cv2

# Import model architecture from train.py
import sys
sys.path.append('.')
from train import SimpleUNet, OptimizedUNet


class MethaneDetector:
    """Real-time methane plume detector"""
    
    def __init__(self, model_path, model_type='optimized', device='cpu', threshold=0.5):
        """
        Initialize detector
        
        Args:
            model_path: path to trained model weights
            model_type: 'baseline' or 'optimized'
            device: 'cpu' or 'cuda'
            threshold: detection threshold (0-1)
        """
        self.device = torch.device(device)
        self.threshold = threshold
        self.img_size = 256
        
        # Load model
        print(f"Loading {model_type} model from {model_path}...")
        if model_type == 'baseline':
            self.model = SimpleUNet()
        else:
            self.model = OptimizedUNet()
        
        # Load weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("✓ Model loaded successfully")
        else:
            print(f"⚠ Model file not found: {model_path}")
            print("  Using randomly initialized model for demo")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Warmup the model for faster inference"""
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_input)
        print("✓ Warmup complete")
    
    def preprocess(self, image_path):
        """Load and preprocess image"""
        # Load image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Resize
        image = image.resize((self.img_size, self.img_size))
        
        # Convert to tensor
        image_np = np.array(image) / 255.0
        image_tensor = torch.FloatTensor(image_np).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, image_np
    
    def detect(self, image_path):
        """
        Detect methane plume in image
        
        Returns:
            dict with detection results
        """
        # Preprocess
        image_tensor, image_np = self.preprocess(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            pred_mask = self.model(image_tensor)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Post-process
        pred_mask_np = pred_mask.cpu().squeeze().numpy()
        binary_mask = (pred_mask_np > self.threshold).astype(np.uint8)
        
        # Calculate detection metrics
        plume_detected = binary_mask.sum() > 0
        plume_area_pixels = binary_mask.sum()
        plume_area_percent = (plume_area_pixels / (self.img_size ** 2)) * 100
        confidence = pred_mask_np.max()
        
        # Estimate emission rate (simplified)
        # Real implementation would use wind speed, concentration, etc.
        estimated_emission = self._estimate_emission(plume_area_pixels, confidence)
        
        results = {
            'plume_detected': bool(plume_detected),
            'confidence': float(confidence),
            'plume_area_pixels': int(plume_area_pixels),
            'plume_area_percent': float(plume_area_percent),
            'estimated_emission_kg_hr': float(estimated_emission),
            'inference_time_ms': float(inference_time),
            'image': image_np,
            'pred_mask': pred_mask_np,
            'binary_mask': binary_mask
        }
        
        return results
    
    def _estimate_emission(self, area_pixels, confidence):
        """
        Estimate emission rate based on plume size
        
        This is a simplified estimation. Real implementation would use:
        - Wind speed and direction
        - Plume concentration values
        - Sensor calibration data
        - Atmospheric conditions
        """
        # Rough estimation: larger plume = higher emission
        # Typical range: 10-5000 kg/hr
        area_percent = (area_pixels / (self.img_size ** 2)) * 100
        
        if area_percent < 0.1:
            return 0.0
        
        # Simple linear model
        base_emission = area_percent * 20  # kg/hr per % area
        confidence_factor = confidence
        
        estimated = base_emission * confidence_factor
        
        return min(estimated, 5000)  # Cap at 5000 kg/hr
    
    def generate_alert(self, results, image_path):
        """Generate alert if plume detected"""
        
        if not results['plume_detected']:
            return None
        
        alert = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'ALERT',
            'message': 'Methane plume detected!',
            'location': str(image_path),
            'confidence': results['confidence'],
            'plume_area_percent': results['plume_area_percent'],
            'estimated_emission_kg_hr': results['estimated_emission_kg_hr'],
            'co2e_per_year_tonnes': results['estimated_emission_kg_hr'] * 24 * 365 * 28 / 1000,  # CH4 GWP=28
            'inference_time_ms': results['inference_time_ms'],
            'severity': self._get_severity(results['estimated_emission_kg_hr'])
        }
        
        return alert
    
    def _get_severity(self, emission_rate):
        """Determine alert severity based on emission rate"""
        if emission_rate < 50:
            return 'LOW'
        elif emission_rate < 200:
            return 'MEDIUM'
        elif emission_rate < 500:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def visualize(self, results, save_path=None):
        """Visualize detection results"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(results['image'])
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Prediction mask
        axes[1].imshow(results['pred_mask'], cmap='hot')
        axes[1].set_title(f'Prediction (Confidence: {results["confidence"]:.3f})')
        axes[1].axis('off')
        
        # Overlay
        overlay = results['image'].copy()
        if results['plume_detected']:
            # Create colored mask overlay
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = results['binary_mask']  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Detection Overlay\nArea: {results["plume_area_percent"]:.2f}%')
        axes[2].axis('off')
        
        # Add info text
        info_text = f"Plume Detected: {results['plume_detected']}\n"
        info_text += f"Inference Time: {results['inference_time_ms']:.1f} ms\n"
        info_text += f"Estimated Emission: {results['estimated_emission_kg_hr']:.1f} kg/hr"
        
        plt.suptitle(info_text, fontsize=12, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        return fig


def process_single_image(detector, image_path, output_dir='results/detections'):
    """Process a single image"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Detect
    results = detector.detect(image_path)
    
    # Print results
    print(f"\nResults:")
    print(f"  Plume detected: {results['plume_detected']}")
    print(f"  Confidence: {results['confidence']:.4f}")
    print(f"  Plume area: {results['plume_area_percent']:.2f}%")
    print(f"  Estimated emission: {results['estimated_emission_kg_hr']:.1f} kg/hr")
    print(f"  Inference time: {results['inference_time_ms']:.1f} ms")
    
    # Generate alert
    alert = detector.generate_alert(results, image_path)
    
    if alert:
        print(f"\n⚠️  ALERT GENERATED:")
        print(f"  Severity: {alert['severity']}")
        print(f"  Message: {alert['message']}")
        print(f"  CO2e impact: {alert['co2e_per_year_tonnes']:.1f} tonnes/year")
        
        # Save alert
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        alert_file = Path(output_dir) / f"alert_{int(time.time())}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        print(f"  Alert saved to: {alert_file}")
    
    # Visualize
    output_path = Path(output_dir) / f"{Path(image_path).stem}_detection.png"
    detector.visualize(results, save_path=output_path)
    
    return results, alert


def process_batch(detector, image_dir, output_dir='results/batch_detections'):
    """Process multiple images"""
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    print(f"\nProcessing {len(image_files)} images from {image_dir}")
    print(f"{'='*60}\n")
    
    all_results = []
    alerts = []
    
    for img_file in image_files:
        try:
            results = detector.detect(img_file)
            alert = detector.generate_alert(results, img_file)
            
            all_results.append(results)
            if alert:
                alerts.append(alert)
            
            print(f"✓ {img_file.name}: {'DETECTED' if results['plume_detected'] else 'NO PLUME'} "
                  f"({results['inference_time_ms']:.1f}ms)")
        
        except Exception as e:
            print(f"✗ {img_file.name}: ERROR - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(image_files)}")
    print(f"Plumes detected: {sum(r['plume_detected'] for r in all_results)}")
    print(f"Alerts generated: {len(alerts)}")
    print(f"Average inference time: {np.mean([r['inference_time_ms'] for r in all_results]):.1f} ms")
    print(f"Total estimated emissions: {sum(r['estimated_emission_kg_hr'] for r in all_results):.1f} kg/hr")
    
    # Save summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': len(image_files),
        'plumes_detected': sum(r['plume_detected'] for r in all_results),
        'alerts_generated': len(alerts),
        'avg_inference_ms': float(np.mean([r['inference_time_ms'] for r in all_results])),
        'total_emission_kg_hr': float(sum(r['estimated_emission_kg_hr'] for r in all_results)),
        'alerts': alerts
    }
    
    with open(Path(output_dir) / 'batch_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {output_dir}/batch_summary.json")
    
    return all_results, alerts


def benchmark_speed(detector, num_runs=100):
    """Benchmark inference speed"""
    
    print(f"\n{'='*60}")
    print(f"SPEED BENCHMARK ({num_runs} runs)")
    print(f"{'='*60}\n")
    
    # Create random test image
    test_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    
    times = []
    
    print("Running benchmark...")
    for i in range(num_runs):
        results = detector.detect(test_image)
        times.append(results['inference_time_ms'])
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs}")
    
    times = np.array(times)
    
    print(f"\nResults:")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {times.min():.2f} ms")
    print(f"  Max: {times.max():.2f} ms")
    print(f"  Std: {times.std():.2f} ms")
    print(f"  FPS: {1000/times.mean():.1f}")
    
    # Check if meets requirement
    if times.mean() < 100:
        print(f"\n✓ REQUIREMENT MET: <100ms average inference time")
    else:
        print(f"\n✗ WARNING: Average inference time exceeds 100ms target")
    
    return times


def main():
    parser = argparse.ArgumentParser(description='Methane Plume Detection - Inference')
    parser.add_argument('--model', type=str, default='models/optimized_best.pth',
                       help='Path to model weights')
    parser.add_argument('--model-type', type=str, default='optimized',
                       choices=['baseline', 'optimized'],
                       help='Model architecture type')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0-1)')
    parser.add_argument('--output', type=str, default='results/detections',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MethaneDetector(
        model_path=args.model,
        model_type=args.model_type,
        device='cpu',  # AMD 5600U - use CPU
        threshold=args.threshold
    )
    
    # Run requested operation
    if args.benchmark:
        benchmark_speed(detector)
    
    elif args.image:
        process_single_image(detector, args.image, args.output)
    
    elif args.batch:
        process_batch(detector, args.batch, args.output)
    
    else:
        print("\nNo operation specified. Use --image, --batch, or --benchmark")
        print("Example: python inference.py --image test.png")
        print("         python inference.py --benchmark")


if __name__ == "__main__":
    main()
