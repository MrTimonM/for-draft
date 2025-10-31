# 🚀 Complete Contest Submission Script
# Run everything needed for the contest
# Windows PowerShell Script

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  METHANE PLUME DETECTOR - COMPLETE SETUP" -ForegroundColor Cyan
Write-Host "  Contest Ready in 15 Minutes!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/7] Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python OK" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "[2/7] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take 3-5 minutes..." -ForegroundColor Gray
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Some packages failed, continuing anyway..." -ForegroundColor Yellow
}
Write-Host ""

# Prepare dataset
Write-Host "[3/7] Creating synthetic dataset (100 samples)..." -ForegroundColor Yellow
python prepare_dataset.py --synthetic --num-samples 100
Write-Host "✓ Dataset ready" -ForegroundColor Green
Write-Host ""

# Train model
Write-Host "[4/7] Training models (this will take 10-15 minutes)..." -ForegroundColor Yellow
Write-Host "Training both baseline and optimized for comparison..." -ForegroundColor Gray
Write-Host "Grab a coffee! ☕" -ForegroundColor Magenta
python train.py --both --epochs 10 --batch-size 4
Write-Host "✓ Training complete" -ForegroundColor Green
Write-Host ""

# Run benchmark
Write-Host "[5/7] Running speed benchmark..." -ForegroundColor Yellow
python inference.py --benchmark --model models/optimized_best.pth
Write-Host "✓ Benchmark complete" -ForegroundColor Green
Write-Host ""

# Generate visualizations
Write-Host "[6/7] Generating visualizations and reports..." -ForegroundColor Yellow
python visualize_results.py
Write-Host "✓ Visualizations created" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "[7/7] Setup complete! Here's what you have:" -ForegroundColor Yellow
Write-Host ""
Write-Host "✓ Trained models:" -ForegroundColor Green
Write-Host "  - models/baseline_best.pth" -ForegroundColor Gray
Write-Host "  - models/optimized_best.pth" -ForegroundColor Gray
Write-Host ""
Write-Host "✓ Documentation:" -ForegroundColor Green
Write-Host "  - README.md" -ForegroundColor Gray
Write-Host "  - FOOTPRINT.md" -ForegroundColor Gray
Write-Host "  - model_card.md" -ForegroundColor Gray
Write-Host "  - data_card.md" -ForegroundColor Gray
Write-Host ""
Write-Host "✓ Results:" -ForegroundColor Green
Write-Host "  - results/emissions_*.csv (Energy tracking)" -ForegroundColor Gray
Write-Host "  - results/*.png (Visualizations)" -ForegroundColor Gray
Write-Host "  - results/SUMMARY.md (Complete summary)" -ForegroundColor Gray
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NEXT STEPS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Launch demo:" -ForegroundColor Yellow
Write-Host "   streamlit run demo_app.py" -ForegroundColor White
Write-Host ""
Write-Host "2. Test detection:" -ForegroundColor Yellow
Write-Host "   python inference.py --image test.png" -ForegroundColor White
Write-Host ""
Write-Host "3. Check results:" -ForegroundColor Yellow
Write-Host "   cat results/SUMMARY.md" -ForegroundColor White
Write-Host ""
Write-Host "4. Read guides:" -ForegroundColor Yellow
Write-Host "   - QUICKSTART.md (Detailed instructions)" -ForegroundColor White
Write-Host "   - DEMO_GUIDE.md (How to present)" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  YOU'RE READY FOR THE CONTEST! 🏆" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
