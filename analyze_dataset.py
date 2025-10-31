"""
Analyze the downloaded CH4 dataset to verify it's suitable for ML training
"""
import json
from pathlib import Path

# Load all plumes data
batch_files = sorted(Path("ch4_dataset").glob("plumes_batch_*.json"))
all_plumes = []

print("=" * 80)
print("CARBON MAPPER CH4 DATASET ANALYSIS")
print("=" * 80)

for batch_file in batch_files:
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
        all_plumes.extend(batch_data)

print(f"\n📊 DATASET OVERVIEW")
print(f"   Total plumes collected: {len(all_plumes)}")
print(f"   Batch files: {len(batch_files)}")

# Analyze what data we have
print(f"\n🔍 DATA FIELDS AVAILABLE:")
if all_plumes:
    sample_plume = all_plumes[0]
    print(f"   Fields per plume: {len(sample_plume.keys())}")
    print(f"\n   Key fields for ML training:")
    
    # Check critical fields
    critical_fields = {
        'plume_id': 'Unique identifier',
        'gas': 'Gas type (CH4/CO2)',
        'geometry_json': 'Plume location coordinates',
        'emission_auto': 'Emission rate (kg/hr)',
        'plume_png': 'Plume visualization PNG',
        'plume_rgb_png': 'RGB image with plume overlay',
        'plume_tif': 'Plume mask GeoTIFF',
        'con_tif': 'Concentration GeoTIFF',
        'rgb_png': 'RGB satellite image',
        'plume_bounds': 'Bounding box coordinates',
        'instrument': 'Sensor used',
        'sector': 'Source sector (e.g., oil/gas)',
        'wind_speed_avg_auto': 'Wind speed',
        'wind_direction_avg_auto': 'Wind direction',
    }
    
    for field, description in critical_fields.items():
        has_field = field in sample_plume
        value = sample_plume.get(field, 'N/A')
        if value is not None and value != 'N/A':
            if isinstance(value, str) and len(str(value)) > 60:
                value = str(value)[:60] + "..."
        status = "✅" if has_field and value not in [None, 'N/A'] else "❌"
        print(f"   {status} {field:30s} - {description}")
        if has_field and value not in [None, 'N/A']:
            print(f"      Example: {value}")

# Check image URLs
print(f"\n🖼️  IMAGE DATA AVAILABLE:")
images_available = {
    'RGB Images': 'rgb_png',
    'Plume Masks (PNG)': 'plume_png', 
    'Plume + RGB (PNG)': 'plume_rgb_png',
    'Plume Masks (TIF)': 'plume_tif',
    'Concentration Maps': 'con_tif',
}

for name, field in images_available.items():
    count = sum(1 for p in all_plumes if p.get(field) not in [None, ''])
    percentage = (count / len(all_plumes) * 100) if all_plumes else 0
    print(f"   {name:25s}: {count:5d} / {len(all_plumes)} ({percentage:.1f}%)")

# Check instruments
print(f"\n🛰️  INSTRUMENTS/SENSORS:")
instruments = {}
for plume in all_plumes:
    inst = plume.get('instrument', 'unknown')
    instruments[inst] = instruments.get(inst, 0) + 1

for inst, count in sorted(instruments.items(), key=lambda x: x[1], reverse=True):
    print(f"   {inst:15s}: {count:5d} plumes")

# Check sectors
print(f"\n🏭 SOURCE SECTORS:")
sectors = {}
for plume in all_plumes:
    sector = plume.get('sector', 'unknown')
    if sector is None:
        sector = 'NULL'
    sectors[sector] = sectors.get(sector, 0) + 1

for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"   {sector:15s}: {count:5d} plumes")

# Emission statistics
print(f"\n💨 EMISSION STATISTICS:")
emissions = [p.get('emission_auto') for p in all_plumes if p.get('emission_auto') is not None]
if emissions:
    print(f"   Total plumes with emission data: {len(emissions)}")
    print(f"   Min emission: {min(emissions):.2f} kg/hr")
    print(f"   Max emission: {max(emissions):.2f} kg/hr")
    print(f"   Mean emission: {sum(emissions)/len(emissions):.2f} kg/hr")
    print(f"   Median emission: {sorted(emissions)[len(emissions)//2]:.2f} kg/hr")

# Check what we can use for training
print(f"\n" + "=" * 80)
print(f"✅ SUITABILITY FOR ML TRAINING")
print(f"=" * 80)

suitable = True
issues = []

if len(all_plumes) < 100:
    issues.append("❌ Dataset too small (< 100 samples)")
    suitable = False
else:
    print(f"✅ Dataset size: {len(all_plumes)} plumes (sufficient for training)")

# Check if we have image URLs
img_count = sum(1 for p in all_plumes if p.get('rgb_png'))
mask_count = sum(1 for p in all_plumes if p.get('plume_tif') or p.get('plume_png'))

if img_count > 0 and mask_count > 0:
    print(f"✅ Image-Label pairs available:")
    print(f"   - RGB Images: {img_count}")
    print(f"   - Plume Masks: {mask_count}")
else:
    issues.append("❌ Missing image-mask pairs")
    suitable = False

# Check geometries
geom_count = sum(1 for p in all_plumes if p.get('geometry_json'))
if geom_count > len(all_plumes) * 0.8:
    print(f"✅ Spatial data: {geom_count} plumes have coordinates")
else:
    print(f"⚠️  Warning: Only {geom_count} plumes have coordinates")

print(f"\n" + "=" * 80)
print(f"WHAT YOU CAN TRAIN:")
print(f"=" * 80)

print(f"""
1. 🎯 PLUME DETECTION MODEL
   - Input: RGB satellite images (from 'rgb_png' URLs)
   - Labels: Plume masks (from 'plume_tif' or 'plume_png' URLs)
   - Task: Semantic segmentation to detect CH4 plumes
   - Samples: {min(img_count, mask_count)} image-mask pairs

2. 📊 EMISSION ESTIMATION MODEL
   - Input: Plume features (size, shape, wind conditions)
   - Labels: Emission rates ('emission_auto' field)
   - Task: Regression to predict emission rates
   - Samples: {len(emissions)} plumes with emission data

3. 🏷️  SOURCE CLASSIFICATION MODEL
   - Input: Plume characteristics + satellite imagery
   - Labels: Source sectors ('sector' field)
   - Task: Multi-class classification
   - Samples: {len(all_plumes)} plumes with sector labels

4. 🗺️  SPATIAL DETECTION MODEL
   - Input: Location, time, environmental conditions
   - Labels: Plume presence/absence + coordinates
   - Task: Object detection or point prediction
   - Samples: {geom_count} plumes with coordinates
""")

print(f"=" * 80)
print(f"NEXT STEPS TO BUILD TRAINING DATASET:")
print(f"=" * 80)
print(f"""
1. ⬇️  Download the actual images:
   - Download RGB images from 'rgb_png' URLs
   - Download plume masks from 'plume_tif' or 'plume_png' URLs
   - Current status: {img_count} URLs available (not yet downloaded)

2. 📁 Organize into train/val/test splits:
   - Recommended: 70% train, 15% validation, 15% test
   - Stratify by sector or instrument for balanced splits

3. 🔄 Preprocessing:
   - Normalize/standardize RGB images
   - Convert masks to binary or multi-class format
   - Handle GeoTIFF format and coordinate systems
   - Resize images to consistent dimensions

4. 📝 Create annotation files:
   - COCO format for object detection
   - Pixel masks for semantic segmentation
   - CSV/JSON for regression tasks
""")

if issues:
    print(f"\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")

print(f"\n" + "=" * 80)
