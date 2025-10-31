"""
Download complete CH4 plume dataset from Carbon Mapper API
"""
import requests
import json
import os
from pathlib import Path
import time

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzYyMTgxNzE2LCJpYXQiOjE3NjE1NzY5MTYsImp0aSI6IjcwM2VhMDdkNGRmMjQ5YmJiYjk5MWI3MTNlODVlNzk4Iiwic2NvcGUiOiJzdGFjIGNhdGFsb2c6cmVhZCIsImdyb3VwcyI6IlB1YmxpYyIsImFsbF9ncm91cF9uYW1lcyI6eyJjb21tb24iOlsiUHVibGljIl19LCJvcmdhbml6YXRpb25zIjoiIiwic2V0dGluZ3MiOnt9LCJpc19zdGFmZiI6ZmFsc2UsImlzX3N1cGVydXNlciI6ZmFsc2UsInVzZXJfaWQiOjE4MjUwfQ.vSs3OCZUDrB9wXMoMXlu9XMx-a9LC4ClXtOkVPd5VM4"
BASE_URL = "https://api.carbonmapper.org/api/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Create directories
OUTPUT_DIR = Path("ch4_dataset")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
METADATA_DIR = OUTPUT_DIR / "metadata"

for dir_path in [OUTPUT_DIR, IMAGES_DIR, LABELS_DIR, METADATA_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


def get_all_ch4_plumes():
    """
    Fetch all CH4 plume detections using pagination
    API limits: max 1000 per request, use offset for pagination
    Using the public 'annotated' endpoint which doesn't require special scopes
    Saves incrementally to avoid data loss
    """
    all_plumes = []
    offset = 0
    limit = 1000  # Maximum allowed
    batch_file = OUTPUT_DIR / "plumes_batch_{}.json"
    batch_num = 0
    
    # Check for existing batches to resume
    existing_batches = list(OUTPUT_DIR.glob("plumes_batch_*.json"))
    if existing_batches:
        print(f"Found {len(existing_batches)} existing batch files. Loading...")
        for batch_path in sorted(existing_batches):
            with open(batch_path, 'r') as f:
                batch_data = json.load(f)
                all_plumes.extend(batch_data)
        batch_num = len(existing_batches)
        offset = len(all_plumes)
        print(f"Resuming from offset {offset} with {len(all_plumes)} plumes already fetched")
    
    print("Fetching CH4 plume metadata...")
    
    while True:
        params = {
            "plume_gas": "CH4",
            "limit": limit,
            "offset": offset,
            "sort": "published_desc"
        }
        
        try:
            response = requests.get(
                f"{BASE_URL}/catalog/plumes/annotated",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text[:500])
                break
            
            data = response.json()
            items = data.get('items', [])
            total_count = data.get('total_count', 0)
            
            if not items:
                break
            
            # Save this batch immediately
            batch_path = OUTPUT_DIR / f"plumes_batch_{batch_num}.json"
            with open(batch_path, 'w') as f:
                json.dump(items, f, indent=2)
            
            all_plumes.extend(items)
            print(f"Fetched {len(all_plumes)}/{total_count} plumes... (saved batch {batch_num})")
            
            batch_num += 1
            
            # Check if we've retrieved all plumes
            if len(all_plumes) >= total_count:
                break
            
            offset += limit
            time.sleep(0.3)  # Rate limiting
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved!")
            print(f"Resume will continue from offset {offset}")
            break
        except Exception as e:
            print(f"Error fetching plumes: {e}")
            print(f"Progress saved. You can resume from offset {offset}")
            time.sleep(2)
            continue
    
    print(f"\nTotal CH4 plumes retrieved: {len(all_plumes)}")
    return all_plumes


def download_asset(asset_url, output_path):
    """Download a single asset (image or label)"""
    try:
        response = requests.get(asset_url, headers=headers, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"  Failed to download: {response.status_code}")
            return False
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def download_plume_data(plume, index):
    """
    Download satellite image and plume mask for a single plume
    """
    plume_id = plume.get('id')
    plume_name = plume.get('name', f"plume_{index}")
    
    # Get detailed plume information including assets
    try:
        detail_response = requests.get(
            f"{BASE_URL}/catalog/plume/{plume_id}",
            headers=headers
        )
        
        if detail_response.status_code != 200:
            print(f"  Failed to get details for {plume_name}")
            return False
        
        plume_detail = detail_response.json()
        assets = plume_detail.get('assets', {})
        
        # Common asset types in Carbon Mapper API:
        # - rgb: RGB image
        # - plume: Plume mask/geometry
        # - ch4: Methane concentration
        # - cog: Cloud Optimized GeoTIFF
        
        downloaded_assets = {}
        
        # Download RGB image
        if 'rgb' in assets:
            img_path = IMAGES_DIR / f"{plume_name}_rgb.tif"
            if download_asset(assets['rgb']['href'], img_path):
                downloaded_assets['rgb'] = str(img_path)
        
        # Download plume mask
        if 'plume' in assets:
            mask_path = LABELS_DIR / f"{plume_name}_mask.tif"
            if download_asset(assets['plume']['href'], mask_path):
                downloaded_assets['plume'] = str(mask_path)
        
        # Download CH4 concentration if available
        if 'ch4' in assets:
            ch4_path = IMAGES_DIR / f"{plume_name}_ch4.tif"
            if download_asset(assets['ch4']['href'], ch4_path):
                downloaded_assets['ch4'] = str(ch4_path)
        
        # Save metadata
        metadata = {
            'plume_id': plume_id,
            'plume_name': plume_name,
            'gas': plume.get('gas'),
            'emission_auto': plume.get('emission_auto'),
            'datetime': plume.get('datetime'),
            'instrument': plume.get('instrument'),
            'quality': plume.get('quality'),
            'sector': plume.get('sector'),
            'geometry': plume.get('geometry'),
            'assets': assets,
            'downloaded_files': downloaded_assets
        }
        
        metadata_path = METADATA_DIR / f"{plume_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return len(downloaded_assets) > 0
        
    except Exception as e:
        print(f"  Error processing {plume_name}: {e}")
        return False


def main():
    print("=" * 80)
    print("Carbon Mapper CH4 Dataset Downloader")
    print("=" * 80)
    
    # Step 1: Get all CH4 plume metadata (saves incrementally)
    plumes = get_all_ch4_plumes()
    
    if not plumes:
        print("No plumes found!")
        return
    
    # Save complete metadata list
    print("\nConsolidating all batches into single metadata file...")
    with open(OUTPUT_DIR / "all_plumes_metadata.json", 'w') as f:
        json.dump(plumes, f, indent=2)
    
    print(f"Saved complete metadata to {OUTPUT_DIR / 'all_plumes_metadata.json'}")
    
    # Ask user if they want to download images now
    print("\n" + "=" * 80)
    print(f"Metadata collection complete: {len(plumes)} plumes")
    print("=" * 80)
    print("\nNote: Downloading images for all plumes may take significant time and storage.")
    print(f"Estimated downloads: {len(plumes) * 2} files (image + mask per plume)")
    
    response = input("\nProceed with image downloads? (yes/no): ").lower().strip()
    
    if response not in ['yes', 'y']:
        print("\nSkipping image download. You can run the download later.")
        print("Metadata has been saved successfully.")
        return
    
    # Step 2: Download images and labels for each plume
    print("\n" + "=" * 80)
    print("Downloading satellite images and plume masks...")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for i, plume in enumerate(plumes):
        plume_name = plume.get('name', f"plume_{i}")
        print(f"\n[{i+1}/{len(plumes)}] Processing: {plume_name}")
        
        if download_plume_data(plume, i):
            successful += 1
        else:
            failed += 1
        
        # Rate limiting
        time.sleep(0.2)
        
        # Save progress every 100 plumes
        if (i + 1) % 100 == 0:
            progress = {
                'processed': i + 1,
                'successful': successful,
                'failed': failed,
                'timestamp': time.time()
            }
            with open(OUTPUT_DIR / 'download_progress.json', 'w') as f:
                json.dump(progress, f, indent=2)
    
    # Step 3: Create summary
    print("\n" + "=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"Total plumes: {len(plumes)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed: {failed}")
    print(f"\nDataset saved to: {OUTPUT_DIR.absolute()}")
    print(f"  - Images: {IMAGES_DIR}")
    print(f"  - Labels: {LABELS_DIR}")
    print(f"  - Metadata: {METADATA_DIR}")


if __name__ == "__main__":
    main()
