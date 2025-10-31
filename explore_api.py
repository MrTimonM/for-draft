"""
Explore Carbon Mapper API to understand the data structure
"""
import requests
import json

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzYyMTgxNzE2LCJpYXQiOjE3NjE1NzY5MTYsImp0aSI6IjcwM2VhMDdkNGRmMjQ5YmJiYjk5MWI3MTNlODVlNzk4Iiwic2NvcGUiOiJzdGFjIGNhdGFsb2c6cmVhZCIsImdyb3VwcyI6IlB1YmxpYyIsImFsbF9ncm91cF9uYW1lcyI6eyJjb21tb24iOlsiUHVibGljIl19LCJvcmdhbml6YXRpb25zIjoiIiwic2V0dGluZ3MiOnt9LCJpc19zdGFmZiI6ZmFsc2UsImlzX3N1cGVydXNlciI6ZmFsc2UsInVzZXJfaWQiOjE4MjUwfQ.vSs3OCZUDrB9wXMoMXlu9XMx-a9LC4ClXtOkVPd5VM4"

# Carbon Mapper STAC API endpoint
BASE_URL = "https://api.carbonmapper.org/api/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def explore_api():
    """Explore the Carbon Mapper API structure"""
    
    # Try to get plume data directly
    print("=" * 80)
    print("Fetching CH4 plumes...")
    print("=" * 80)
    
    # Try direct plume endpoint
    plume_url = f"{BASE_URL}/plume"
    params = {
        "limit": 5,
        "gas": "CH4"
    }
    
    response = requests.get(plume_url, headers=headers, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(json.dumps(results, indent=2))
    else:
        print(f"Error: {response.text[:500]}")
    
    # Try catalog endpoint
    print("\n" + "=" * 80)
    print("Trying catalog endpoint...")
    print("=" * 80)
    
    catalog_url = f"{BASE_URL}/catalog/search"
    search_params = {
        "limit": 5
    }
    
    response = requests.post(catalog_url, headers=headers, json=search_params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nTotal features: {results.get('context', {}).get('matched', 'unknown')}")
        print(f"Returned: {len(results.get('features', []))}")
        
        if results.get('features'):
            # Show first plume in detail
            first_plume = results['features'][0]
            print("\n" + "-" * 80)
            print("FIRST PLUME EXAMPLE:")
            print("-" * 80)
            print(json.dumps(first_plume, indent=2))
            
            # Show available assets
            print("\n" + "-" * 80)
            print("AVAILABLE ASSETS:")
            print("-" * 80)
            if 'assets' in first_plume:
                for asset_name, asset_info in first_plume['assets'].items():
                    print(f"\n{asset_name}:")
                    print(f"  Type: {asset_info.get('type')}")
                    print(f"  Title: {asset_info.get('title')}")
                    print(f"  URL: {asset_info.get('href')}")
    else:
        print(f"Error: {response.text[:500]}")

if __name__ == "__main__":
    explore_api()
