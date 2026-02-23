import json
from pathlib import Path

def sync_pair(folder, weather_file, ped_file):
    """
    Synchronizes a pair of weather and pedestrian JSON files by aligning 
    pedestrian counts to the exact timestamps found in the weather data.
    
    This function addresses record mismatches caused by weather station 
    sensor dropouts. It ensures that both files have an identical number 
    of rows and matching start/end timestamps by using the weather file 
    as the master template. Any pedestrian records not present in the 
    weather timeline are discarded, and missing hours are filled with 0.
    """
    base_dir = Path(__file__).parent / "data" / folder
    w_path = base_dir / weather_file
    p_path = base_dir / ped_file
    
    if not w_path.exists() or not p_path.exists():
        print(f"Skipping {folder}: Files not found at {base_dir}")
        return

    # Load Weather (Master List)
    with open(w_path, 'r', encoding='utf-8') as f:
        w_data = json.load(f)
    
    # Extract master timestamps from 'rows' key used in weather_dataset_builder.py
    master_timestamps = [row['ts'] for row in w_data['rows']]

    # Load Pedestrians
    with open(p_path, 'r', encoding='utf-8') as f:
        p_data = json.load(f)
    
    # Create lookup from 'ecoCounterSiteData' key used in pedestrianDataFetcher.py
    ped_lookup = {row['date']: row['counts'] for row in p_data['ecoCounterSiteData']}

    # Align: Force pedestrian data to match the master weather timeline
    synced_ped_rows = []
    for ts in master_timestamps:
        # Defaults to 0 if hour is missing in pedestrian data
        count = ped_lookup.get(ts, 0) 
        synced_ped_rows.append({"date": ts, "counts": count})

    # Overwrite the pedestrian file with the synchronized version
    with open(p_path, 'w', encoding='utf-8') as f:
        json.dump({"ecoCounterSiteData": synced_ped_rows}, f, indent=2)

    print(f"✅ {folder}: Synced to {len(synced_ped_rows)} records.")
    print(f"   Start: {master_timestamps[0]} | End: {master_timestamps[-1]}")

if __name__ == "__main__":
    dataset_configs = [
        ("training", "weather_train.json", "pedestrians_train.json"),
        ("validation", "weather_val.json", "pedestrians_val.json"),
        ("testing", "weather_test.json", "pedestrians_test.json")
    ]

    print("--- Starting Dataset Synchronization ---")
    for folder, w_file, p_file in dataset_configs:
        sync_pair(folder, w_file, p_file)
    print("--- Sync Complete. Datasets are now perfectly aligned. ---")