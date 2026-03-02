import json
import pandas as pd
import numpy as np
from pathlib import Path

def edit_files_in_place():
    base_dir_pedestrians = Path(__file__).resolve().parent / "data" / "pedestrians" / "default"
    base_dir_weather = Path(__file__).resolve().parent / "data" / "weather" / "default"
    splits = ["train", "val", "test"]
    folders = {"train": "training", "val": "validation", "test": "testing"}

    for split in splits:
        folder = folders[split]
        ped_path = base_dir_pedestrians / folder / f"pedestrians_{split}.json"
        wea_path = base_dir_weather / folder / f"weather_{split}.json"

        if not ped_path.exists() or not wea_path.exists():
            continue

        print(f"--- Interpolating & Formatting {split} files ---")

        # LOAD
        with open(ped_path, 'r') as f:
            ped_raw = json.load(f)
            ped_df = pd.DataFrame(ped_raw.get('ecoCounterSiteData', []))
        with open(wea_path, 'r') as f:
            wea_raw = json.load(f)
            wea_df = pd.DataFrame(wea_raw.get('rows', []))

        # STANDARDIZE
        ped_df['date'] = pd.to_datetime(ped_df['date'])
        if 'ts' in wea_df.columns:
            wea_df = wea_df.rename(columns={'ts': 'date'})
        wea_df['date'] = pd.to_datetime(wea_df['date'])

        ped_df['count'] = ped_df['counts'].apply(lambda x: sum(x) if isinstance(x, list) else x)
        ped_df = ped_df.set_index('date').sort_index()
        wea_df = wea_df.set_index('date').sort_index()

        # FILL MOCK VALUES (Interpolation)
        # Fix weather first
        wea_df['temp_c'] = wea_df['temp_c'].interpolate(method='time').ffill().bfill().round(1)

        # Fix Pedestrians: Identify outages (>24h zeros) and set to NaN
        is_zero = ped_df['count'] == 0
        zero_runs = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).transform('sum')
        ped_df.loc[(is_zero) & (zero_runs > 24), 'count'] = np.nan
        
        # Fill pedestrian NaNs with mock values based on surrounding hours
        # We limit interpolation to 6 hours to keep it realistic
        ped_df['count'] = ped_df['count'].interpolate(method='time', limit=6).fillna(0)

        # SYNC (Inner join to ensure exact same timestamps)
        combined = ped_df[['count']].join(wea_df, how='inner')

        # STRING FORMATTING
        formatted_dates = combined.index.strftime('%Y-%m-%dT%H:%M:%SZ')

        # OVERWRITE
        final_ped_rows = []
        for i in range(len(combined)):
            final_ped_rows.append({
                "date": formatted_dates[i],
                "counts": int(round(combined['count'].iloc[i]))
            })
        
        final_wea_rows = []
        wea_cols = [c for c in combined.columns if c != 'count']
        for i in range(len(combined)):
            row = {"ts": formatted_dates[i]}
            for col in wea_cols:
                row[col] = combined[col].iloc[i]
            final_wea_rows.append(row)

        with open(ped_path, 'w') as f:
            json.dump({"ecoCounterSiteData": final_ped_rows}, f, indent=2)
        with open(wea_path, 'w') as f:
            json.dump({"location": wea_raw.get("location", {"place": "Oulu"}), "rows": final_wea_rows}, f, indent=2)

        print(f"✅ Updated {split}: {len(combined)} rows with interpolated values.")

if __name__ == "__main__":
    edit_files_in_place()