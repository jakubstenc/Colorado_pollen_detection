import pandas as pd
import glob
import os

master_tables = glob.glob('/home/meow/Documents/Antigravity/Pollen_latitude/master_tables/*.csv') + \
                glob.glob('/home/meow/Documents/Antigravity/Pollen_latitude/master_tables/*.xlsx')

mapping = {}

for f in master_tables:
    try:
        if f.endswith('.csv'):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)
            
        # Try to find columns like Sample_code, Season
        cols = [c.lower() for c in df.columns]
        
        sample_code_col = None
        locality_col = None
        
        for c in df.columns:
            if 'sample_code' in c.lower() or 'sample code' in c.lower():
                sample_code_col = c
            if 'season' in c.lower() or 'locality' in c.lower():
                locality_col = c
                
        if sample_code_col and locality_col:
            for _, row in df.iterrows():
                code = str(row[sample_code_col]).strip()
                loc = str(row[locality_col]).strip()
                if code != 'nan' and loc != 'nan':
                    mapping[code] = loc
    except Exception as e:
        print(f"Failed to parse {f}: {e}")

print(f"Found {len(mapping)} mappings.")
import itertools
print("Sample mappings:", dict(itertools.islice(mapping.items(), 20)))

