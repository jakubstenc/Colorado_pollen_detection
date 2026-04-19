import os
import glob
import pandas as pd
import re
from datetime import datetime

# Parse Excel first so we can map indices
excel_path = 'results/Produkce tabulka Ranunculus, Caltha, Gentiana_VK.xlsx'
df_excel = pd.read_excel(excel_path)
df_excel['Cleaned_Code'] = df_excel['kod kytky + datum'].astype(str).str.strip()

# Build the sequential map for Ran_ado specifically
ran_sequence = df_excel[df_excel['Cleaned_Code'].str.contains('Ran_Ado', case=False, na=False)]['Cleaned_Code'].tolist()

counts = []
for species in ['Gen_alg', 'Ran_ado']:
    results_dir = f'results/{species}'
    csv_files = glob.glob(os.path.join(results_dir, '*_details.csv'))

    for file in csv_files:
        basename = os.path.basename(file)
        
        # Fast row count
        try:
            with open(file, 'r') as f:
                num_detections = sum(1 for _ in f) - 1
        except Exception as e:
            continue
            
        code = None
        if species == 'Gen_alg':
            match = re.search(r'(Gen_Alg_\d+_\d+_\d+)', basename, re.IGNORECASE)
            if match:
                code = match.group(1).lower()
        elif species == 'Ran_ado':
            match = re.search(r'Ran_ado_2025_(\d+)', basename)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(ran_sequence):
                    # We map it to the raw Excel code directly!
                    code = ran_sequence[idx]
                    
        if code is not None:
            counts.append({
                'Join_Code': code if species == 'Ran_ado' else code, 
                'Detections': max(0, num_detections), 
                'Original_File': basename,
                'Species': species
            })

counts_df = pd.DataFrame(counts)

# Now prepare Excel for join
# For Gen_alg we need normalized lower unslashed codes, for Ran_ado we can use exact match
def generate_join_code(row):
    val = str(row['kod kytky + datum']).strip()
    if 'Gen_Alg' in val:
        return val.lower().replace('/', '_')
    return val # Exact match for Ran_ado

df_excel['Join_Code'] = df_excel.apply(generate_join_code, axis=1)

# Merge
merged_df = pd.merge(counts_df, df_excel, on='Join_Code', how='inner')

# Extract proper sortable date
def parse_date(row):
    val = str(row['Join_Code'])
    try:
        if row['Species'] == 'Gen_alg':
            parts = val.split('_')
            return datetime(year=2025, month=int(parts[3]), day=int(parts[2]))
        elif row['Species'] == 'Ran_ado':
            # Ran_Ado_22/6_1
            # remove prefix Ran_Ado_
            dt_segment = val.split('_')[-2] # '22/6'
            d, m = dt_segment.split('/')
            return datetime(year=2025, month=int(m), day=int(d))
    except:
        return None
    return None

merged_df['Collection_Date'] = merged_df.apply(parse_date, axis=1)

# Sort logically by date
merged_df = merged_df.dropna(subset=['Collection_Date'])
merged_df.sort_values(['Collection_Date', 'Species'], inplace=True)

# Output
out_path = 'results/Gen_alg_summary.csv'
merged_df.to_csv(out_path, index=False)
print(f"✅ Summary saved to {out_path} with {len(merged_df)} perfectly matched multi-species records.")
