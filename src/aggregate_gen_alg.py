import os
import glob
import pandas as pd
import re
from datetime import datetime
import numpy as np

# Parse Excel first so we can map indices
excel_path = 'results/Produkce tabulka Ranunculus, Caltha, Gentiana_VK.xlsx'
df_excel = pd.read_excel(excel_path)
df_excel['Cleaned_Code'] = df_excel['kod kytky + datum'].astype(str).str.strip()

# Build the sequential map for Ran_ado specifically
ran_sequence = df_excel[df_excel['Cleaned_Code'].str.contains('Ran_Ado', case=False, na=False)]['Cleaned_Code'].tolist()

counts = []
for species in ['Gen_alg', 'Ran_ado', 'Cal_chi', 'Sed_lan']:
    results_dir = f'results/{species}'
    csv_files = glob.glob(os.path.join(results_dir, '*_details.csv'))

    for file in csv_files:
        basename = os.path.basename(file)
        
        # Fast row count
        try:
            with open(file, 'r') as f:
                num_detections = sum(1 for _ in f) - 1
        except Exception:
            continue
            
        code = None
        if species in ['Gen_alg', 'Cal_chi']:
            # Gen_Alg_22_7_2 or Cal_Chi_17_7_4
            match = re.search(rf'({species}_\d+_\d+_\d+)', basename, re.IGNORECASE)
            if match:
                code = match.group(1).lower()
        elif species in ['Ran_ado', 'Sed_lan']:
            match = re.search(rf'{species}_2025_(\d+)', basename)
            if match:
                idx = int(match.group(1)) - 1
                if species == 'Ran_ado':
                    if 0 <= idx < len(ran_sequence):
                        code = ran_sequence[idx]
                elif species == 'Sed_lan':
                    # Sed_lan is purely unmapped metadata placeholder
                    code = f"Sed_lan_unmapped_{idx+1}"
                    
        if code is not None:
            counts.append({
                'Join_Code': code,
                'Detections': max(0, num_detections), 
                'Original_File': basename,
                'Species': species
            })

counts_df = pd.DataFrame(counts)

# Now prepare Excel for join
def generate_join_code(row):
    if pd.isna(row['kod kytky + datum']): return ''
    val = str(row['kod kytky + datum']).strip()
    if 'Gen_Alg' in val or 'Cal_Chi' in val:
        return val.lower().replace('/', '_')
    return val # Exact match for Ran_ado

df_excel['Join_Code'] = df_excel.apply(generate_join_code, axis=1)

# Merge
# IMPORTANT: Use leftover left join so Sed_lan doesn't disappear without Excel data
merged_df = pd.merge(counts_df, df_excel, on='Join_Code', how='left')

# Extract proper sortable date
def parse_date(row):
    val = str(row['Join_Code'])
    try:
        if row['Species'] in ['Gen_alg', 'Cal_chi']:
            # Either gen_alg_day_mo_rep or cal_chi_day_mo_rep
            parts = val.split('_')
            return datetime(year=2025, month=int(parts[-2]), day=int(parts[-3]))
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

# Sort logically by date (don't blindly dropna so we retain Sed_lan!)
merged_df.sort_values(['Collection_Date', 'Species'], inplace=True, na_position='last')

# Load focus_report.csv if it exists and merge Focus_Status
if os.path.exists('focus_report.csv'):
    focus_df = pd.read_csv('focus_report.csv')
    # focus_df['file'] is something like 20260701_002_Dep_Ran_Ado...czi
    # merged_df['Original_File'] is 20260701_002_Dep_Ran_Ado..._details.csv
    # Create a mapping key by stripping extensions
    focus_df['Base_CZI'] = focus_df['file'].str.replace('.czi', '', regex=False)
    merged_df['Base_CZI'] = merged_df['Original_File'].astype(str).str.replace('_details.csv', '', regex=False)
    
    # Merge the Status
    merged_df = pd.merge(merged_df, focus_df[['Base_CZI', 'status']], on='Base_CZI', how='left')
    merged_df.rename(columns={'status': 'Focus_Status'}, inplace=True)
    merged_df.drop(columns=['Base_CZI'], inplace=True)
else:
    merged_df['Focus_Status'] = 'Unknown'

# Output
out_path = 'results/Gen_alg_summary.csv'
merged_df.to_csv(out_path, index=False)
print(f"✅ Summary saved to {out_path} with {len(merged_df)} natively left-joined records.")

