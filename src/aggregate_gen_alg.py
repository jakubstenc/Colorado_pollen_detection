import os
import glob
import pandas as pd
import re
from datetime import datetime

# 1. Parse CSVs
results_dir = 'results/Gen_alg'
csv_files = glob.glob(os.path.join(results_dir, '*_details.csv'))

counts = []
for file in csv_files:
    basename = os.path.basename(file)
    # Extract code like Gen_Alg_27_7_1
    match = re.search(r'(Gen_Alg_\d+_\d+_\d+)', basename, re.IGNORECASE)
    if match:
        code = match.group(1).lower()
        # Fast row count
        try:
            with open(file, 'r') as f:
                num_detections = sum(1 for _ in f) - 1
            counts.append({'Normalized_Code': code, 'Detections': max(0, num_detections), 'Original_File': basename})
        except Exception as e:
            print(f"Error reading {file}: {e}")

counts_df = pd.DataFrame(counts)
print(f"Found {len(counts_df)} valid CSV result files.")

# 2. Parse Excel
excel_path = 'results/Produkce tabulka Ranunculus, Caltha, Gentiana_VK.xlsx'
df_excel = pd.read_excel(excel_path)
# Clean excel code: replace slash with underscore, strip whitespace, make lowercase
df_excel['Normalized_Code'] = df_excel['kod kytky + datum'].astype(str).str.replace('/', '_').str.lower().str.strip()

# 3. Merge
merged_df = pd.merge(counts_df, df_excel, on='Normalized_Code', how='inner')

unmapped_csv = set(counts_df['Normalized_Code']) - set(df_excel['Normalized_Code'])
if unmapped_csv:
    print(f"Warning: {len(unmapped_csv)} CSV files did not match any code in the Excel table.")
else:
    print(f"Success: All CSV files correctly matched an Excel row!")

# 4. Extract proper sortable date
def parse_date(norm_code):
    try:
        parts = norm_code.split('_')
        day = int(parts[2])
        month = int(parts[3])
        return datetime(year=2025, month=month, day=day)
    except:
        return None

merged_df['Collection_Date'] = merged_df['Normalized_Code'].apply(parse_date)

# Sort logically by date
merged_df.sort_values('Collection_Date', inplace=True)

# 5. Output
out_path = 'results/Gen_alg_summary.csv'
merged_df.to_csv(out_path, index=False)
print(f"✅ Summary saved to {out_path} with {len(merged_df)} perfectly matched records.")
