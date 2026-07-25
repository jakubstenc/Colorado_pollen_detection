import os
import glob
from collections import defaultdict

lbl_dir = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Curated_Retrain_Data/labels")
files = glob.glob(os.path.join(lbl_dir, "*.txt"))

summary = defaultdict(int)
total_positive_files = 0
total_negative_files = 0

for f in files:
    if os.path.getsize(f) > 0:
        total_positive_files += 1
        basename = os.path.basename(f)
        # Extract species logic:
        # Example: Cal_chi_20260224_031_Cal_Chi_29_6_1... -> "Cal_chi"
        # Example: Deposition_Stigmas_20260701_054_Dep_Ran_Ado_... -> "Ran_ado"
        if "Deposition_Stigmas" in basename:
            # Usually Deposition_Stigmas_YYYYMMDD_XXX_Dep_Species_Name...
            parts = basename.split("_")
            try:
                # Find "Dep" index
                dep_idx = parts.index("Dep")
                species = f"{parts[dep_idx+1]}_{parts[dep_idx+2]}"
            except ValueError:
                species = "Unknown"
        else:
            species = "_".join(basename.split("_")[:2])
            
        summary[species] += 1
    else:
        total_negative_files += 1

print(f"Total Hard Negatives (Skipped): {total_negative_files}")
print(f"Total Positive Images (Exported): {total_positive_files}")
print("\n--- Breakdown by Species ---")
for sp, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
    print(f"- {sp}: {count} images")
