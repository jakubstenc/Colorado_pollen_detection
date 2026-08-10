import glob
import os
STAGED_AREA_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated")
img_files = glob.glob(os.path.join(STAGED_AREA_DIR, "*", "Discarded", "Images", "*.jpg"))
print(f"Total Discarded: {len(img_files)}")
