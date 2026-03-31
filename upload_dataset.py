import os
import shutil
from pathlib import Path

src_dir = Path("dataset_general_v1")
dst_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/staging_area/dataset_general_v1")

if dst_dir.exists():
    print(f"Destination {dst_dir} already exists. Cleaning up...")
    shutil.rmtree(dst_dir, ignore_errors=True)

for root, dirs, files in os.walk(src_dir):
    rel_path = os.path.relpath(root, src_dir)
    target_dir = dst_dir / rel_path
    os.makedirs(target_dir, exist_ok=True)
    for file in files:
        src_file = Path(root) / file
        dst_file = target_dir / file
        shutil.copy(str(src_file), str(dst_file))
        
print("Dataset successfully uploaded to the cloud mount staging area!")
