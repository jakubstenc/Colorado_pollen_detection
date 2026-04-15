import sys
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

sys.path.append("/home/meow/Documents/Antigravity/Colorado_pollen_detection/src")
from build_species_dataset import tile_image, extract_general_pollen, get_mip_rgb, AICSImage

out_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Species_model/Trainig_data")
model_path = "/home/meow/Documents/Antigravity/Colorado_pollen_detection/best.pt"
src_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source/Cal_chi")

print("Loading model...")
model = YOLO(model_path)

czi_files = list(src_dir.glob("*.czi"))
print(f"Found {len(czi_files)} Cal_chi files. Picking up to 5.")
valid_keys = [f for f in czi_files if "20260227" not in f.name]
if not valid_keys:
    valid_keys = czi_files
import random

# We want 1 new file
target_count = 1

species = "Cal_chi"
class_id = 19

spec_img_dir = out_dir / species / "Images"
spec_lbl_dir = out_dir / species / "Labels"
spec_viz_dir = out_dir / species / "Vizualization"

spec_img_dir.mkdir(parents=True, exist_ok=True)
spec_lbl_dir.mkdir(parents=True, exist_ok=True)
spec_viz_dir.mkdir(parents=True, exist_ok=True)

success = 0
import time
random.seed(time.time())
random.shuffle(valid_keys)

for czi_path in valid_keys:
    # Check if we already processed this one:
    existing_imgs = list(spec_img_dir.glob(f"{species}_{czi_path.stem}_*"))
    if len(existing_imgs) > 0:
        continue # skip already processed
        
    print(f"Processing {czi_path.name}")
    try:
        tmp_path = Path("/tmp") / czi_path.name
        if not tmp_path.exists():
            print(f"Downloading to {tmp_path} from S3 directly...")
            import boto3
            import urllib3
            urllib3.disable_warnings()
            from botocore.config import Config
            import os
            
            s3_endpoint = "https://s3.cl4.du.cesnet.cz"
            s3_bucket = "bucket"
            key = f"PEG/Colorado/Source/Cal_chi/{czi_path.name}"
            
            s3 = boto3.client('s3', 
                endpoint_url=s3_endpoint, 
                verify=False,
                config=Config(signature_version='s3v4')
            )
            s3.download_file(s3_bucket, key, str(tmp_path))

        print("Loading AICSImage")
        img = AICSImage(str(tmp_path))
        rgb = get_mip_rgb(img)
        if len(rgb.shape) < 3 or rgb.shape[0] < 640 or rgb.shape[1] < 640:
            raise Exception(f"Invalid RGB Extracted: {rgb.shape}")
        
        n_tiles, n_hits = 0, 0
        for tile, tx, ty in tile_image(rgb):
            n_tiles += 1
            stem = f"{species}_{tmp_path.stem}_x{tx:06d}_y{ty:06d}"
            detections = extract_general_pollen(tile, model, 0.15)
            
            if len(detections) > 0:
                n_hits += 1
                tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(spec_img_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                lbl_lines = []
                for d in detections:
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in d['poly_norm'])
                    lbl_lines.append(f"{class_id} {coords}")
                (spec_lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines))
        
        print(f"Done processing {tmp_path.name}. Hits: {n_hits}/{n_tiles}")
        success += 1
        tmp_path.unlink()
        
    except Exception as e:
        print(f"Error on {czi_path.name}: {e}")
    if success >= target_count:
        break
print("Cal_chi extraction script finished.")
