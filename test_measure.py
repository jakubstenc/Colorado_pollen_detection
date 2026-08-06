import sys
import os
import cv2
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append("/home/meow/Documents/Antigravity/Colorado_pollen_detection/src")
from measure_deposition import get_s3_client
from build_species_dataset import get_mip_rgb, tile_image, extract_general_pollen, AICSImage
from focus_check import compute_focus_score
from ultralytics import YOLO

def main():
    s3 = get_s3_client()
    s3_bucket = "bucket"
    target_key = "PEG/Colorado/Source/Pollen_deposition/Ran_ado/20260701_001_Dep_Ran_Ado_24_7_161b_Colorado2025.czi"
    filename = Path(target_key).name
    stigma_species = "Ran_ado"
    
    general_model = YOLO("best.pt")
    
    local_czi = Path(f"/tmp/{filename}")
    if not local_czi.exists():
        print("Downloading...")
        s3.download_file(s3_bucket, target_key, str(local_czi))
    
    img = AICSImage(str(local_czi))
    rgb = get_mip_rgb(img)
    
    px_size = getattr(img.physical_pixel_sizes, 'X', 1.0)
    if px_size is None: px_size = 1.0
    print(f"Original Physical Pixel Size: {px_size} um/pixel")
    
    # We removed the downsampling here.
    print(f"Shape after (no) downsampling: {rgb.shape}")
    
    blur_score = compute_focus_score(rgb)
    is_focused = blur_score >= 50.0
    status = "OK" if is_focused else "BLURRY"
    print(f"Focus Check: Score {blur_score:.2f} -> {status}")
    
    print("Testing general model on one tile...")
    for tile, tx, ty in tile_image(rgb, size=640, overlap=0.15):
        detections = extract_general_pollen(tile, general_model, conf_thresh=0.65)
        print(f"Tile x:{tx} y:{ty} detections: {len(detections)}")
        break

if __name__ == "__main__":
    main()
