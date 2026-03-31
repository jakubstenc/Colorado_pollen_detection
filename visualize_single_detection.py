import os
import json
import zipfile
import boto3
from botocore.client import Config
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# S3 configuration
s3_endpoint  = "https://s3.cl4.du.cesnet.cz"
s3_bucket    = "bucket"
access_key   = "1Y920BKC0SAWPNDE8RD6"
secret_key   = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

resource = boto3.resource(
    "s3",
    endpoint_url=s3_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
)
bucket = resource.Bucket(s3_bucket)

os.makedirs("sample_tiles", exist_ok=True)
zip_path = "sample_tiles/single_detection_results.zip"
extract_dir = "sample_tiles/single_detection_results"

# 1. Download the zip from S3
s3_key = "PEG/Colorado/staging_area/single_detection_results.zip"
try:
    print(f"Downloading {s3_key}...")
    bucket.download_file(s3_key, zip_path)
except Exception as e:
    print(f"Error downloading {s3_key}: {e}")
    # Might not exist if the job hasn't finished yet!
    exit(1)

# 2. Extract
print(f"Extracting to {extract_dir}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 3. Read metadata
manifest_path = os.path.join(extract_dir, "tile_manifest.json")
if not os.path.exists(manifest_path):
    print("No tile_manifest.json found!")
    exit(1)

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# 4. Parse annotations and visualize
results = []
output_viz_dir = "sample_tiles/visualizations"
os.makedirs(output_viz_dir, exist_ok=True)

manifest_dict = {m['tile_id']: m for m in manifest}

for split in ['train', 'val']:
    labels_dir = os.path.join(extract_dir, split, "labels")
    images_dir = os.path.join(extract_dir, split, "images")
    if not os.path.exists(labels_dir):
        continue

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        tile_id = label_file.replace(".txt", "")
        img_path = os.path.join(images_dir, f"{tile_id}.jpg")
        
        if tile_id not in manifest_dict:
            continue
            
        m = manifest_dict[tile_id]
        scale_x = m.get('um_per_px_x')
        scale_y = m.get('um_per_px_y')
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        H, W = img.shape[:2]
        
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            # parts[0] is class, the rest are x y pairs
            pts_norm = np.array(parts[1:], dtype=float).reshape(-1, 2)
            
            # Convert to pixel coordinates
            pts_px = (pts_norm * [W, H]).astype(np.int32)
            
            # Calculate Area in pixels
            area_px = cv2.contourArea(pts_px)
            
            # Calculate Area in sq microns
            area_um2 = 0
            if scale_x and scale_y:
                area_um2 = area_px * (scale_x * scale_y)
                
            results.append({
                "tile_id": tile_id,
                "detection_idx": i+1,
                "area_px": round(area_px, 2),
                "area_um2": round(area_um2, 2) if area_um2 else None,
                "um_per_px": scale_x
            })
            
            # Draw polygon on the image
            cv2.polylines(img, [pts_px], True, (0, 255, 0), 2)
            # Annotate with ID and area
            cv2.putText(img, f"#{i+1} Area: {area_um2:.0f}um2", (pts_px[0][0], pts_px[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
        if len(lines) > 0:
            viz_path = os.path.join(output_viz_dir, f"{tile_id}_viz.jpg")
            cv2.imwrite(viz_path, img)

df = pd.DataFrame(results)
print("Detection Results:")
if len(df) > 0:
    print(df.to_markdown(index=False))
else:
    print("No pollen detected.")

df.to_csv("sample_tiles/detection_results.csv", index=False)
print(f"\\nVisualization images saved to {output_viz_dir}/")
print(f"Results table saved to sample_tiles/detection_results.csv")
