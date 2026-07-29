import sys
import os
from pathlib import Path
import cv2
import csv
import boto3
from botocore.config import Config
from aicsimageio import AICSImage
import numpy as np
import urllib3
urllib3.disable_warnings()

sys.path.append("/home/meow/Documents/Antigravity/Colorado_pollen_detection/src")
from build_species_dataset import get_mip_rgb

def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "1Y920BKC0SAWPNDE8RD6")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")
    
    config = Config(connect_timeout=60, retries={'max_attempts': 5})
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
        config=config
    )

def compute_focus_score(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Process in chunks to avoid OOM on massive images
    chunk_size = 4000
    h, w = gray.shape
    
    n_pixels = h * w
    sum_x = 0.0
    sum_x2 = 0.0
    
    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            y_end = min(y + chunk_size, h)
            x_end = min(x + chunk_size, w)
            
            y_start_pad = max(0, y - 1)
            y_end_pad = min(h, y_end + 1)
            x_start_pad = max(0, x - 1)
            x_end_pad = min(w, x_end + 1)
            
            chunk = gray[y_start_pad:y_end_pad, x_start_pad:x_end_pad]
            lap = cv2.Laplacian(chunk, cv2.CV_64F)
            
            valid_y_start = 1 if y > 0 else 0
            valid_y_end = lap.shape[0] - (1 if y_end < h else 0)
            valid_x_start = 1 if x > 0 else 0
            valid_x_end = lap.shape[1] - (1 if x_end < w else 0)
            
            valid_lap = lap[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            
            sum_x += np.sum(valid_lap)
            sum_x2 += np.sum(valid_lap ** 2)
            
    mean = sum_x / n_pixels
    variance = (sum_x2 / n_pixels) - (mean ** 2)
    return variance

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Focus assessment for .czi files with auto downsampling")
    parser.add_argument("--dir", type=str, default=None, help="Local directory containing .czi files")
    parser.add_argument("--s3-prefix", type=str, default="PEG/Colorado/Source/Pollen_deposition/", help="S3 prefix")
    parser.add_argument("--output", type=str, default="results/deposition_focus_report.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=10, help="Number of files to process (default: 10)")
    parser.add_argument("--threshold", type=float, default=150.0, help="Focus threshold cutoff")
    args = parser.parse_args()

    s3 = get_s3_client()
    bucket = "bucket"
    
    czi_targets = []
    
    if args.dir and Path(args.dir).exists():
        czi_files = list(Path(args.dir).rglob("*.czi"))
        for p in czi_files:
            czi_targets.append(("local", str(p), p.name, p.parent.name))
    else:
        print(f"Polling S3 for CZIs under {args.s3_prefix}...")
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=args.s3_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".czi"):
                    parts = obj["Key"].split("/")
                    folder = parts[-2] if len(parts) > 1 else "Unknown"
                    czi_targets.append(("s3", obj["Key"], parts[-1], folder))
                    
    if not czi_targets:
        print("No .czi files found!")
        return

    import random
    random.seed(42)
    # Ensure representation across species folders if possible
    by_folder = {}
    for item in czi_targets:
        folder = item[3]
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(item)

    selected = []
    per_folder_limit = max(1, args.limit // max(1, len(by_folder)))
    for folder, items in by_folder.items():
        random.shuffle(items)
        selected.extend(items[:per_folder_limit])
    
    # Fill up to limit if selected < limit
    if len(selected) < args.limit:
        remaining = [i for i in czi_targets if i not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:args.limit - len(selected)])
        
    selected = selected[:args.limit]
    print(f"Selected {len(selected)} sample files across folders: {set(x[3] for x in selected)}")

    results = []
    
    for i, (source_type, location, filename, folder) in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] Processing [{folder}] {filename}...")
        tmp_path = Path(f"/tmp/{filename}")
        
        try:
            if source_type == "s3":
                s3.download_file(bucket, location, str(tmp_path))
                img_path = str(tmp_path)
            else:
                img_path = location
                
            img = AICSImage(img_path)
            rgb = get_mip_rgb(img)
            
            # Check spatial resolution & apply cautious 2x downsampling if needed
            px_size = getattr(img.physical_pixel_sizes, 'X', None)
            scale_applied = "1.0x"
            if px_size is not None and px_size < 0.65:
                # ~0.44 um/px -> 2x downsample to ~0.88 um/px
                rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
                scale_applied = "0.5x (2x downsampled)"
                
            score = compute_focus_score(rgb)
            status = "OK" if score >= args.threshold else "BLURRY (RE-DO)"
            
            print(f"   => Pixel Size: {px_size} um/px | Scale: {scale_applied}")
            print(f"   => Focus Score: {score:.2f} [{status}]")
            
            results.append({
                "species_folder": folder,
                "file": filename,
                "pixel_size_x": px_size,
                "scale_applied": scale_applied,
                "focus_score": round(score, 2),
                "status": status
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "species_folder": folder,
                "file": filename,
                "pixel_size_x": None,
                "scale_applied": "ERROR",
                "focus_score": -1,
                "status": f"ERROR: {e}"
            })
        finally:
            if source_type == "s3" and tmp_path.exists():
                tmp_path.unlink()
                
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["species_folder", "file", "pixel_size_x", "scale_applied", "focus_score", "status"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSaved focus report to {args.output}")
    print("\nSummary Results:")
    for r in sorted(results, key=lambda x: x["focus_score"], reverse=True):
        print(f" [{r['species_folder']}] {r['focus_score']:8.2f} | {r['status']:14s} | {r['file']}")

if __name__ == "__main__":
    main()

