import sys
import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from aicsimageio import AICSImage
import cv2
import boto3
from botocore.config import Config
import urllib3
urllib3.disable_warnings()

sys.path.append("/home/meow/Documents/Antigravity/Colorado_pollen_detection/src")
from build_species_dataset import get_mip_rgb, tile_image, extract_general_pollen
from focus_check import compute_focus_score, get_s3_client

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Conspecific vs Heterospecific Pollen Analysis")
    parser.add_argument("--dir", type=str, default=None, help="Local directory containing Pollen_deposition folders")
    parser.add_argument("--s3-prefix", type=str, default="PEG/Colorado/Source/Pollen_deposition/", help="S3 prefix")
    parser.add_argument("--model", type=str, help="Path to species-specific YOLO model", default="/home/meow/Documents/Antigravity/Colorado_pollen_detection/best.pt")
    parser.add_argument("--output", type=str, default="results/conspecific_heterospecific_summary.csv")
    parser.add_argument("--blur-threshold", type=float, default=150.0, help="Laplacian variance threshold for blur")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of sample files to process")
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

    if len(selected) < args.limit:
        remaining = [i for i in czi_targets if i not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:args.limit - len(selected)])

    selected = selected[:args.limit]
    print(f"Selected {len(selected)} sample deposition files: {[x[2] for x in selected]}")

    print(f"Loading species model from {args.model}")
    try:
        model = YOLO(args.model)
        model_classes = model.names
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results = []

    for i, (source_type, location, filename, stigma_species) in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] Analyzing Stigma Species [{stigma_species}]: {filename}")
        tmp_path = Path(f"/tmp/{filename}")
        
        try:
            if source_type == "s3":
                s3.download_file(bucket, location, str(tmp_path))
                img_path = str(tmp_path)
            else:
                img_path = location
                
            img = AICSImage(img_path)
            rgb = get_mip_rgb(img)
            
            # Spatial Resolution Check & Cautious 2x Downsampling
            px_size = getattr(img.physical_pixel_sizes, 'X', None)
            scale_applied = "1.0x"
            if px_size is not None and px_size < 0.65:
                rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
                scale_applied = "0.5x (2x downsampled)"
                
            blur_score = compute_focus_score(rgb)
            if blur_score < args.blur_threshold:
                print(f"   -> ⚠️ Skipping due to blur (score: {blur_score:.2f} < threshold {args.blur_threshold})")
                results.append({
                    "Stigma_Species": stigma_species,
                    "File": filename,
                    "Pixel_Size": px_size,
                    "Scale_Applied": scale_applied,
                    "Total_Pollen": 0,
                    "Conspecific": 0,
                    "Heterospecific": 0,
                    "Blur_Score": round(blur_score, 2),
                    "Status": "BLURRY"
                })
                continue

            print(f"   -> ✅ Focused (score: {blur_score:.2f}). Running YOLO inference...")
            conspecific_count = 0
            heterospecific_count = 0

            # Run tiling inference over 640x640 windows
            for tile, tx, ty in tile_image(rgb, size=640, overlap=0.15):
                detections = extract_general_pollen(tile, model, conf_thresh=0.25)
                
                for d in detections:
                    # Crop bbox region from tile for multi-class classification
                    poly_px = d['poly_px']
                    x_min, y_min = poly_px[:, 0].min(), poly_px[:, 1].min()
                    x_max, y_max = poly_px[:, 0].max(), poly_px[:, 1].max()
                    
                    crop = tile[max(0, y_min):min(tile.shape[0], y_max), max(0, x_min):min(tile.shape[1], x_max)]
                    if crop.size == 0:
                        class_name = "Unknown"
                    else:
                        cls_res = model(crop, verbose=False)
                        if len(cls_res[0].boxes) > 0:
                            top_cls = int(cls_res[0].boxes.cls[0].item())
                            class_name = model_classes.get(top_cls, "Unknown")
                        else:
                            class_name = "Unclassified_Pollen"

                    # Conspecific if classified species matches stigma species name exactly
                    if class_name.lower() == stigma_species.lower():
                        conspecific_count += 1
                    else:
                        # Heterospecific includes other known pollen species and unclassified/novel pollen grains
                        heterospecific_count += 1

            print(f"   ✓ Results: Conspecific={conspecific_count} | Heterospecific={heterospecific_count} | Total={conspecific_count + heterospecific_count}")

            results.append({
                "Stigma_Species": stigma_species,
                "File": filename,
                "Pixel_Size": px_size,
                "Scale_Applied": scale_applied,
                "Total_Pollen": conspecific_count + heterospecific_count,
                "Conspecific": conspecific_count,
                "Heterospecific": heterospecific_count,
                "Blur_Score": round(blur_score, 2),
                "Status": "OK"
            })
            
        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")
            results.append({
                "Stigma_Species": stigma_species,
                "File": filename,
                "Pixel_Size": None,
                "Scale_Applied": "ERROR",
                "Total_Pollen": 0,
                "Conspecific": 0,
                "Heterospecific": 0,
                "Blur_Score": -1,
                "Status": f"ERROR: {e}"
            })
        finally:
            if source_type == "s3" and tmp_path.exists():
                tmp_path.unlink()

    df = pd.DataFrame(results)
    if not df.empty:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nAnalysis complete. Results saved to {args.output}")
        print("\nSummary Grouped by Stigma Species:")
        print(df.groupby("Stigma_Species")[["Conspecific", "Heterospecific", "Total_Pollen"]].sum())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

