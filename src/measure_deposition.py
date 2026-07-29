import sys
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
import boto3
from botocore.config import Config
import urllib3
urllib3.disable_warnings()

from aicsimageio import AICSImage
from ultralytics import YOLO

# Add src to path
sys.path.append("/home/meow/Documents/Antigravity/Colorado_pollen_detection/src")
sys.path.append("/app/src")
sys.path.append("/scripts")
sys.path.append("src")
from build_species_dataset import get_mip_rgb, tile_image, extract_general_pollen
from focus_check import compute_focus_score

s3_endpoint = "https://s3.cl4.du.cesnet.cz"
s3_bucket = "bucket"
aws_access_key_id = "1Y920BKC0SAWPNDE8RD6"
aws_secret_access_key = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
        verify=False
    )

def main():
    s3 = get_s3_client()
    
    # 1. Select Random Deposition Image
    print("Fetching list of deposition images from S3...")
    prefix = "PEG/Colorado/Source/Pollen_deposition/"
    paginator = s3.get_paginator('list_objects_v2')
    
    czi_targets = []
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".czi"):
                parts = obj["Key"].split("/")
                folder = parts[-2] if len(parts) > 1 else "Unknown"
                czi_targets.append((obj["Key"], parts[-1], folder))
                
    if not czi_targets:
        print("No CZI images found!")
        return
        
    random.seed()
    target_key, filename, stigma_species = random.choice(czi_targets)
    print(f"Selected: {filename} (Stigma Species: {stigma_species})")
    
    # Parse date from filename
    import re
    from datetime import datetime
    
    day = None
    month = None
    day_of_year = None
    
    match = re.search(r'Dep_[a-zA-Z]+_[a-zA-Z]+_(\d{1,2})_(\d{1,2})_', filename, re.IGNORECASE)
    if not match:
        match = re.search(r'_(\d{1,2})_(\d{1,2})_', filename)
        
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        
        # Try to infer year, fallback to 2025
        year = 2025
        if "2026" in filename: year = 2026
        elif "2024" in filename: year = 2024
        
        try:
            date_obj = datetime(year, month, day)
            day_of_year = date_obj.timetuple().tm_yday
        except ValueError:
            pass
    
    # Download
    local_czi = Path(f"/tmp/{filename}")
    print("Downloading...")
    s3.download_file(s3_bucket, target_key, str(local_czi))
    
    img = AICSImage(str(local_czi))
    
    # Fast path for 3-channel brightfield images to avoid massive memory spikes
    # in normalize_to_uint8 contrast stretching
    if 'S' in img.dims.order and getattr(img.dims, 'S', 1) == 3:
        rgb = img.get_image_data("YXS", T=0, C=0, Z=0)
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = (rgb >> 8).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
    else:
        rgb = get_mip_rgb(img)
    
    px_size = getattr(img.physical_pixel_sizes, 'X', 1.0) # Default to 1.0 if None
    if px_size is None:
        px_size = 1.0
        
    print(f"Physical Pixel Size: {px_size} um/pixel")
    
    # Downscale the overview image early to avoid massive memory usage and OOMs
    # This reduces memory from ~9GB to ~100MB during Laplacian computation
    max_dim = 4000
    h_full, w_full = rgb.shape[:2]
    overview_scale = 1.0
    if max(h_full, w_full) > max_dim:
        overview_scale = max_dim / max(h_full, w_full)
        # Use numpy slicing for fast, memory-efficient nearest-neighbor downsampling
        stride = int(1.0 / overview_scale)
        overview_img = rgb[::stride, ::stride].copy()
        # Actual scale after integer stride might be slightly different
        overview_scale = 1.0 / stride
    else:
        overview_img = rgb.copy()

    # 2. Focus Check (run on a center crop of full-res to preserve Laplacian statistics without OOM)
    cy, cx = h_full // 2, w_full // 2
    crop_size = 1024
    y1, y2 = max(0, cy - crop_size//2), min(h_full, cy + crop_size//2)
    x1, x2 = max(0, cx - crop_size//2), min(w_full, cx + crop_size//2)
    focus_crop = rgb[y1:y2, x1:x2]
    
    blur_score = compute_focus_score(focus_crop)
    is_focused = blur_score >= 150.0
    status = "OK" if is_focused else "BLURRY"
    print(f"Focus Check: Score {blur_score:.2f} -> {status}")
    
    measurements = []
    summary = {
        "File": filename,
        "Stigma_Species": stigma_species,
        "Blur_Score": round(blur_score, 2),
        "Status": status,
        "Total_Grains": 0,
        "Conspecific": 0,
        "Heterospecific": 0
    }
    
    if not is_focused:
        # Generate summary and exit
        os.makedirs("results", exist_ok=True)
        pd.DataFrame([summary]).to_csv("results/sample_summary.csv", index=False)
        pd.DataFrame(columns=["Grain_ID", "File", "Species_Predicted", "Class_Type", "Area_um2", "Circularity", "Conf"]).to_csv("results/pollen_grains_measurements.csv", index=False)
        print("Image is blurry. Summary generated. Exiting.")
        return

    # 3. Load Models
    general_model_path = "best.pt" if Path("best.pt").exists() else "/app/best.pt" if Path("/app/best.pt").exists() else "/home/meow/Documents/Antigravity/Colorado_pollen_detection/best.pt"
    if Path(general_model_path).exists():
        general_model = YOLO(general_model_path)
    else:
        print(f"General model {general_model_path} not found locally!")
        return
        
    # Download species model
    print("Downloading latest species model from S3...")
    s3.download_file(s3_bucket, "PEG/Colorado/trained_models/species_classifier/latest.pt", "/tmp/species_latest.pt")
    species_model = YOLO("/tmp/species_latest.pt")
    species_classes = species_model.names
    conspecific_count = 0
    heterospecific_count = 0
    grain_id = 0
    
    # Setup directories for active learning UI
    al_base_prefix = f"PEG/Colorado/Species_model/Trainig_data/{stigma_species}"
    
    print("Processing tiles...")
    for tile, tx, ty in tile_image(rgb, size=640, overlap=0.15):
        detections = extract_general_pollen(tile, general_model, conf_thresh=0.25)
        
        if len(detections) > 0:
            stem = f"{stigma_species}_{local_czi.stem}_x{tx:06d}_y{ty:06d}"
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            viz_bgr = tile_bgr.copy()
            lbl_lines = []
            
            for d in detections:
                grain_id += 1
                poly_px = d['poly_px']
                
                # Geometrics
                area_px = cv2.contourArea(poly_px)
                perimeter_px = cv2.arcLength(poly_px, True)
                
                area_um2 = area_px * (px_size ** 2)
                
                if perimeter_px > 0:
                    circularity = 4 * np.pi * area_px / (perimeter_px ** 2)
                else:
                    circularity = 0.0
                
                # Crop & Classify
                x_min, y_min = poly_px[:, 0].min(), poly_px[:, 1].min()
                x_max, y_max = poly_px[:, 0].max(), poly_px[:, 1].max()
                
                crop = tile[max(0, y_min):min(tile.shape[0], y_max), max(0, x_min):min(tile.shape[1], x_max)]
                
                if crop.size == 0:
                    class_name = "Unknown"
                    class_id = 0
                else:
                    cls_res = species_model(crop, verbose=False)
                    if cls_res[0].probs is not None:
                        top_cls = cls_res[0].probs.top1
                        class_name = species_classes.get(top_cls, "Unknown")
                        class_id = top_cls
                    else:
                        class_name = "Unclassified_Pollen"
                        class_id = 0
                        
                is_conspecific = (class_name.lower() == stigma_species.lower())
                if is_conspecific:
                    conspecific_count += 1
                    class_type = "Conspecific"
                    color = (0, 255, 0)
                else:
                    heterospecific_count += 1
                    class_type = "Heterospecific"
                    color = (0, 0, 255)
                    
                measurements.append({
                    "Grain_ID": grain_id,
                    "File": filename,
                    "Day": day,
                    "Month": month,
                    "Day_of_Year": day_of_year,
                    "Species_Predicted": class_name,
                    "Class_Type": class_type,
                    "Area_um2": round(area_um2, 2),
                    "Circularity": round(circularity, 3),
                    "Conf": round(d['conf'], 3)
                })
                
                # AL Label formatting
                H, W = tile.shape[:2]
                norm_xy = poly_px.astype(float)
                norm_xy[:, 0] /= W
                norm_xy[:, 1] /= H
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_xy)
                lbl_lines.append(f"{class_id} {coords}")
                
                # Draw on Tile Viz
                cv2.polylines(viz_bgr, [poly_px.reshape((-1, 1, 2))], True, color, 2)
                
                # Draw on Overview
                global_poly = poly_px.copy()
                global_poly[:, 0] += tx
                global_poly[:, 1] += ty
                
                # Scale for downscaled overview image
                overview_poly = (global_poly * overview_scale).astype(int)
                
                cv2.polylines(overview_img, [overview_poly.reshape((-1, 1, 2))], True, color, 2)
                
                # Add text to overview
                px, py = overview_poly[0]
                cv2.putText(overview_img, class_name, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(overview_img, class_name, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
                
            # Upload to S3 for AL UI
            cv2.imwrite(f"/tmp/{stem}.jpg", tile_bgr)
            cv2.imwrite(f"/tmp/{stem}_viz.jpg", viz_bgr)
            with open(f"/tmp/{stem}.txt", "w") as f:
                f.write("\n".join(lbl_lines))
                
            s3.upload_file(f"/tmp/{stem}.jpg", s3_bucket, f"{al_base_prefix}/Images/{stem}.jpg")
            s3.upload_file(f"/tmp/{stem}_viz.jpg", s3_bucket, f"{al_base_prefix}/Vizualization/{stem}_viz.jpg")
            s3.upload_file(f"/tmp/{stem}.txt", s3_bucket, f"{al_base_prefix}/Labels/{stem}.txt")

    summary["Day"] = day
    summary["Month"] = month
    summary["Day_of_Year"] = day_of_year
    summary["Total_Grains"] = grain_id
    summary["Conspecific"] = conspecific_count
    summary["Heterospecific"] = heterospecific_count
    
    out_summary_path = f"results/summary_{filename}.csv"
    out_csv_path = f"results/measurements_{filename}.csv"
    
    os.makedirs("results", exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_summary_path, index=False)
    pd.DataFrame(measurements).to_csv(out_csv_path, index=False)
    
    overview_bgr = cv2.cvtColor(overview_img, cv2.COLOR_RGB2BGR)
        
    out_img_path = Path(f"/tmp/overview_{filename}.jpg")
    cv2.imwrite(str(out_img_path), overview_bgr)
    
    # Upload to S3
    print("Uploading results to S3...")
    s3.upload_file(str(out_img_path), s3_bucket, f"PEG/Colorado/Detected/{out_img_path.name}")
    s3.upload_file(str(out_csv_path), s3_bucket, f"PEG/Colorado/Detected/{out_csv_path.name}")
    s3.upload_file(str(out_summary_path), s3_bucket, f"PEG/Colorado/Detected/{out_summary_path.name}")
    
    print("Done! Check S3 Detected directory.")

if __name__ == "__main__":
    main()
