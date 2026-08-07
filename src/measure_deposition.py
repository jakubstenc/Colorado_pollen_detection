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

def nms_numpy(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def safe_read_czi_rgb(filepath):
    import aicspylibczi
    from build_species_dataset import normalize_to_uint8

    czi = aicspylibczi.CziFile(filepath)
    dims = czi.dims
    
    kwargs = {}
    if 'Z' in dims: kwargs['Z'] = 0
    if 'T' in dims: kwargs['T'] = 0
    if 'C' in dims: kwargs['C'] = 0
    
    if czi.is_mosaic():
        data = czi.read_mosaic(**kwargs)
    else:
        data, _ = czi.read_image(**kwargs)
        
    data = np.squeeze(data)
    
    if len(data.shape) == 2:
        rgb = np.stack([data, data, data], axis=-1)
    elif len(data.shape) == 3:
        c_axis = np.argmin(data.shape)
        if c_axis == 0:
            rgb = np.transpose(data, (1, 2, 0))
        elif c_axis == 2:
            rgb = data
        else:
            raise ValueError(f"Unexpected channel axis in shape {data.shape}")
            
        if rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
        elif rgb.shape[-1] == 1:
            rgb = np.stack([rgb[..., 0]] * 3, axis=-1)
        elif rgb.shape[-1] == 2:
            rgb = np.stack([rgb[..., 0], rgb[..., 1], np.zeros_like(rgb[..., 0])], axis=-1)
    else:
        raise ValueError(f"Unexpected squeezed data shape: {data.shape}")
            
    return normalize_to_uint8(rgb)

def get_pixel_size_from_czi(filepath):
    import aicspylibczi
    try:
        czi = aicspylibczi.CziFile(filepath)
        root = czi.meta
        for distance in root.findall(".//Distance"):
            if distance.attrib.get("Id") == "X":
                value_elem = distance.find("Value")
                if value_elem is not None:
                    return float(value_elem.text) * 1e6
    except Exception:
        pass
    return 1.0

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
        
    czi_targets.sort(key=lambda x: x[0])
    shard_idx = int(os.environ.get('JOB_COMPLETION_INDEX', '0'))
    total_shards = 5
    my_targets = [t for i, t in enumerate(czi_targets) if i % total_shards == shard_idx]
    print(f'Shard {shard_idx}/{total_shards} processing {len(my_targets)} of {len(czi_targets)} images.')
    
    general_model_path = 'best.pt' if Path('best.pt').exists() else '/app/best.pt' if Path('/app/best.pt').exists() else '/home/meow/Documents/Antigravity/Colorado_pollen_detection/best.pt'
    if Path(general_model_path).exists():
        general_model = YOLO(general_model_path)
    else:
        print(f'General model {general_model_path} not found locally!')
        return
        
    print('Downloading latest species model from S3...')
    s3.download_file(s3_bucket, 'PEG/Colorado/trained_models/species_classifier/latest.pt', '/tmp/species_latest.pt')
    species_model = YOLO('/tmp/species_latest.pt')
    species_classes = species_model.names
    
    for target_key, filename, stigma_species in my_targets:
        print(f'\n--- Processing: {filename} (Stigma Species: {stigma_species}) ---')
    
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
        
        try:
            rgb = safe_read_czi_rgb(str(local_czi))
            px_size = get_pixel_size_from_czi(str(local_czi))
                
            print(f"Physical Pixel Size: {px_size} um/pixel")
            
            # Scale Normalization (match production training data scale)
            if px_size is not None and px_size < 0.65:
                rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
                px_size *= 2.0  # Adjust pixel size so physical measurements (area) remain accurate
            
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
        
            # Focus Check (match focus_check.py logic)
            blur_score = compute_focus_score(rgb)
            is_focused = blur_score >= 10.0
            status = "OK" if is_focused else "BLURRY"
            print(f"Focus Check: Score {blur_score:.2f} -> {status}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error reading image {filename}: {e}")
            summary = {
                "File": filename,
                "Stigma_Species": stigma_species,
                "Blur_Score": 0.0,
                "Status": "ERROR",
                "Total_Grains": 0,
                "Conspecific": 0,
                "Heterospecific": 0,
                "Day": day,
                "Month": month,
                "Day_of_Year": day_of_year
            }
            os.makedirs("results", exist_ok=True)
            out_summary_path = f"results/summary_{filename}.csv"
            pd.DataFrame([summary]).to_csv(out_summary_path, index=False)
            s3_dest_prefix = f"PEG/Colorado/Detected/Pollen_deposition/{stigma_species}"
            s3.upload_file(out_summary_path, s3_bucket, f"{s3_dest_prefix}/{Path(out_summary_path).name}")
            
            if local_czi.exists():
                local_czi.unlink()
            continue
        
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
            pd.DataFrame([summary]).to_csv(f"results/summary_{filename}.csv", index=False)
            pd.DataFrame(columns=["Grain_ID", "File", "Species_Predicted", "Class_Type", "Area_um2", "Circularity", "Conf", "X_px", "Y_px"]).to_csv(f"results/measurements_{filename}.csv", index=False)
            print("Image is blurry. Summary generated. Exiting.")
            if local_czi.exists():
                local_czi.unlink()
            continue

        conspecific_count = 0
        heterospecific_count = 0
        grain_id = 0
        
        # Setup directories for active learning UI
        al_base_prefix = f"PEG/Colorado/Species_model/Trainig_data/{stigma_species}"
        
        print("Processing tiles...")
        global_detections = []
        global_boxes = []
        global_scores = []
        
        for tile, tx, ty in tile_image(rgb, size=640, overlap=0.15):
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            detections = extract_general_pollen(tile_bgr, general_model, conf_thresh=0.45)
            
            if len(detections) > 0:
                stem = f"{stigma_species}_{local_czi.stem}_x{tx:06d}_y{ty:06d}"
                viz_bgr = tile_bgr.copy()
                lbl_lines = []
                
                for d in detections:
                    grain_id += 1
                    poly_px = d['poly_px']
                    
                    # Geometrics
                    area_px = cv2.contourArea(poly_px)
                    perimeter_px = cv2.arcLength(poly_px, True)
                    
                    area_um2 = area_px * (px_size ** 2)
                    
                    if area_um2 < 100.0:
                        continue
                    
                    if perimeter_px > 0:
                        circularity = 4 * np.pi * area_px / (perimeter_px ** 2)
                    else:
                        circularity = 0.0
                    
                    # Crop & Classify
                    x_min, y_min = poly_px[:, 0].min(), poly_px[:, 1].min()
                    x_max, y_max = poly_px[:, 0].max(), poly_px[:, 1].max()
                    
                    crop = tile_bgr[max(0, y_min):min(tile_bgr.shape[0], y_max), max(0, x_min):min(tile_bgr.shape[1], x_max)]
                    
                    if crop.size == 0:
                        class_name = "Unknown"
                        class_id = 0
                    else:
                        cls_res = species_model(crop, verbose=False)
                        if cls_res[0].probs is not None:
                            top_cls = cls_res[0].probs.top1
                            class_conf = float(cls_res[0].probs.top1conf)
                            class_name = species_classes.get(top_cls, "Unknown")
                            class_id = top_cls
                        else:
                            class_name = "Unclassified_Pollen"
                            class_conf = 0.0
                            class_id = 0
                            
                    display_text = f"{class_name} (Det: {d['conf']:.2f}, Cls: {class_conf:.2f})" if 'class_conf' in locals() else f"{class_name} (Det: {d['conf']:.2f})"
                    
                    # Draw on Tile Viz (For AL UI)
                    cv2.polylines(viz_bgr, [poly_px.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                    
                    # Collect global detection for NMS
                    global_poly = poly_px.copy()
                    global_poly[:, 0] += tx
                    global_poly[:, 1] += ty
                    
                    gx_min, gy_min = global_poly[:, 0].min(), global_poly[:, 1].min()
                    gx_max, gy_max = global_poly[:, 0].max(), global_poly[:, 1].max()
                    
                    global_boxes.append([gx_min, gy_min, gx_max, gy_max])
                    global_scores.append(d['conf'])
                    
                    is_conspecific = (class_name.lower() == stigma_species.lower()) or (class_name.lower() == "conspecific")
                    color = (0, 255, 0) if is_conspecific else (255, 0, 0)
                    
                    global_detections.append({
                        'global_poly': global_poly,
                        'class_id': class_id,
                        'class_name': class_name,
                        'conf': d['conf'],
                        'area_um2': area_um2,
                        'circularity': circularity,
                        'display_text': display_text,
                        'color': color
                    })
                    
                    # AL Label formatting
                    H, W = tile.shape[:2]
                    norm_xy = poly_px.astype(float)
                    norm_xy[:, 0] /= W
                    norm_xy[:, 1] /= H
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_xy)
                    lbl_lines.append(f"{class_id} {coords}")
                    
                # Upload to S3 for AL UI
                cv2.imwrite(f"/tmp/{stem}.jpg", tile_bgr)
                cv2.imwrite(f"/tmp/{stem}_viz.jpg", viz_bgr)
                with open(f"/tmp/{stem}.txt", "w") as f:
                    f.write("\n".join(lbl_lines))
                    
                s3.upload_file(f"/tmp/{stem}.jpg", s3_bucket, f"{al_base_prefix}/Images/{stem}.jpg")
                s3.upload_file(f"/tmp/{stem}_viz.jpg", s3_bucket, f"{al_base_prefix}/Vizualization/{stem}_viz.jpg")
                s3.upload_file(f"/tmp/{stem}.txt", s3_bucket, f"{al_base_prefix}/Labels/{stem}.txt")

                # Cleanup tile files immediately
                os.remove(f"/tmp/{stem}.jpg")
                os.remove(f"/tmp/{stem}_viz.jpg")
                os.remove(f"/tmp/{stem}.txt")

        # Process NMS on global overview
        keep_indices = nms_numpy(global_boxes, global_scores, iou_threshold=0.3)
        for idx in keep_indices:
            det = global_detections[idx]
            grain_id += 1
            if det['class_name'] == "Conspecific":
                conspecific_count += 1
            elif det['class_name'] == "Heterospecific":
                heterospecific_count += 1
            else:
                heterospecific_count += 1
                
            cx = int(det['global_poly'][:, 0].mean())
            cy = int(det['global_poly'][:, 1].mean())
                
            measurements.append({
                "Grain_ID": grain_id,
                "File": filename,
                "Day": day,
                "Month": month,
                "Day_of_Year": day_of_year,
                "Species_Predicted": det['class_name'],
                "Class_Type": "Conspecific" if det['class_name'] == "Conspecific" else "Heterospecific",
                "Area_um2": round(det['area_um2'], 2),
                "Circularity": round(det['circularity'], 3),
                "Conf": round(det['conf'], 3),
                "X_px": cx,
                "Y_px": cy
            })
            
            # Draw on Overview
            overview_poly = (det['global_poly'] * overview_scale).astype(int)
            cv2.polylines(overview_img, [overview_poly.reshape((-1, 1, 2))], True, det['color'], 2)
            
            px, py = overview_poly[0]
            cv2.putText(overview_img, det['display_text'], (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(overview_img, det['display_text'], (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

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
        s3_dest_prefix = f"PEG/Colorado/Detected/Pollen_deposition/{stigma_species}"
        print(f"Uploading results to S3 ({s3_dest_prefix})...")
        s3.upload_file(str(out_img_path), s3_bucket, f"{s3_dest_prefix}/{out_img_path.name}")
        s3.upload_file(str(out_csv_path), s3_bucket, f"{s3_dest_prefix}/{Path(out_csv_path).name}")
        s3.upload_file(str(out_summary_path), s3_bucket, f"{s3_dest_prefix}/{Path(out_summary_path).name}")
        
        # Cleanup
        if local_czi.exists():
            local_czi.unlink()
        if out_img_path.exists():
            out_img_path.unlink()
        if Path(out_csv_path).exists():
            Path(out_csv_path).unlink()
        if Path(out_summary_path).exists():
            Path(out_summary_path).unlink()
        
        print("Done! Check S3 Detected directory.")
    
if __name__ == "__main__":
    main()
