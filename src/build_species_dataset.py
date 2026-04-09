import os
import cv2
import json
import argparse
import random
import numpy as np
import subprocess
from pathlib import Path
from aicsimageio import AICSImage
from ultralytics import YOLO

import csv
import boto3
import urllib3
urllib3.disable_warnings()
from botocore.client import Config

def load_species_manifest(manifest_path):
    registry = {}
    if not os.path.exists(manifest_path):
        print(f"⚠️ WARNING: Manifest file '{manifest_path}' not found! The extraction will skip all files.")
        return registry
        
    with open(manifest_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['species_code'].strip()
            cid = int(row['class_id'].strip())
            registry[code] = cid
            
    print(f"📋 Loaded {len(registry)} species definitions from manifest.")
    return registry

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-channel 0.5-99.5 percentile contrast stretch -> uint8 RGB."""
    out = np.zeros(arr.shape[:2] + (arr.shape[2],), dtype=np.uint8)
    for c in range(arr.shape[2]):
        ch = arr[..., c].astype(np.float32)
        valid = ch[ch > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [0.5, 99.5])
        ch = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1) * 255.0
        out[..., c] = ch.astype(np.uint8)
    return out

def get_mip_rgb(img, channels=(0, 1, 2), grayscale=False) -> np.ndarray:
    """
    Compute Maximum Intensity Projection across Z for the given channels.
    Returns uint8 RGB array (H, W, 3).
    """
    if 'S' in img.dims.order and getattr(img.dims, 'S', 1) == 3:
        rgb = img.get_image_data("YXS", T=0, C=0, Z=0)
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = (rgb / 256).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        if grayscale:
            import cv2
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            rgb = np.stack([gray, gray, gray], axis=-1)
        return rgb

    kwargs = {}
    if 'S' in img.dims.order: kwargs['S'] = 0
    if 'T' in img.dims.order: kwargs['T'] = 0
    
    dask_czyx = img.get_image_dask_data("CZYX", **kwargs)
    num_c = dask_czyx.shape[0]
    channels_data = []
    target_channels = [c for c in channels[:3] if c < num_c]
    
    for c in target_channels:
        mip = dask_czyx[c].max(axis=0).compute()
        channels_data.append(mip)

    if grayscale and len(channels_data) > 0:
        combined = np.max(np.stack(channels_data, axis=-1), axis=-1)
        channels_data = [combined, combined, combined]
    else:
        while len(channels_data) < 3:
            if len(channels_data) > 0:
                channels_data.append(np.zeros_like(channels_data[0]))
            else:
                channels_data.append(np.zeros((1024, 1024), dtype=np.uint8))

        is_single_channel = (len(target_channels) == 1)
        if is_single_channel:
            active_ch = channels_data[0]
            channels_data = [active_ch, active_ch, active_ch]

    rgb = np.stack(channels_data, axis=-1)
    return normalize_to_uint8(rgb)

def tile_image(rgb, size=640, overlap=0.2):
    stride = int(size * (1 - overlap))
    H, W, _ = rgb.shape
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            y_end, x_end = min(y + size, H), min(x + size, W)
            tile = np.zeros((size, size, 3), dtype=np.uint8)
            crop = rgb[y:y_end, x:x_end]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            yield tile, x, y

def extract_general_pollen(tile, model, conf_thresh):
    """
    Runs the general_pollen YOLOv8 segmentation model on a 640x640 tile.
    Returns: list of dictionary detections -> [{'poly': [x1, y1, ...], 'conf': 0.9}]
    """
    results = model(tile, verbose=False)
    detections = []

    if results[0].masks is None or results[0].boxes is None:
        return detections

    H, W = tile.shape[:2]
    # In segmentation YOLO models, masks.xy holds the polygons
    for mask_xy, box in zip(results[0].masks.xy, results[0].boxes):
        if mask_xy.shape[0] < 3:
            continue
            
        c_conf = float(box.conf[0])
        if c_conf < conf_thresh:
            continue
            
        # Normalize polygon points to 0.0 - 1.0 for YOLO string definitions
        norm = mask_xy.copy().astype(float)
        norm[:, 0] /= W
        norm[:, 1] /= H
        
        detections.append({
            'poly_norm': norm,
            'poly_px': mask_xy.copy().astype(np.int32),
            'conf': c_conf
        })
        
    return detections

def generate_stats_report(stats, out_dir, registry):
    """
    Creates dynamic Quarto report summing up the totals from the extracted datasets.
    """
    stats_dir = out_dir / "Stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    qmd_path = stats_dir / "dataset_stats.qmd"
    
    total_imgs = sum(s['images'] for s in stats.values())
    total_negs = sum(s['negatives'] for s in stats.values())
    total_anns = sum(s['annotations'] for s in stats.values())
    
    # Write the QMD
    with open(qmd_path, 'w') as f:
        f.write("---\n")
        f.write("title: 'Species Model: Training Data Summary'\n")
        f.write("author: 'Antigravity Pipeline'\n")
        f.write("format:\n")
        f.write("  html: default\n")
        f.write("  pdf: default\n")
        f.write("---\n\n")
        
        f.write("## 📊 Overall Dataset Compilation\n\n")
        f.write(f"- **Total Pure Negatives Generated**: `{total_negs}`\n")
        f.write(f"- **Total Species Images Generated**: `{total_imgs}`\n")
        f.write(f"- **Total Pollen Grain Annotations**: `{total_anns}`\n\n")
        
        f.write("## 🌸 Species Breakdown\n\n")
        f.write("| Species Code | Class ID | Images | Annotations |\n")
        f.write("|--------------|----------|--------|-------------|\n")
        
        for sp, data in sorted(stats.items()):
            if sp == "Negatives": continue
            class_id = registry.get(sp, "Unknown")
            f.write(f"| `{sp}` | `{class_id}` | {data['images']} | {data['annotations']} |\n")
            
    print(f"\n📄 Rendering Quarto Specs in {stats_dir}...")
    try:
        subprocess.run(["quarto", "render", str(qmd_path), "--to", "typst"], check=True)
        subprocess.run(["quarto", "render", str(qmd_path), "--to", "html"], check=True)
        print("✅ Quarto PDF and HTML rendering complete!")
    except Exception as e:
        print(f"⚠️ Quarto rendering failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Species Model Dataset Generator")
    parser.add_argument("--root", required=True, help="Directory containing raw CZI scans")
    parser.add_argument("--out", required=True, help="Path to Species_model/Trainig_data/ structure")
    parser.add_argument("--model", required=True, help="Path to the trained general_pollen model")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold to dictate positive vs negative")
    parser.add_argument("--manifest", type=str, default="/app/species_manifest.csv", help="Path to species manifest CSV")
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out)
    
    print(f"🤖 Loading general pollen model: {args.model}")
    model = YOLO(args.model)
    
    neg_dir = out_dir / "Negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    registry = load_species_manifest(args.manifest)

    print("\n🌍 Connecting to S3 to dynamically list and select 1 CZI per species...")
    s3_endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    s3_bucket = os.environ.get("S3_BUCKET", "bucket")
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    s3_kwargs = {"endpoint_url": s3_endpoint, "verify": False, "config": Config(signature_version="s3v4", s3={"payload_signing_enabled": False})}
    if aws_access_key_id and aws_secret_access_key:
        s3_kwargs["aws_access_key_id"] = aws_access_key_id
        s3_kwargs["aws_secret_access_key"] = aws_secret_access_key
        
    s3 = boto3.client("s3", **s3_kwargs)
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix="PEG/Colorado/Source/")
    
    czi_keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".czi"):
                czi_keys.append(obj["Key"])
                
    print(f"🔍 Found {len(czi_keys)} raw CZIs on S3.")
    
    czi_by_species = {}
    for key in czi_keys:
        filename = key.split("/")[-1]
        species = None
        for code in registry.keys():
            if code.lower() in filename.lower():
                species = code
                break
        if species:
            if species not in czi_by_species:
                czi_by_species[species] = []
            czi_by_species[species].append(key)
            
    print(f"🎲 Will attempt to dynamically process {len(czi_by_species)} species with fallback handling...")
    
    for sp, keys_for_species in czi_by_species.items():            
        if not keys_for_species:
            continue
            
        # Filter physically massive, completely corrupted CZIs from 20260227
        # Sort explicitly in reverse to probabilistically guarantee orthogonal selections
        valid_keys = [k for k in keys_for_species if '20260227' not in k]
        
        # If the blocklist removes literally everything (unlikely), fallback seamlessly
        if not valid_keys:
            valid_keys = keys_for_species
            
        import random
        random.seed(4242)
        random.shuffle(valid_keys)
        keys_for_species = valid_keys
        success_count = 0
        
        for czi_key in keys_for_species:
            filename = czi_key.split("/")[-1]
            czi_path = root_dir / filename
            
            print(f"\n📥 Downloading {filename} from S3...")
            s3.download_file(s3_bucket, czi_key, str(czi_path))
            
            species = sp
            class_id = registry[species]
            
            if species not in stats:
                stats[species] = {"images": 0, "annotations": 0, "negatives": 0}
                
            # Ensure hierarchy boundaries
            spec_img_dir = out_dir / species / "Images"
            spec_lbl_dir = out_dir / species / "Labels"
            spec_viz_dir = out_dir / species / "Vizualization"
            
            spec_img_dir.mkdir(parents=True, exist_ok=True)
            spec_lbl_dir.mkdir(parents=True, exist_ok=True)
            spec_viz_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🪚 Extracting {czi_path.name}")
            print(f"   -> Assigning class [{class_id}] ({species})")
            
            try:
                img = AICSImage(str(czi_path))
                rgb = get_mip_rgb(img)
                
                # Verify that it didn't just load garbage that will crash silently later.
                if len(rgb.shape) < 3 or rgb.shape[0] < 640 or rgb.shape[1] < 640:
                    raise Exception(f"Invalid RGB Extracted: {rgb.shape}")

                n_tiles, n_hits, n_negs = 0, 0, 0
                
                for tile, tx, ty in tile_image(rgb):
                    n_tiles += 1
                    stem = f"{species}_{czi_path.stem}_x{tx:06d}_y{ty:06d}"
                    
                    detections = extract_general_pollen(tile, model, args.conf)
                    
                    # Render to Negatives gracefully
                    if len(detections) == 0:
                        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(neg_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        n_negs += 1
                        stats[species]["negatives"] += 1
                        continue
                    
                    # Valid Detections - Save to Positive Species Layout
                    n_hits += 1
                    stats[species]["images"] += 1
                    stats[species]["annotations"] += len(detections)
                    
                    # 1. Image
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(spec_img_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # 2. Labels mapped flawlessly to Species Class ID
                    lbl_lines = []
                    for d in detections:
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in d['poly_norm'])
                        lbl_lines.append(f"{class_id} {coords}")
                    
                    (spec_lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines))
                    
                    # 3. Vizualization (Tile-level Overlays & CV Curve tracking)
                    viz_bgr = tile_bgr.copy()
                    
                    for d in detections:
                        poly_px = d['poly_px'].reshape((-1, 1, 2))
                        
                        mask = np.zeros(viz_bgr.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [poly_px], 255)
                        
                        hsv = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2HSV)
                        S = hsv[:,:,1]
                        
                        S_blurred = cv2.GaussianBlur(S, (5, 5), 0)
                        _, binary = cv2.threshold(S_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        binary = cv2.bitwise_and(binary, binary, mask=mask)
                        
                        kernel = np.ones((5,5), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                        
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            epsilon = 0.003 * cv2.arcLength(largest_contour, True)
                            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                            cv2.polylines(viz_bgr, [smoothed_contour], True, (255, 0, 255), 2)
                        else:
                            cv2.polylines(viz_bgr, [poly_px], True, (255, 0, 255), 2)
                        
                        # Print explicitly over the tile
                        text_str = f"{d['conf']:.2f}"
                        px, py = d['poly_px'][0]
                        cv2.putText(viz_bgr, text_str, (int(px) - 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(viz_bgr, text_str, (int(px) - 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        
                    cv2.imwrite(str(spec_viz_dir / f"{stem}_viz.jpg"), viz_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
                print(f"   ✓ Extracted {n_tiles} tiles -> {n_hits} Overlays | {n_negs} Negatives")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Error parsing {czi_path.name}: {e}")
                print(f"  🔁 Attempting to load another candidate for {sp}...")
                
            # Clean up the large CZI file immediately to save disk space
            if czi_path.exists():
                os.remove(str(czi_path))
                print(f"   🧹 Removed temporary file {czi_path.name}")
                
            if success_count >= 5:
                break

    generate_stats_report(stats, out_dir, registry)
    print("\n🎉 Species Dataset Built Sucessfully!")


if __name__ == "__main__":
    main()
