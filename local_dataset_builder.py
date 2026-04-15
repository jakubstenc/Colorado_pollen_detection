import sys
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import random
import csv

sys.path.append("/home/meow/Documents/Antigravity/Colorado_pollen_detection/src")
from build_species_dataset import tile_image, extract_general_pollen, get_mip_rgb, AICSImage, generate_stats_report

out_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Species_model/Trainig_data")
model_path = "/home/meow/cesnet_cloud/bucket/PEG/Colorado/trained_models/general_pollen/latest.pt"
src_root = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source")
manifest_path = "/home/meow/Documents/Antigravity/Colorado_pollen_detection/src/species_manifest.csv"
conf_thresh = 0.65

print("🤖 Loading YOLO model...")
model = YOLO(model_path)

print("📋 Loading species manifest...")
registry = {}
with open(manifest_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        code = row['species_code'].strip()
        cid = int(row['class_id'].strip())
        registry[code] = cid
print(f"Loaded {len(registry)} species definitions.")

neg_dir = out_dir / "Negatives"
neg_dir.mkdir(parents=True, exist_ok=True)
stats = {}

print("🔍 Searching for all CZIs in mounted S3 bucket...")
all_czis = list(src_root.rglob("*.czi"))
print(f"Found {len(all_czis)} total CZI files.")

czi_by_species = {}
for code in registry.keys():
    matched = [f for f in all_czis if code.lower() in f.name.lower()]
    if matched:
         # filter corrupted 20260227
        valid_keys = [f for f in matched if '20260227' not in f.name]
        if not valid_keys:
            valid_keys = matched
        czi_by_species[code] = valid_keys

print(f"⚙️ Will process {len(czi_by_species)} species.")

for species, keys_for_species in sorted(czi_by_species.items()):
    class_id = registry[species]
    stats[species] = {"images": 0, "annotations": 0, "negatives": 0}
    
    spec_img_dir = out_dir / species / "Images"
    spec_lbl_dir = out_dir / species / "Labels"
    spec_viz_dir = out_dir / species / "Vizualization"
    spec_img_dir.mkdir(parents=True, exist_ok=True)
    spec_lbl_dir.mkdir(parents=True, exist_ok=True)
    spec_viz_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(4242)
    random.shuffle(keys_for_species)
    
    success_count = 0
    t_tiles = 0
    t_hits = 0
    t_negs = 0

    print(f"\n🌸 Processing {species} [Class {class_id}]")
    for czi_path in keys_for_species:
        if success_count >= 5:
            break
            
        print(f"📥 Loading {czi_path.name}")
        tmp_path = Path("/tmp") / czi_path.name
        try:
            if not tmp_path.exists():
                shutil.copy(str(czi_path), str(tmp_path))
                
            img = AICSImage(str(tmp_path))
            rgb = get_mip_rgb(img)
            
            if len(rgb.shape) < 3 or rgb.shape[0] < 640 or rgb.shape[1] < 640:
                raise Exception(f"Invalid RGB Extracted: {rgb.shape}")
                
            n_tiles, n_hits, n_negs = 0, 0, 0
            for tile, tx, ty in tile_image(rgb):
                n_tiles += 1
                stem = f"{species}_{tmp_path.stem}_x{tx:06d}_y{ty:06d}"
                detections = extract_general_pollen(tile, model, conf_thresh)
                
                if len(detections) == 0:
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(neg_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_negs += 1
                    stats[species]["negatives"] += 1
                else:
                    n_hits += 1
                    stats[species]["images"] += 1
                    stats[species]["annotations"] += len(detections)
                    
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(spec_img_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    lbl_lines = []
                    H, W = tile_bgr.shape[:2]
                    for d in detections:
                        poly_px = d['poly_px'].reshape((-1, 1, 2))
                        mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly_px], 255)
                        
                        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
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
                            final_poly_px = smoothed_contour.reshape(-1, 2)
                        else:
                            final_poly_px = d['poly_px']
                            
                        # Save snapped polygon to ground-truth label
                        norm_xy = final_poly_px.astype(float)
                        norm_xy[:, 0] /= W
                        norm_xy[:, 1] /= H
                        
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_xy)
                        lbl_lines.append(f"{class_id} {coords}")
                        
                        # Store snapped variant strictly back for viz rendering
                        d['poly_px'] = final_poly_px
                        
                    (spec_lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines))
                    
                    viz_bgr = tile_bgr.copy()
                    for d in detections:
                        poly_px = d['poly_px'].reshape((-1, 1, 2))
                        cv2.polylines(viz_bgr, [poly_px], True, (255, 0, 255), 2)
                        
                        text_str = f"{d['conf']:.2f}"
                        px, py = poly_px[0][0]
                        cv2.putText(viz_bgr, text_str, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(viz_bgr, text_str, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                        
                    cv2.imwrite(str(spec_viz_dir / f"{stem}_viz.jpg"), viz_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
            print(f"   ✓ {n_tiles} tiles -> {n_hits} Overlays | {n_negs} Negatives")
            t_tiles += n_tiles
            t_hits += n_hits
            t_negs += n_negs
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error parsing {czi_path.name}: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    print(f"   => TOTAL for {species}: {t_tiles} tiles -> {t_hits} Overlays | {t_negs} Negatives")

print("\nGenerating final stats report...")
generate_stats_report(stats, out_dir, registry)
print("✅ ALL SETUP COMPLETED")
