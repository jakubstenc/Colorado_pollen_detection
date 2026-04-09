#!/usr/bin/env python3
import os
import argparse
import csv
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO

import czi_ingest
try:
    from aicsimageio import AICSImage
except ImportError:
    AICSImage = None

def load_species_manifest(manifest_path):
    registry = {}
    if not os.path.exists(manifest_path):
        print(f"⚠️ Warning: Manifest '{manifest_path}' not found!")
        return registry
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            registry[int(row['class_id'].strip())] = row['species_code'].strip()
    return registry

def pseudo_label_two_stage(tile_bgr, model_seg, model_cls, registry, scale_um_px, conf=0.1):
    """Run generalized segmentation, crop out instances, and run species classifier."""
    results = model_seg(tile_bgr, verbose=False, conf=conf, retina_masks=True)
    detections = []
    
    if results[0].masks is None or results[0].boxes is None:
        return detections
        
    H, W = tile_bgr.shape[:2]
    
    for mask_xy, box in zip(results[0].masks.xy, results[0].boxes):
        if mask_xy.shape[0] < 3:
            continue
        c_conf = float(box.conf[0])
        if c_conf < conf:
            continue
            
        pts = np.array(mask_xy, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # Physical size filter
        area_um2 = None
        if scale_um_px is not None:
            area_px = cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32))
            area_um2 = area_px * (scale_um_px ** 2)
            if not (czi_ingest.MIN_POLLEN_AREA_UM2 < area_um2 < czi_ingest.MAX_POLLEN_AREA_UM2):
                continue
                
        # Stage 2: Crop and Classify
        pad = 0.1
        pad_x = int(w * pad)
        pad_y = int(h * pad)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        
        crop = tile_bgr[y1:y2, x1:x2]
        
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
            
        # Classify
        cls_results = model_cls(crop, verbose=False)
        top1_idx = int(cls_results[0].probs.top1)
        top1_conf = float(cls_results[0].probs.top1conf)
        
        pred_species = registry.get(top1_idx, "Unknown")
        
        # Normalize polygon back to 0-1 for saving labels explicitly
        norm_xy = mask_xy.copy().astype(float)
        norm_xy[:, 0] /= W
        norm_xy[:, 1] /= H
        
        detections.append({
            "species": pred_species,
            "class_id": top1_idx,
            "seg_conf": c_conf,
            "cls_conf": top1_conf,
            "area_um2": area_um2,
            "poly_px": pts,
            "poly_norm": norm_xy
        })
        
    return detections

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Inference Pipeline for Species Specific Model")
    parser.add_argument("--root", required=True, help="Path to input CZI files")
    parser.add_argument("--out", required=True, help="Output directory for UI framework and overview maps")
    parser.add_argument("--model-seg", required=True, help="Path to best.pt general pollen segmentation model")
    parser.add_argument("--model-cls", required=True, help="Path to best.pt species classification model")
    parser.add_argument("--manifest", default="/home/meow/Documents/Antigravity/Colorado_pollen_detection/src/species_manifest.csv")
    args = parser.parse_args()
    
    registry = load_species_manifest(args.manifest)
    print(f"📋 Loaded {len(registry)} species from manifest.")
    
    model_seg = YOLO(args.model_seg)
    model_cls = YOLO(args.model_cls)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_czis = sorted(Path(args.root).rglob("*.czi"))
    print(f"🔍 Found {len(all_czis)} .czi files to process.")
    
    all_summaries = []
    
    neg_dir = out_dir / "Negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    for czi_path in all_czis:
        print(f"\n🔬 Processing: {czi_path.name}")
        img = AICSImage(str(czi_path))
        ps = img.physical_pixel_sizes
        scale_um_px = float(ps.X) if ps.X else None
        
        rgb = czi_ingest.get_mip_rgb(img, channels=[0, 1, 2])
        
        # Build Overview Maps
        H_orig, W_orig = rgb.shape[:2]
        scale_factor = min(4000.0 / H_orig, 4000.0 / W_orig)
        new_W, new_H = int(W_orig * scale_factor), int(H_orig * scale_factor)
        overview_rgb = cv2.resize(rgb, (new_W, new_H))
        overview_bgr = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2BGR)
        overview_overlay = overview_bgr.copy()
        
        n_hits = 0
        
        for tile_rgb, tx, ty in czi_ingest.tile_image(rgb):
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
            stem = f"{czi_path.stem}_x{tx:06d}_y{ty:06d}"
            
            detections = pseudo_label_two_stage(tile_bgr, model_seg, model_cls, registry, scale_um_px)
            
            if not detections:
                cv2.imwrite(str(neg_dir / f"{stem}.jpg"), tile_bgr)
                continue
                
            n_hits += len(detections)
            viz_bgr = tile_bgr.copy()
            
            # Group by species to save in the Active Learning Format
            # Active Learning format expects: <Species>/Images/, <Species>/Labels/, <Species>/Vizualization/
            # If a tile has multiple species... this gets tricky.
            # We will just save the tile under the species of the MOST CONFIDENT detection for UI review.
            best_det = max(detections, key=lambda x: x['cls_conf'])
            primary_species = best_det['species']
            
            sp_dir = out_dir / primary_species
            (sp_dir / "Images").mkdir(parents=True, exist_ok=True)
            (sp_dir / "Labels").mkdir(parents=True, exist_ok=True)
            (sp_dir / "Vizualization").mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(sp_dir / "Images" / f"{stem}.jpg"), tile_bgr)
            
            lbl_lines = []
            for d in detections:
                # Add to text file (yolo format: class_id x1 y1 x2 y2 ... cls_conf)
                poly_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in d['poly_norm'])
                lbl_lines.append(f"{d['class_id']} {poly_str} {d['cls_conf']:.2f}")
                
                # Append to CSV
                all_summaries.append({
                    "CZI_File": czi_path.name,
                    "Tile": stem,
                    "Species": d['species'],
                    "Confidence": round(d['cls_conf'], 3),
                    "Area_um2": round(d['area_um2'], 2) if d['area_um2'] else None
                })
                
                # Draw on Tile Vizualization
                poly_px = d['poly_px'].reshape((-1, 1, 2))
                cv2.polylines(viz_bgr, [poly_px], True, (255, 0, 255), 2)
                text = f"{d['species']} {d['cls_conf']:.2f}"
                px, py = d['poly_px'][0]
                cv2.putText(viz_bgr, text, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw on Overview image
                pts_orig_px = d['poly_px'] + [tx, ty]
                pts_overview_px = (pts_orig_px * scale_factor).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overview_overlay, [pts_overview_px], (200, 0, 200))
                cv2.polylines(overview_overlay, [pts_overview_px], True, (255, 0, 255), 2)
                ox, oy = pts_overview_px[0][0]
                cv2.putText(overview_overlay, text, (int(ox), int(oy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overview_overlay, text, (int(ox), int(oy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            (sp_dir / "Labels" / f"{stem}.txt").write_text("\n".join(lbl_lines))
            cv2.imwrite(str(sp_dir / "Vizualization" / f"{stem}_viz.jpg"), viz_bgr)
            
        alpha = 0.4
        overview_blended = cv2.addWeighted(overview_overlay, alpha, overview_bgr, 1 - alpha, 0)
        overview_path = out_dir / f"overview_{czi_path.stem}_labeled.jpg"
        cv2.imwrite(str(overview_path), overview_blended, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"   ✅ Saved {n_hits} detections and overview map to {overview_path.name}")
        
    csv_path = out_dir / "summary_results.csv"
    if all_summaries:
        keys = all_summaries[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_summaries)
    else:
        csv_path.write_text("CZI_File,Tile,Species,Confidence,Area_um2\n")
        
    print(f"\n🎉 Inference Pipeline Complete! Results saved to {out_dir}")
    print(f"📄 Full execution tabular metrics logged to: {csv_path}")

if __name__ == "__main__":
    main()
