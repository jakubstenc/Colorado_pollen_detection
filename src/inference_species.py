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

def extract_qr_code(czi_path):
    try:
        from aicsimageio import AICSImage
        import cv2
        import numpy as np
        img = AICSImage(str(czi_path))
        detector = cv2.QRCodeDetector()
        for scene in reversed(img.scenes):
            img.set_scene(scene)
            try:
                if 'S' in img.dims.order:
                    arr = img.get_image_data("YXS", T=0, Z=0, C=0)
                    if arr.max() <= 1.0: arr = (arr * 255).astype(np.uint8)
                    else: arr = arr.astype(np.uint8)
                    if len(arr.shape) == 3 and arr.shape[-1] == 3:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                else:
                    arr = img.get_image_data("YX", T=0, Z=0, C=0)
                    if arr.max() <= 1.0: arr = (arr * 255).astype(np.uint8)
                    else: arr = arr.astype(np.uint8)
                if arr.shape[0] < 50 or arr.shape[1] < 50:
                    continue
                data, bbox, _ = detector.detectAndDecode(arr)
                if data:
                    return data.strip()
            except Exception:
                continue
    except Exception:
        pass
    return "Unknown_Sample"

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
                
        # Calculate extended shape geometry
        M = cv2.moments(pts.reshape(-1, 1, 2).astype(np.float32))
        cx_tile = int(M['m10']/M['m00']) if M['m00'] != 0 else int(x + w/2)
        cy_tile = int(M['m01']/M['m00']) if M['m00'] != 0 else int(y + h/2)
        perimeter_px = float(cv2.arcLength(pts.reshape(-1, 1, 2).astype(np.float32), True))
                
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
            "poly_norm": norm_xy,
            "cx_tile": cx_tile,
            "cy_tile": cy_tile,
            "w_px": w,
            "h_px": h,
            "perimeter_px": perimeter_px
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
        
        try:
            rel_path = czi_path.parent.relative_to(Path(args.root))
        except ValueError:
            rel_path = Path("")
            
        current_out_dir = out_dir / rel_path
        current_out_dir.mkdir(parents=True, exist_ok=True)
        
        sample_id = extract_qr_code(czi_path)
        print(f"   🏷️ Sample ID (QR): {sample_id} | Relative Path: {rel_path}")
        
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
        image_summaries = []
        
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
                global_cx = tx + d['cx_tile']
                global_cy = ty + d['cy_tile']
                
                row = {
                    "Sample_ID": sample_id,
                    "CZI_File": czi_path.name,
                    "Species": d['species'],
                    "Area_um2": round(d['area_um2'], 2) if d['area_um2'] else None,
                    "Centroid_X": float(global_cx),
                    "Centroid_Y": float(global_cy),
                    "Width_px": d['w_px'],
                    "Height_px": d['h_px'],
                    "Perimeter_px": round(d['perimeter_px'], 2),
                    "Confidence": round(d['cls_conf'], 3),
                    "Tile": stem
                }
                all_summaries.append(row)
                image_summaries.append(row)
                
                # Draw on Tile Vizualization
                poly_px = d['poly_px'].reshape((-1, 1, 2))
                cv2.polylines(viz_bgr, [poly_px], True, (255, 0, 255), 2)
                text = f"{d['species']} {d['cls_conf']:.2f}"
                px, py = d['poly_px'][0]
                cv2.putText(viz_bgr, text, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw on Overview image - thick red circles for microscopic visibility
                ocx, ocy = int(global_cx * scale_factor), int(global_cy * scale_factor)
                cv2.circle(overview_overlay, (ocx, ocy), 15, (0, 0, 255), -1)
                cv2.circle(overview_overlay, (ocx, ocy), 17, (0, 255, 255), 2)
                cv2.putText(overview_overlay, text, (ocx + 20, ocy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overview_overlay, text, (ocx + 20, ocy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            (sp_dir / "Labels" / f"{stem}.txt").write_text("\n".join(lbl_lines))
            cv2.imwrite(str(sp_dir / "Vizualization" / f"{stem}_viz.jpg"), viz_bgr)
            
        alpha = 0.4
        overview_blended = cv2.addWeighted(overview_overlay, alpha, overview_bgr, 1 - alpha, 0)
        overview_path = current_out_dir / f"overview_{czi_path.stem}_labeled.jpg"
        cv2.imwrite(str(overview_path), overview_blended, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Save individual per-image detailed stat table
        if image_summaries:
            csv_path_img = current_out_dir / f"{czi_path.stem}_details.csv"
            keys = image_summaries[0].keys()
            with open(csv_path_img, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(image_summaries)
        
        print(f"   ✅ Saved {n_hits} detections and overview map to {overview_path.name}")
        
        # AGGRESSIVE MEMORY CLEANUP FOR K8S OOM PREVENTION
        if 'img' in locals() and hasattr(img, 'close'):
            try: img.close()
            except Exception: pass
        if 'rgb' in locals(): del rgb
        if 'overview_overlay' in locals(): del overview_overlay
        if 'overview_blended' in locals(): del overview_blended
        import gc
        gc.collect()

        # DYNAMIC UPLOAD: Upload per-file immediately to bypass S3 upload drops if OOM kills the ephemeral wrapper
        s3_bucket = os.environ.get('S3_BUCKET', 'bucket')
        s3_endpoint = os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz')
        out_target = f"s3://{s3_bucket}/PEG/Colorado/Species_model/Trainig_data/"
        os.system(f"aws --endpoint-url {s3_endpoint} s3 sync /app/Inference_Results/ {out_target} --no-verify-ssl > /dev/null 2>&1")
        
    csv_path = out_dir / "master_results_summary.csv"
    if all_summaries:
        keys = all_summaries[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_summaries)
    else:
        csv_path.write_text("Sample_ID,CZI_File,Species,Area_um2,Centroid_X,Centroid_Y,Width_px,Height_px,Perimeter_px,Confidence,Tile\n")
        
    print(f"\n🎉 Inference Pipeline Complete! Results saved to {out_dir}")
    print(f"📄 Full execution tabular metrics logged to: {csv_path}")

if __name__ == "__main__":
    main()
