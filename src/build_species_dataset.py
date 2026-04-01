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

# Global Species ID Registry for the multi-class Species Model
SPECIES_REGISTRY = {
    "Ran_ado": 0,
    "Ran_adu": 1,
    "Ran_niv": 2,
    "Pol_pro": 3,
    "Poa_pra": 4,
    "Cal_pal": 5,
    "Car_nig": 6,
    "Dio_pyr": 7,
    "Epi_ana": 8,
    "Fes_sup": 9,
    "Geu_rot": 10,
    "Pot_div": 11,
    "Pot_nivea": 12,
    "Ran_pyg": 13,
    "Sal_arc": 14,
    "Sal_gla": 15,
    "Sax_riv": 16,
    "Sib_pro": 17,
    "Tri_par": 18,
    "Vax_uli": 19,
    "Ver_alp": 20
}

def get_mip_rgb(img, channels=(0, 1, 2), grayscale=False):
    # Use explicitly selected channels if not single channel
    if img.shape[1] > max(channels):
        raw = img.get_image_data("ZYX", C=channels, T=0)
    else:
        raw = img.get_image_data("ZYX", C=0, T=0)

    # Handle Z projection
    if len(raw.shape) == 3:  # ZYX
        raw = np.max(raw, axis=0) # mip

    # Handle grayscale requested
    if grayscale or len(raw.shape) == 2:
        if len(raw.shape) == 3 and raw.shape[0] > 1:
            raw = np.mean(raw, axis=0).astype(raw.dtype)
        # Convert back to HWC RGB for OpenCV compatibility
        raw = np.stack([raw, raw, raw], axis=-1)
    elif len(raw.shape) == 3 and raw.shape[0] == 3: # CYX to YXC
        raw = np.moveaxis(raw, 0, -1)

    # Normalize to 8-bit
    raw = raw.astype(np.float32)
    min_val, max_val = np.percentile(raw, (1, 99))
    if max_val > min_val:
        raw = np.clip((raw - min_val) / (max_val - min_val) * 255.0, 0, 255)
    else:
        raw = np.zeros_like(raw)

    return raw.astype(np.uint8)

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

def generate_stats_report(stats, out_dir):
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
            class_id = SPECIES_REGISTRY.get(sp, "Unknown")
            f.write(f"| `{sp}` | `{class_id}` | {data['images']} | {data['annotations']} |\n")
            
    print(f"\n📄 Rendering Quarto Specs in {stats_dir}...")
    try:
        subprocess.run(["quarto", "render", str(qmd_path), "--to", "pdf"], check=True)
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
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out)
    
    print(f"🤖 Loading general pollen model: {args.model}")
    model = YOLO(args.model)
    
    neg_dir = out_dir / "Negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}

    czi_files = list(root_dir.glob("**/*.czi"))
    print(f"\n🔍 Found {len(czi_files)} CZIs to aggressively process...")
    
    for czi_path in czi_files:
        species = None
        for code in SPECIES_REGISTRY.keys():
            if code in czi_path.name:
                species = code
                break
                
        if not species:
            # Fallback if the filename does not elegantly contain the species code
            print(f"⚠️ Unknown species inside {czi_path.name}. Skipping!")
            continue
            
        class_id = SPECIES_REGISTRY[species]
        
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
                
                # 3. Vizualization (Tile-level Overlays)
                viz_bgr = tile_bgr.copy()
                overlay = viz_bgr.copy()
                
                for d in detections:
                    poly_px = d['poly_px'].reshape((-1, 1, 2))
                    # Purple fill with solid outline precisely around the boundary
                    cv2.fillPoly(overlay, [poly_px], (200, 0, 200))
                    cv2.polylines(overlay, [poly_px], True, (255, 0, 255), 2)
                    
                    # Print precisely over the tile
                    text_str = f"{d['conf']:.2f}"
                    px, py = poly_px[0][0]
                    cv2.putText(overlay, text_str, (int(px) - 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(overlay, text_str, (int(px) - 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                # 0.4 Alpha transparency masking
                viz_blended = cv2.addWeighted(overlay, 0.4, viz_bgr, 0.6, 0)
                cv2.imwrite(str(spec_viz_dir / f"{stem}_viz.jpg"), viz_blended, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
            print(f"   ✓ Extracted {n_tiles} tiles -> {n_hits} Overlays | {n_negs} Negatives")
            
        except Exception as e:
            print(f"  ❌ Error parsing {czi_path.name}: {e}")

    generate_stats_report(stats, out_dir)
    print("\n🎉 Species Dataset Built Sucessfully!")


if __name__ == "__main__":
    main()
