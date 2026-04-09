#!/usr/bin/env python3
import os
import cv2
import argparse
import random
from pathlib import Path

def extract_crops_from_labels(img_path, lbl_path, pad=0.1):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    
    with open(lbl_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    H, W = img.shape[:2]
    crops = []
    
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
            
        # Class ID is ignored from label since the folder structure (SpeciesCode) 
        # is the ground truth for this extraction.
        # Format is likely polygon: class_id x1 y1 x2 y2 ...
        if len(parts) == 5 or len(parts) == 6:
            # If bounding box format: class_id x_center y_center w h (and maybe conf)
            x_cen = float(parts[1]) * W
            y_cen = float(parts[2]) * H
            box_w = float(parts[3]) * W
            box_h = float(parts[4]) * H
            x_min = x_cen - box_w / 2
            x_max = x_cen + box_w / 2
            y_min = y_cen - box_h / 2
            y_max = y_cen + box_h / 2
        else:
            # Polygon format
            coords = [float(p) for p in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs) * W, max(xs) * W
            y_min, y_max = min(ys) * H, max(ys) * H

        box_w = x_max - x_min
        box_h = y_max - y_min
        
        pad_x = box_w * pad
        pad_y = box_h * pad
        
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(W, int(x_max + pad_x))
        y2 = min(H, int(y_max + pad_y))
        
        crop = img[y1:y2, x1:x2]
        
        if crop.shape[0] >= 16 and crop.shape[1] >= 16:
            crops.append(crop)
            
    return crops

def main():
    parser = argparse.ArgumentParser(description="Extract classification crops from Species_model/Training_data tiles")
    parser.add_argument("--src", required=True, help="Source directory (e.g., Species_model/Training_data)")
    parser.add_argument("--out", required=True, help="Output YOLO Classification dataset directory")
    parser.add_argument("--pad", type=float, default=0.20, help="Padding fraction around bounding box (default: 0.20)")
    parser.add_argument("--split", type=float, default=0.8, help="Train split fraction")
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist.")
        return
        
    species_folders = [d for d in src_dir.iterdir() if d.is_dir() and d.name != "Negatives" and d.name != "Stats"]
    
    print(f"🔍 Found {len(species_folders)} species to extract crops from.")
    
    for split in ["train", "val"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        
    total_train = 0
    total_val = 0
    
    for sf in species_folders:
        species_code = sf.name
        img_dir = sf / "Images"
        lbl_dir = sf / "Labels"
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        (out_dir / "train" / species_code).mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / species_code).mkdir(parents=True, exist_ok=True)
        
        images = sorted(list(img_dir.glob("*.jpg")))
        crops_extracted = 0
        
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
                
            crops = extract_crops_from_labels(img_path, lbl_path, args.pad)
            
            for idx, crop in enumerate(crops):
                # Random split per instance
                is_train = random.random() < args.split
                split = "train" if is_train else "val"
                
                crop_dst = out_dir / split / species_code / f"{img_path.stem}_crop_{idx:03d}.jpg"
                cv2.imwrite(str(crop_dst), crop)
                
                if is_train:
                    total_train += 1
                else:
                    total_val += 1
                crops_extracted += 1
                
        print(f"   🌸 {species_code}: Extracted {crops_extracted} valid classification crops.")
        
    print(f"\n✅ Finished extraction! Dataset structure ready for YOLOv8 Classification:")
    print(f"   Train crops: {total_train}")
    print(f"   Val crops:   {total_val}")
    print(f"   Path:        {out_dir}")

if __name__ == "__main__":
    main()
