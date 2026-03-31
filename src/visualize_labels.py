#!/usr/bin/env python3
"""
visualize_labels.py — Overlay YOLO segmentation polygons on image tiles.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

# High-contrast neon colors for better visibility over red pollen
COLORS = [
    (0, 255, 255),   # Yellow
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Bright Green
    (255, 255, 255), # White
]
def draw_yolo_polygons(image, label_path):
    """Draw polygons from a YOLO format txt file onto an image."""
    if not os.path.exists(label_path):
        return image
    
    h, w = image.shape[:2]
    overlay = image.copy()
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        class_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        
        # Denormalize coordinates
        pts = (coords * [w, h]).astype(np.int32)
        
        color = COLORS[class_id % len(COLORS)]
        
        # Draw polygon
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=4)
        # Fill with transparency
        cv2.fillPoly(overlay, [pts], color=color)
    
    # Blend with original (60% fill color opacity)
    return cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory with images")
    parser.add_argument("--labels", required=True, help="Directory with labels")
    parser.add_argument("--out", required=True, help="Output directory for visualizations")
    parser.add_argument("--num", type=int, default=20, help="Number of samples to visualize")
    args = parser.parse_args()
    
    img_root = Path(args.images)
    lbl_root = Path(args.labels)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    
    images = list(img_root.glob("*.jpg"))
    if not images:
        print("❌ No images found in path.")
        return
        
    num_samples = min(args.num, len(images))
    samples = random.sample(images, num_samples)
    
    print(f"🎨 Visualizing {num_samples} samples to {args.out}...")
    
    for img_path in samples:
        lbl_path = lbl_root / f"{img_path.stem}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        viz = draw_yolo_polygons(img, lbl_path)
        
        out_path = out_root / f"viz_{img_path.name}"
        cv2.imwrite(str(out_path), viz)
        print(f"   ✅ Saved {out_path.name}", end='\r')

    print(f"\n✨ Visualization complete. Check {args.out}")

if __name__ == "__main__":
    main()
