import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from aicsimageio import AICSImage
import torch

from src.build_species_dataset import get_mip_rgb, tile_image, extract_general_pollen

def main():
    print("Loading model...")
    general_model = YOLO("best.pt")
    
    img_path = "test.czi"
    if not os.path.exists(img_path):
        print("test.czi not found!")
        return

    print("Loading image...")
    img = AICSImage(img_path)
    rgb = get_mip_rgb(img)
    
    px_size = getattr(img.physical_pixel_sizes, 'X', 1.0)
    if px_size is None: px_size = 1.0
        
    print(f"Physical Pixel Size: {px_size} um/pixel")
    
    max_dim = 4000
    h_full, w_full = rgb.shape[:2]
    overview_scale = 1.0
    if max(h_full, w_full) > max_dim:
        overview_scale = max_dim / max(h_full, w_full)
        stride = int(1.0 / overview_scale)
        overview_img = rgb[::stride, ::stride].copy()
        overview_scale = 1.0 / stride
    else:
        overview_img = rgb.copy()
        
    # Draw scale bar
    bar_length_um = 100
    bar_length_px = int(bar_length_um / px_size * overview_scale)
    cv2.rectangle(overview_img, (50, 50), (50 + bar_length_px, 70), (255, 255, 255), -1)
    cv2.putText(overview_img, f"{bar_length_um} um", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    print("Processing tiles...")
    # LOWER CONF THRESH TO 0.05 TO SEE IF MODEL EVEN SEES IT
    conf_thresh = 0.05
    for tile, tx, ty in tile_image(rgb, size=640, overlap=0.40):
        detections = extract_general_pollen(tile, general_model, conf_thresh=conf_thresh)
        
        if len(detections) > 0:
            for d in detections:
                poly_px = d['poly_px']
                conf = d['conf']
                area_px = cv2.contourArea(poly_px)
                area_um2 = area_px * (px_size ** 2)
                
                # NO AREA FILTER
                
                # Draw on Overview
                global_poly = poly_px.copy()
                global_poly[:, 0] += tx
                global_poly[:, 1] += ty
                overview_poly = (global_poly * overview_scale).astype(int)
                
                color = (0, 255, 255) # Yellow for general detections
                cv2.polylines(overview_img, [overview_poly.reshape((-1, 1, 2))], True, color, 2)
                
                # Add text to overview (Confidence & Area)
                px, py = overview_poly[0]
                text = f"{conf:.2f} ({area_um2:.0f}um)"
                cv2.putText(overview_img, text, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(overview_img, text, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                
    out_path = f"overview_general_test.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(overview_img, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
