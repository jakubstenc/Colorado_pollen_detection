import sys
import os
sys.path.append('src')
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def extract_general_pollen(tile, model, conf_thresh):
    results = model(tile, verbose=False, retina_masks=True)
    detections = []
    if results[0].masks is None or results[0].boxes is None:
        return detections
    H, W = tile.shape[:2]
    for mask_xy, box in zip(results[0].masks.xy, results[0].boxes):
        if mask_xy.shape[0] < 3:
            continue
        c_conf = float(box.conf[0])
        if c_conf < conf_thresh:
            continue
        poly_px = []
        for point in mask_xy:
            x, y = point
            poly_px.append((x, y))
        detections.append({
            'poly_px': np.array(poly_px, dtype=np.int32),
            'conf': c_conf
        })
    return detections

general_model = YOLO("models/general_pollen/latest.pt")
# Take the patch I extracted earlier!
rgb = cv2.imread("patch_raw.jpg")

detections = extract_general_pollen(rgb, general_model, conf_thresh=0.1)

overview_img = rgb.copy()
for d in detections:
    poly_px = d['poly_px']
    cv2.polylines(overview_img, [poly_px.reshape((-1, 1, 2))], True, (0, 0, 255), 2)

print(f"Total general pollen detected in patch: {len(detections)}")
cv2.imwrite("patch_general_only.jpg", overview_img)
