from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os

model = YOLO('/home/meow/Documents/Antigravity/Colorado_pollen_detection/models/general_pollen/latest.pt')

# find an image from the newly pulled zip
images = glob.glob('sample_tiles/single_detection_results/train/images/*.jpg')
if not images:
    images = glob.glob('sample_tiles/single_detection_results/val/images/*.jpg')

if images:
    print(f"Running inference on {images[0]}")
    img = cv2.imread(images[0])
    results = model(img, conf=0.1, retina_masks=True)
    if results[0].masks is not None:
        for mask_xy in results[0].masks.xy:
            pts = np.array(mask_xy, dtype=np.int32).reshape(-1, 1, 2)
            area_px = cv2.contourArea(pts)
            scale_um_px = 0.877
            area_um2 = area_px * (scale_um_px**2)
            print(f"Detected object area_px: {area_px}, area_um2: {area_um2}")
    else:
        print("No masks found even at conf 0.1")
else:
    print("No images found to test.")
