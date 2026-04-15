import cv2
import numpy as np
from pathlib import Path

train_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Species_model/Trainig_data/Cal_chi")
img_dir = train_dir / "Images"
lbl_dir = train_dir / "Labels"
viz_dir = train_dir / "Vizualization"

viz_dir.mkdir(parents=True, exist_ok=True)

success = 0
for lbl_file in lbl_dir.glob("*.txt"):
    stem = lbl_file.stem
    img_file = img_dir / f"{stem}.jpg"
    viz_file = viz_dir / f"{stem}_viz.jpg"
    
    if not img_file.exists():
        continue
        
    img = cv2.imread(str(img_file))
    if img is None:
        continue
        
    H, W = img.shape[:2]
    with open(lbl_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.split()
        if len(parts) > 5:
            # Polygon
            coords = np.array([float(x) for x in parts[1:]])
            xs = coords[0::2] * W
            ys = coords[1::2] * H
            pts = np.vstack((xs, ys)).astype(np.int32).T
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = hsv[:,:,1]
            S_blurred = cv2.GaussianBlur(S, (5, 5), 0)
            _, binary = cv2.threshold(S_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.bitwise_and(binary, binary, mask=mask)
            
            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                epsilon = 0.003 * cv2.arcLength(largest, True)
                smoothed = cv2.approxPolyDP(largest, epsilon, True)
                cv2.polylines(img, [smoothed], True, (255, 0, 255), 2)
            else:
                cv2.polylines(img, [pts], True, (255, 0, 255), 2)
            
    cv2.imwrite(str(viz_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    success += 1

print(f"Generated {success} visualizations for UI.")
