import cv2
import numpy as np
import glob
import os

viz_dir = os.path.expanduser('~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated/Deposition_Stigmas/Vizualization/')
img_dir = os.path.expanduser('~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated/Deposition_Stigmas/Images/')

for viz_f in glob.glob(os.path.join(viz_dir, "*.jpg"))[:3]:
    base = os.path.basename(viz_f).replace("_viz.jpg", "")
    img_f = os.path.join(img_dir, base + ".jpg")
    lbl_f = img_f.replace("Images", "Labels").replace(".jpg", ".txt")
    if not os.path.exists(img_f) or not os.path.exists(lbl_f): continue
    
    img = cv2.imread(img_f)
    if img is None: continue
    h, w = img.shape[:2]
    
    output = img.copy()
    
    with open(lbl_f, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 5:
                coords = [float(p) for p in parts[1:]]
                x = [int(c * w) for c in coords[0::2]]
                y = [int(c * h) for c in coords[1::2]]
                
                # Get bounding box with padding
                pad = 15
                min_x = max(0, min(x) - pad)
                max_x = min(w, max(x) + pad)
                min_y = max(0, min(y) - pad)
                max_y = min(h, max(y) + pad)
                
                crop = img[min_y:max_y, min_x:max_x]
                if crop.size == 0: continue
                
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # Apply Gaussian Blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get largest contour
                    largest = max(contours, key=cv2.contourArea)
                    
                    # Offset back to original image
                    largest = largest + np.array([[min_x, min_y]])
                    
                    # Draw original blocky polygon in Blue
                    orig_pts = np.array(list(zip(x, y)), np.int32)
                    cv2.polylines(output, [orig_pts], True, (255, 0, 0), 1)
                    
                    # Draw new snapped polygon in Green
                    cv2.polylines(output, [largest], True, (0, 255, 0), 1)

    out_name = f"snap_test_{base}.jpg"
    cv2.imwrite(out_name, output)
    print(f"Saved {out_name}")

