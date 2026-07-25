import cv2
import glob
import os
import numpy as np

def analyze_color(viz_dir, img_dir):
    for viz_f in glob.glob(os.path.join(viz_dir, "*.jpg"))[:5]:
        base = os.path.basename(viz_f).replace("_viz.jpg", "")
        img_f = os.path.join(img_dir, base + ".jpg")
        lbl_f = img_f.replace("Images", "Labels").replace(".jpg", ".txt")
        if not os.path.exists(img_f) or not os.path.exists(lbl_f): continue
        
        img = cv2.imread(img_f)
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        print(f"File: {base}")
        with open(lbl_f, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 5:
                    coords = [float(p) for p in parts[1:]]
                    x = [int(c * w) for c in coords[0::2]]
                    y = [int(c * h) for c in coords[1::2]]
                    min_x, max_x = max(0, min(x)), min(w, max(x))
                    min_y, max_y = max(0, min(y)), min(h, max(y))
                    
                    crop = hsv[min_y:max_y, min_x:max_x]
                    if crop.size == 0: continue
                    
                    # Calculate mean saturation
                    mean_s = np.mean(crop[:, :, 1])
                    mean_v = np.mean(crop[:, :, 2])
                    print(f"  - Crop Saturation: {mean_s:.1f}, Brightness: {mean_v:.1f}")

analyze_color(
    os.path.expanduser('~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated/Deposition_Stigmas/Discarded/Vizualization/'),
    os.path.expanduser('~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated/Deposition_Stigmas/Discarded/Images/')
)
