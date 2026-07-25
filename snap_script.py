import os
import cv2
import numpy as np
import glob
import shutil

def run_snap():
    src_dir = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Curated_Retrain_Data")
    src_img = os.path.join(src_dir, "images")
    src_lbl = os.path.join(src_dir, "labels")
    
    out_dir = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Roboflow_Export")
    out_img = os.path.join(out_dir, "images")
    out_lbl = os.path.join(out_dir, "labels")
    
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    lbl_files = glob.glob(os.path.join(src_lbl, "*.txt"))
    
    for lbl_f in lbl_files:
        base = os.path.basename(lbl_f)
        img_f = os.path.join(src_img, base.replace(".txt", ".jpg"))
        if not os.path.exists(img_f):
            continue
            
        out_lbl_f = os.path.join(out_lbl, base)
        out_img_f = os.path.join(out_img, base.replace(".txt", ".jpg"))
        
        # Copy image
        shutil.copy(img_f, out_img_f)
        
        # Process label
        with open(lbl_f, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            # Empty negative file, just copy empty
            open(out_lbl_f, 'w').close()
            continue
            
        img = cv2.imread(img_f)
        if img is None: continue
        h, w = img.shape[:2]
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:
                class_id = parts[0]
                coords = [float(p) for p in parts[1:]]
                x = [int(c * w) for c in coords[0::2]]
                y = [int(c * h) for c in coords[1::2]]
                
                pad = 20
                min_x = max(0, min(x) - pad)
                max_x = min(w, max(x) + pad)
                min_y = max(0, min(y) - pad)
                max_y = min(h, max(y) + pad)
                
                crop = img[min_y:max_y, min_x:max_x]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if len(largest) >= 3:
                            # Normalize points back to 0-1
                            norm_pts = []
                            for pt in largest:
                                px = (pt[0][0] + min_x) / w
                                py = (pt[0][1] + min_y) / h
                                norm_pts.append(f"{px:.6f} {py:.6f}")
                            
                            new_lines.append(f"{class_id} " + " ".join(norm_pts) + "\n")
                            continue
            # Fallback to original
            new_lines.append(line)
            
        with open(out_lbl_f, 'w') as f:
            f.writelines(new_lines)
            
    print(f"Exported {len(lbl_files)} files to Roboflow_Export")

if __name__ == "__main__":
    run_snap()
