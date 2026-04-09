import os
import cv2
import numpy as np
from pathlib import Path

def redraw_visualizations():
    base_dir = Path.home() / "Desktop" / "Species_model"
    if not base_dir.exists():
        print("Desktop directory does not exist yet.")
        return
        
    for species_dir in base_dir.iterdir():
        if not species_dir.is_dir(): continue
        
        img_dir = species_dir / "Images"
        lbl_dir = species_dir / "Labels"
        viz_dir = species_dir / "Vizualization"
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        print(f"♻️ Calculating Geometric Ellipses for {species_dir.name}...")
        
        # Clear out Old Vizualizations natively
        if viz_dir.exists():
            for f in viz_dir.iterdir():
                try:
                    os.remove(f)
                except:
                    pass
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in img_dir.glob("*.jpg"):
            stem = img_file.stem
            lbl_file = lbl_dir / f"{stem}.txt"
            
            if not lbl_file.exists():
                continue
                
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            H, W = img.shape[:2]
            viz_bgr = img.copy()
            
            with open(lbl_file, 'r') as f:
                lines = f.read().strip().split('\n')
                
            for line in lines:
                if not line.strip(): continue
                parts = line.split()
                # poly coordinates normalized
                coords = [float(p) for p in parts[1:]]
                pts = []
                for i in range(0, len(coords), 2):
                    pts.append([int(coords[i]*W), int(coords[i+1]*H)])
                    
                poly_px = np.array(pts, np.int32).reshape((-1, 1, 2))
                
                # 1. Mask to YOLO's explicit control domain
                mask = np.zeros(viz_bgr.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly_px], 255)
                
                # 2. Extract biological dye exclusively using HSV Saturation
                hsv = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2HSV)
                S = hsv[:,:,1]
                
                # 3. Threshold Saturation using Otsu
                S_blurred = cv2.GaussianBlur(S, (5, 5), 0)
                _, binary = cv2.threshold(S_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 4. Strictly limit to the YOLO boundaries
                binary = cv2.bitwise_and(binary, binary, mask=mask)
                
                # 5. Fix internal biological gaps in dye using morphology
                kernel = np.ones((5,5), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # 6. Extract extreme contour bounds
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # 7. Apply highly aggressive vector smoothing to permanently dissolve harsh pixel stairs
                    epsilon = 0.003 * cv2.arcLength(largest_contour, True)
                    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    cv2.polylines(viz_bgr, [smoothed_contour], True, (255, 0, 255), 2)
                else:
                    cv2.polylines(viz_bgr, [poly_px], True, (255, 0, 255), 2)
                    
            cv2.imwrite(str(viz_dir / f"{stem}_viz.jpg"), viz_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
    print("✅ All Visualizations globally smoothed with absolute geometrical matrices!")

if __name__ == "__main__":
    redraw_visualizations()
