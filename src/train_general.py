import os
import shutil
import random
import glob
from ultralytics import YOLO
import yaml
from pathlib import Path
from datetime import datetime

# Define Config Paths (with ENV fallbacks for containerized runs)
STAGED_AREA = os.getenv("STAGED_AREA", "/home/meow/cesnet_cloud/bucket/PEG/Colorado/Staged_area/")
STAGED_NEGATIVES = os.getenv("STAGED_NEGATIVES", "/home/meow/cesnet_cloud/bucket/PEG/Colorado/Staged_negatives")
STAGED_CURATED = os.getenv("STAGED_CURATED", "/home/meow/cesnet_cloud/bucket/PEG/Colorado/Curated_Retrain_Data")
DATASET_ROOT = os.getenv("DATASET_ROOT", "/tmp/general_pollen_dataset")
MODEL_DIR = os.getenv("MODEL_DIR", "/home/meow/Documents/Antigravity/Colorado_pollen_detection/models/general_pollen")

def prep_dataset():
    print("🧹 Preparing Dataset for General Pollen Detection...")
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
        
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DATASET_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, split, 'labels'), exist_ok=True)
        
    pairs = []
    
    # 1. Gather Positives (Roboflow outputs)
    if os.path.exists(STAGED_AREA):
        for z in glob.glob(os.path.join(STAGED_AREA, "*.zip")):
            # If they are zips, we extract to temp first
            temp_ext = os.path.join(DATASET_ROOT, "temp_pos")
            shutil.unpack_archive(z, temp_ext)
            for img in glob.glob(os.path.join(temp_ext, "**", "*.jpg"), recursive=True):
                lbl = img.replace("images", "labels").replace(".jpg", ".txt")
                if os.path.exists(lbl):
                    pairs.append((img, lbl, True)) # True = is positive
        # Also grab unzipped files directly if any
        for img in glob.glob(os.path.join(STAGED_AREA, "**", "*.jpg"), recursive=True):
            lbl = img.replace("images", "labels").replace(".jpg", ".txt")
            if os.path.exists(lbl):
                pairs.append((img, lbl, True))
            
    # 2. Gather Negatives
    if os.path.exists(STAGED_NEGATIVES):
        for img in glob.glob(os.path.join(STAGED_NEGATIVES, "**", "*.jpg"), recursive=True):
            pairs.append((img, None, False))
            
    # 3. Gather Human-Curated Positives and Explicit Hard Negatives
    if os.path.exists(STAGED_CURATED):
        print("🤝 Injecting Human-Curated Intelligence...")
        curated_imgs = glob.glob(os.path.join(STAGED_CURATED, "images", "*.jpg"))
        for img in curated_imgs:
            lbl = img.replace("images", "labels").replace(".jpg", ".txt")
            if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                pairs.append((img, lbl, True)) # Explicit Curated Positive
            else:
                pairs.append((img, None, False)) # Explicit Curated Hard Negative Dirt
                
    if not pairs:
        print("❌ No images found in staging areas!")
        return False
        
    random.shuffle(pairs)
    total = len(pairs)
    test_idx = int(total * 0.1)
    val_idx = int(total * 0.2)
    
    test_pairs = pairs[:test_idx]
    val_pairs = pairs[test_idx:val_idx]
    train_pairs = pairs[val_idx:]
    
    print(f"📦 Assembling dataset: {len(train_pairs)} Train | {len(val_pairs)} Val | {len(test_pairs)} Test")
    
    for split_pairs, split_name in [(train_pairs, 'train'), (val_pairs, 'val'), (test_pairs, 'test')]:
        for img, lbl, is_pos in split_pairs:
            base = os.path.basename(img)
            name_only = os.path.splitext(base)[0]
            lbl_dest = os.path.join(DATASET_ROOT, split_name, 'labels', f"{name_only}.txt")
            
            shutil.copy(img, os.path.join(DATASET_ROOT, split_name, 'images', base))
            
            # For General Pollen, all positive classes map to 0 ("pollen").
            with open(lbl_dest, 'w') as f_out:
                if is_pos and lbl and os.path.exists(lbl):
                    import cv2
                    import numpy as np
                    
                    tile_bgr = cv2.imread(img)
                    H, W = tile_bgr.shape[:2]
                    
                    with open(lbl, 'r') as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                coords = [float(p) for p in parts[1:]]
                                xs = coords[0::2]
                                ys = coords[1::2]
                                norm_xy = np.array(list(zip(xs, ys)), dtype=float)
                                
                                poly_px = np.zeros_like(norm_xy)
                                poly_px[:, 0] = norm_xy[:, 0] * W
                                poly_px[:, 1] = norm_xy[:, 1] * H
                                poly_px = poly_px.astype(np.int32).reshape((-1, 1, 2))
                                
                                mask = np.zeros((H, W), dtype=np.uint8)
                                cv2.fillPoly(mask, [poly_px], 255)
                                
                                hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
                                S = hsv[:,:,1]
                                S_blurred = cv2.GaussianBlur(S, (5, 5), 0)
                                _, binary = cv2.threshold(S_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                binary = cv2.bitwise_and(binary, binary, mask=mask)
                                kernel = np.ones((5,5), np.uint8)
                                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if contours:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    epsilon = 0.003 * cv2.arcLength(largest_contour, True)
                                    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                                    final_poly_px = smoothed_contour.reshape(-1, 2)
                                else:
                                    final_poly_px = poly_px.reshape(-1, 2)
                                    
                                final_norm_xy = final_poly_px.astype(float)
                                final_norm_xy[:, 0] /= W
                                final_norm_xy[:, 1] /= H
                                
                                final_coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in final_norm_xy)
                                f_out.write(f"0 {final_coords}\n")
                # If negative, we write nothing, creating an empty .txt file for Ultralytics

    # Write unified data.yaml
    data_cfg = {
        'path': os.path.abspath(DATASET_ROOT),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {0: 'pollen'}
    }
    with open(os.path.join(DATASET_ROOT, 'data.yaml'), 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)
        
    return True

def train_general():
    if not prep_dataset():
        return
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    run_name = f"general_pollen_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"🚀 Training {run_name} on General Pollen (Segmentation)...")
    # Upgrade to Segmentation Model (Large)
    model = YOLO("yolov8l-seg.pt")
    
    results = model.train(
        data=os.path.join(DATASET_ROOT, 'data.yaml'),
        epochs=500,      
        patience=200,    
        close_mosaic=30,
        batch=32, 
        imgsz=640,
        project=MODEL_DIR,
        name=run_name,
        # Color Jittering
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Spatial Transformations
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        # Advanced Blending
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.4   # Simulates pollen landing tightly on messy stigma backgrounds
    )
    
    print("🔬 Running Final Evaluation strictly on the isolated Test set...")
    metrics = model.val(data=os.path.join(DATASET_ROOT, 'data.yaml'), split='test', project=MODEL_DIR, name=f"{run_name}_test_eval")
    
    actual_save_dir = results.save_dir
    best_weights = os.path.join(actual_save_dir, "weights", "best.pt")
    
    # Copy to latest.pt for S3 compatibility (S3 does not support symlinks)
    latest_copy = os.path.join(MODEL_DIR, "latest.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, latest_copy)
        print(f"✅ Training Complete. Model copied to {latest_copy}")
    else:
        print("⚠️ Training completed but could not find best.pt!")

if __name__ == "__main__":
    train_general()
