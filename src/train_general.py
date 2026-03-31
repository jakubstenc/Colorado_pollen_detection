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
DATASET_ROOT = os.getenv("DATASET_ROOT", "/tmp/general_pollen_dataset")
MODEL_DIR = os.getenv("MODEL_DIR", "/home/meow/Documents/Antigravity/Colorado_pollen_detection/models/general_pollen")

def prep_dataset():
    print("🧹 Preparing Dataset for General Pollen Detection...")
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
        
    for split in ['train', 'val']:
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
                
    if not pairs:
        print("❌ No images found in staging areas!")
        return False
        
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.2)
    
    val_pairs = pairs[:split_idx]
    train_pairs = pairs[split_idx:]
    
    print(f"📦 Assembling dataset: {len(train_pairs)} Train | {len(val_pairs)} Val")
    
    for split_pairs, split_name in [(train_pairs, 'train'), (val_pairs, 'val')]:
        for img, lbl, is_pos in split_pairs:
            base = os.path.basename(img)
            name_only = os.path.splitext(base)[0]
            lbl_dest = os.path.join(DATASET_ROOT, split_name, 'labels', f"{name_only}.txt")
            
            shutil.copy(img, os.path.join(DATASET_ROOT, split_name, 'images', base))
            
            # For General Pollen, all positive classes map to 0 ("pollen").
            with open(lbl_dest, 'w') as f_out:
                if is_pos and lbl and os.path.exists(lbl):
                    with open(lbl, 'r') as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                parts[0] = "0" # Remap any species class to unified 0
                                f_out.write(" ".join(parts) + "\n")
                # If negative, we write nothing, creating an empty .txt file for Ultralytics

    # Write unified data.yaml
    data_cfg = {
        'path': os.path.abspath(DATASET_ROOT),
        'train': 'train/images',
        'val': 'val/images',
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
        patience=0,
        close_mosaic=20,
        batch=32, # Reduced to 32 to fit the Large Segmentation model in VRAM safely
        imgsz=640,
        project=MODEL_DIR,
        name=run_name,
        # Standard augmentations
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5
    )
    
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
