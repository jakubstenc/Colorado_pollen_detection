import os
import glob
import shutil
import random
from pathlib import Path

def setup_v2_dataset():
    """
    Merges the original general dataset with the new highly-curated Active Learning inferences.
    Automatically handles 80/10/10 train/val/test splits and generates a YOLO data.yaml.
    """
    v1_dir = Path("dataset_general_v1")
    retrain_dir = Path(os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Curated_Retrain_Data"))
    staged_neg_dir = Path(os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Staged_negatives"))
    v2_dir = Path("dataset_general_v2")
    
    if not v1_dir.exists():
        print(f"⚠️ Initial dataset {v1_dir} not found locally! Please sync it from S3 if missing.")
        return
        
    print("🧹 Cleaning old v2 staging area...")
    if v2_dir.exists(): shutil.rmtree(v2_dir)
    os.makedirs(v2_dir, exist_ok=True)
    
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(v2_dir / s / "images", exist_ok=True)
        os.makedirs(v2_dir / s / "labels", exist_ok=True)
        
    print("📥 Copying original V1 dataset...")
    # Just copy the existing partitioned data exactly as-is into v2 to preserve validation baselines
    for s in splits:
        for f in glob.glob(str(v1_dir / s / "images" / "*.jpg")):
            shutil.copy(f, v2_dir / s / "images" / os.path.basename(f))
        for f in glob.glob(str(v1_dir / s / "labels" / "*.txt")):
            shutil.copy(f, v2_dir / s / "labels" / os.path.basename(f))
            
    print("🔍 Harvesting Active Learning Curation & Manual Negatives...")
    new_images = glob.glob(str(retrain_dir / "images" / "*.jpg"))
    neg_images = glob.glob(str(staged_neg_dir / "*.jpg")) + glob.glob(str(staged_neg_dir / "*.jpeg")) + glob.glob(str(staged_neg_dir / "*.png"))
    
    all_new_images = new_images + neg_images
    
    if not all_new_images:
        print("💡 No new images found in Curated_Retrain_Data or Staged_negatives. Waiting for you to review some images using the UI!")
        return
        
    # Split the NEW images 80/10/10
    random.seed(42)
    random.shuffle(all_new_images)
    
    n = len(all_new_images)
    train_split = int(n * 0.8)
    val_split = int(n * 0.9)
    
    train_imgs = all_new_images[:train_split]
    val_imgs = all_new_images[train_split:val_split]
    test_imgs = all_new_images[val_split:]
    
    def transfer_split(imgs, split_name):
        for img_path in imgs:
            img_path = Path(img_path)
            base = img_path.name
            stem = img_path.stem
            
            shutil.copy(str(img_path), v2_dir / split_name / "images" / base)
            
            dest_lbl = v2_dir / split_name / "labels" / f"{stem}.txt"
            if str(retrain_dir) in str(img_path):
                lbl_path = retrain_dir / "labels" / f"{stem}.txt"
                if lbl_path.exists():
                    shutil.copy(str(lbl_path), dest_lbl)
                else:
                    dest_lbl.touch()
            else:
                # It's a manual negative background image
                dest_lbl.touch()
                
    transfer_split(train_imgs, "train")
    transfer_split(val_imgs, "val")
    transfer_split(test_imgs, "test")
    
    print(f"📦 Successfully infused {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test active/manual samples!")

    # Generate data.yaml
    yaml_content = f"""path: dataset_general_v2
train: train/images
val: val/images
test: test/images

nc: 1
names: ['pollen']
"""
    with open(v2_dir / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    print("✅ Generated dataset_general_v2/data.yaml")

if __name__ == "__main__":
    setup_v2_dataset()
