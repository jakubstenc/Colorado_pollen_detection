import os
import glob
import shutil

STAGED_AREA_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated")
DEST_LBL_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Curated_Retrain_Data/labels")

# Find all recently reviewed images
reviewed_imgs = glob.glob(os.path.join(STAGED_AREA_DIR, "*", "Reviewed", "Images", "*.jpg"))

moved_count = 0
for img_path in reviewed_imgs:
    # Check if this image has a corresponding label in DEST_LBL_DIR
    base_stem = os.path.basename(img_path).replace(".jpg", "")
    dest_lbl = os.path.join(DEST_LBL_DIR, base_stem + ".txt")
    
    if not os.path.exists(dest_lbl):
        # This was an "artifact" because it didn't get its label copied to Curated_Retrain_Data
        # (or it didn't have a label to begin with, but usually artifacts don't copy labels)
        
        # Move it to Artifacts
        species_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
        artifact_dir = os.path.join(species_dir, "Artifacts")
        
        os.makedirs(os.path.join(artifact_dir, "Images"), exist_ok=True)
        os.makedirs(os.path.join(artifact_dir, "Labels"), exist_ok=True)
        os.makedirs(os.path.join(artifact_dir, "Vizualization"), exist_ok=True)
        
        # Move Image
        new_img_path = os.path.join(artifact_dir, "Images", os.path.basename(img_path))
        shutil.move(img_path, new_img_path)
        
        # Move Label
        old_lbl_path = img_path.replace("/Images/", "/Labels/").replace(".jpg", ".txt")
        if os.path.exists(old_lbl_path):
            shutil.move(old_lbl_path, os.path.join(artifact_dir, "Labels", os.path.basename(old_lbl_path)))
            
        # Move Vizualization
        old_viz_path = img_path.replace("/Images/", "/Vizualization/")
        if os.path.exists(old_viz_path):
            shutil.move(old_viz_path, os.path.join(artifact_dir, "Vizualization", os.path.basename(old_viz_path)))
            
        moved_count += 1

print(f"Migrated {moved_count} artifacts from Reviewed to Artifacts.")
