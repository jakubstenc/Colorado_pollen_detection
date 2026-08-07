import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

def main():
    print("⛏️ Mining artifacts from Hard Negatives...")
    
    # Paths
    base_dir = os.path.expanduser("~/cesnet_data/PEG/Colorado")
    labels_dir = os.path.join(base_dir, "Curated_Retrain_Data", "labels")
    images_dir = os.path.join(base_dir, "Curated_Retrain_Data", "images")
    artifact_out_dir = os.path.join(base_dir, "dataset_species_curated", "train", "Artifact")
    
    os.makedirs(artifact_out_dir, exist_ok=True)
    
    # Find the latest general pollen model
    models_dir = os.path.join(base_dir, "trained_models", "general_pollen")
    model_paths = glob.glob(os.path.join(models_dir, "*", "weights", "best.pt"))
    if not model_paths:
        print("Could not find a general pollen model.")
        return
        
    latest_model = max(model_paths, key=os.path.getctime)
    print(f"Loading YOLO model: {latest_model}")
    model = YOLO(latest_model)
    
    # Find all empty label files (Hard Negatives)
    empty_labels = []
    for lbl_f in glob.glob(os.path.join(labels_dir, "*.txt")):
        if os.path.getsize(lbl_f) == 0:
            empty_labels.append(lbl_f)
            
    print(f"Found {len(empty_labels)} Hard Negative images.")
    
    mined_count = 0
    for i, lbl_f in enumerate(empty_labels):
        base = os.path.basename(lbl_f)
        img_name = base.replace(".txt", ".jpg")
        img_f = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_f):
            continue
            
        img = cv2.imread(img_f)
        if img is None: continue
        
        H, W = img.shape[:2]
        
        # Run detection at a very low confidence threshold to catch ALL artifacts
        results = model.predict(source=img, conf=0.15, verbose=False)
        
        for r in results:
            if r.boxes is None: continue
            
            for box in r.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Add padding
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(W, x2 + pad)
                y2 = min(H, y2 + pad)
                
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Save artifact crop
                out_name = f"mined_artifact_{base.replace('.txt', '')}_{mined_count}.jpg"
                cv2.imwrite(os.path.join(artifact_out_dir, out_name), crop)
                mined_count += 1
                
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(empty_labels)} images... Mined {mined_count} artifacts.")
            
    print(f"\n✅ Done! Successfully mined {mined_count} artifacts and saved them to {artifact_out_dir}")

if __name__ == "__main__":
    main()
