#!/usr/bin/env python3
import os
import argparse
from ultralytics import YOLO
from datetime import datetime

MODEL_DIR = os.getenv("MODEL_DIR", "/home/meow/Documents/Antigravity/Colorado_pollen_detection/models/species_classifier")

def train_species(dataset_dir, target_species):
    os.makedirs(MODEL_DIR, exist_ok=True)
    run_name = f"species_classifier_{target_species}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print(f"🚀 Training {run_name} on Binary Species Classification (Conspecific vs Heterospecific)...")
    print(f"📂 Dataset: {dataset_dir}")
    
    # We use the fast YOLOv8 Small image classification architecture
    model = YOLO("yolov8s-cls.pt") 
    
    results = model.train(
        data=dataset_dir,
        epochs=100,
        patience=20,
        batch=32,
        imgsz=224, # Standard for classification
        project=MODEL_DIR,
        name=run_name,
        # Standard augmentations for classification
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mixup=0.2,   # Dynamically blends different crops to teach the model how to handle multi-pollen visual overlap boundaries
        erasing=0.1  # Randomly masks parts of pollen so it learns robust multi-feature detection rather than relying on one detail
    )
    
    print("\n📊 Training Completed! Running deep validation for per-class performance insights...")
    
    # Run a dedicated validation phase on the best weights to explicitly generate performance metrics & confusion matrices
    val_results = model.val(
        data=dataset_dir,
        imgsz=224,
        batch=32,
        project=MODEL_DIR,
        name=run_name + "_evaluation",
        plots=True # Guarantees rendering of the Confusion Matrix (showing struggles per class)
    )
    
    actual_save_dir = results.save_dir
    best_weights = os.path.join(actual_save_dir, "weights", "best.pt")
    
    import shutil
    # Copy to the new binary classifier name
    target_model_name = os.path.join(MODEL_DIR, f"species_classifier_{target_species}.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_model_name)
        
        # Also copy to latest.pt just for pipeline backwards compatibility
        latest_copy = os.path.join(MODEL_DIR, "latest.pt")
        shutil.copy(best_weights, latest_copy)
        
        print(f"✅ Binary Species Training (cls) Complete! Target model saved to: {target_model_name}")
    else:
        print("⚠️ Training completed but could not find best.pt!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the specific binary species classification model (YOLOv8-cls).")
    parser.add_argument("--dataset", required=True, help="Path to classification dataset root directory containing train/ and val/")
    parser.add_argument("--target_species", type=str, default="Ran_ado", help="Target species the model was trained for")
    args = parser.parse_args()
    train_species(args.dataset, args.target_species)
