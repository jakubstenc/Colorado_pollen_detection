#!/usr/bin/env python3
"""
train_pollen.py — Train a YOLOv11 model on the Colorado Pollen dataset.

This script:
1. Loads the dataset from a specified data.yaml.
2. Initializes a YOLOv8/v11 model.
3. Automatically uses CUDA if available.
4. Saves high-quality results and logs.
"""

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on pollen tiles.")
    parser.add_argument("--data", default="./dataset_ran_ado/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model base (yolov8n.pt, yolov8m.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--name", default="pollen_species_v1", help="Name of the experiment run")
    parser.add_argument("--project", default="Colorado_Pollen", help="Project directory name")
    args = parser.parse_args()

    # 1. Validate paths
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: data.yaml not found at {data_path}")
        return

    # 2. Initialize Model
    print(f"🚀 Initializing model with {args.model}...")
    model = YOLO(args.model)

    # 3. Train
    print(f"📈 Starting training for {args.epochs} epochs on {args.data}...")
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        device=0,         # Use first GPU if available
        exist_ok=True,    # Overwrite if exists
        save=True,        # Save checkpoints
        plots=True        # Generate curves and visuals
    )

    print("\n✅ Training complete.")
    print(f"📁 Runs saved to: {args.project}/{args.name}")

if __name__ == "__main__":
    main()
