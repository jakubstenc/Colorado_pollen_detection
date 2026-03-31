#!/usr/bin/env python3
import os
import shutil
import zipfile
import argparse
from pathlib import Path
from ultralytics import YOLO

def convert_bbox_to_polygon(bbox_line):
    """
    Takes a YOLO format string: "class x1 y1 x2 y2 ..."
    If it's a bounding box (5 parts), converts to 4-point polygon.
    If it's already a polygon (>5 parts), just forces class to 0.
    """
    parts = bbox_line.strip().split()
    if len(parts) < 5:
        return "" # invalid
    
    # Force class 0 (pollen)
    parts[0] = "0"

    if len(parts) == 5:
        # It's a bounding box: xc, yc, w, h
        try:
            xc, yc, w, h = map(float, parts[1:5])
            
            # Area/Size filter: Reject massive ghost boxes (hallucinations)
            # A pollen grain should not consume more than 25% of the image width/height.
            if w > 0.25 or h > 0.25:
                return ""
            
            x_min = max(0.0, xc - w / 2)
            x_max = min(1.0, xc + w / 2)
            y_min = max(0.0, yc - h / 2)
            y_max = min(1.0, yc + h / 2)
            # 4 corners of a rectangle
            return f"0 {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {x_min:.6f} {y_max:.6f}"
        except:
            return ""
    
    # It's already a polygon or has custom extra info
    return " ".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to input dataset zip")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--outdir", required=True, help="Path to output directory to zip")
    parser.add_argument("--filter-positives", action="store_true", help="Only map images that yield YOLO detections")
    parser.add_argument("--max-images", type=int, default=0, help="Halt mapping after hitting this many tracked tiles")
    args = parser.parse_args()

    work_dir = Path("/tmp/pseudo_label_work")
    output_dir = Path(args.outdir)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"📦 Extracting dataset from {args.zip}...")
    with zipfile.ZipFile(args.zip, 'r') as zip_ref:
        zip_ref.extractall(work_dir)

    print("🤖 Loading model...")
    model = YOLO(args.model)

    images_dir = next(work_dir.glob('**/images'), None)
    if not images_dir or not images_dir.exists():
        print("❌ Could not find images directory in the zip.")
        return

    print("🔍 Running inference and generating YOLO labels...")
    model.predict(
        source=str(images_dir),
        save=False,
        save_txt=True,
        save_conf=False,
        conf=0.25, # standard confidence for the newly trained v3 model
        project=str(work_dir),
        name="predict_out"
    )

    pred_labels_dir = work_dir / "predict_out" / "labels"
    if not pred_labels_dir.exists():
        print("⚠️ No labels were generated. Creating empty directory.")
        pred_labels_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "images").mkdir(parents=True)
    (output_dir / "labels").mkdir(parents=True)

    # 1/2) Consolidate processing: Only write output images relative to predicted txts, matching the positive filter logic
    print("✂️ Converting bounding box predictions to 4-point segment polygons (class 0)...")
    saved_count = 0
    
    for label_file in pred_labels_dir.glob("*.txt"):
        if args.max_images > 0 and saved_count >= args.max_images:
            break
            
        if label_file.is_file():
            original_lines = label_file.read_text().strip().split("\n")
            new_lines = []
            for line in original_lines:
                if not line.strip():
                    continue
                poly_line = convert_bbox_to_polygon(line)
                if poly_line:
                    new_lines.append(poly_line)
            
            if args.filter_positives and not new_lines:
                continue
                
            matched_imgs = list(images_dir.glob(label_file.stem + ".*"))
            if matched_imgs:
                img_file = matched_imgs[0]
                shutil.copy(img_file, output_dir / "images" / img_file.name)
                (output_dir / "labels" / label_file.name).write_text("\n".join(new_lines))
                saved_count += 1

    # If NOT filtering positives, sweep any remaining untouched empty tiles to fulfill capacity
    if not args.filter_positives:
        for img in images_dir.glob("*"):
            if args.max_images > 0 and saved_count >= args.max_images:
                break
            if img.is_file():
                dest_img = output_dir / "images" / img.name
                if not dest_img.exists():
                    shutil.copy(img, dest_img)
                    (output_dir / "labels" / (img.stem + ".txt")).write_text("")
                    saved_count += 1

    # 3) Build a new simple data.yaml for Roboflow (unified "pollen" class)
    yaml_content = (
        f"path: .\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['pollen']\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content)

    print(f"✅ Prepared dataset in {output_dir}")

if __name__ == "__main__":
    main()
