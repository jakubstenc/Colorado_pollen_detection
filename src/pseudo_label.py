#!/usr/bin/env python3
"""
pseudo_label.py — Generate YOLO labels for existing tiles using a pretrained model.
"""

import argparse
import json
import os
import ssl
import urllib.request
from pathlib import Path

import boto3
import cv2
import numpy as np
from botocore.client import Config
from ultralytics import YOLO

def setup_s3():
    s3_endpoint  = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    s3_bucket    = os.environ.get("S3_BUCKET")
    access_key   = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key   = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not all([s3_bucket, access_key, secret_key]):
        return None, None

    resource = boto3.resource(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    )
    return resource, s3_bucket

def download_model_s3(s3_resource, bucket, s3_key, local_path):
    """Download model from S3 if it doesn't exist locally."""
    if os.path.exists(local_path):
        print(f"✅ Model already exists at {local_path}")
        return
    print(f"⬇️ Downloading model from s3://{bucket}/{s3_key}...")
    s3_resource.Bucket(bucket).download_file(s3_key, local_path)

def load_manifest(manifest_path: str):
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, 'r') as f:
        return json.load(f)

def label_tiles(image_dir: str, manifest: list, model_path: str,
                out_dir: str, conf: float = 0.25):
    """Run inference on tiles and save YOLO txt labels."""
    print(f"🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    
    img_root = Path(image_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Map manifest for quick lookup
    manifest_map = {m['tile_id']: m for m in manifest}
    
    # Process images in directory
    images = list(img_root.glob("*.jpg"))
    print(f"🔍 Found {len(images)} images to label.")
    
    for img_path in images:
        tile_id = img_path.stem
        meta = manifest_map.get(tile_id, {})
        class_id = meta.get('class_id', 0)
        scale_um = meta.get('um_per_px_x')
        
        results = model(str(img_path), conf=conf, verbose=False)
        lines = []
        
        if results[0].masks:
            H, W = results[0].orig_shape
            for mask_xy in results[0].masks.xy:
                # Optional: Filter by size here if scale_um is known
                # norm coords
                norm = mask_xy.copy().astype(float)
                norm[:, 0] /= W
                norm[:, 1] /= H
                coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
                lines.append(f"{class_id} {coords_str}")
        
        label_file = out_root / f"{tile_id}.txt"
        label_file.write_text("\n".join(lines))
        print(f"   ✅ {tile_id} → {len(lines)} annotations", end='\r')

    print(f"\n✨ Completed labeling for {len(images)} files.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory with tile images")
    parser.add_argument("--manifest", required=True, help="Path to tile_manifest.json")
    parser.add_argument("--model", required=True, help="YOLO model (.pt) or S3 key")
    parser.add_argument("--out", required=True, help="Labels output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    
    model_path = args.model
    if model_path.startswith("Ostatni/"):
        s3, bucket = setup_s3()
        if s3:
            local_name = os.path.basename(model_path)
            download_model_s3(s3, bucket, model_path, local_name)
            model_path = local_name
        else:
            print("❌ S3 setup failed. Check credentials.")
            return

    manifest = load_manifest(args.manifest)
    label_tiles(args.images, manifest, model_path, args.out, args.conf)

if __name__ == "__main__":
    main()
