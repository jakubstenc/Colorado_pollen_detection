#!/usr/bin/env python3
"""
extract_300.py — Extract exactly 300 YOLO-ready training tiles per species.
"""

import argparse
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from aicsimageio import AICSImage
except ImportError:
    AICSImage = None

try:
    import boto3
    from botocore.client import Config
except ImportError:
    boto3 = None

TILE_SIZE = 640
DEFAULT_OVERLAP = 0.20

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape[:2] + (arr.shape[2],), dtype=np.uint8)
    for c in range(arr.shape[2]):
        ch = arr[..., c].astype(np.float32)
        valid = ch[ch > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [0.5, 99.5])
        ch = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1) * 255.0
        out[..., c] = ch.astype(np.uint8)
    return out

def get_mip_rgb(img, channels: list[int]) -> np.ndarray:
    if 'S' in img.dims.order and getattr(img.dims, 'S', 1) == 3:
        rgb = img.get_image_data("YXS", T=0, C=0, Z=0)
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = (rgb / 256).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        return rgb

    dask_czyx = img.get_image_dask_data("CZYX", S=0)
    num_c = dask_czyx.shape[0]
    channels_data = []
    
    target_channels = [c for c in channels[:3] if c < num_c]
    
    for c in target_channels:
        mip = dask_czyx[c].max(axis=0).compute()
        channels_data.append(mip)

    while len(channels_data) < 3:
        if len(channels_data) > 0:
            channels_data.append(np.zeros_like(channels_data[0]))
        else:
            channels_data.append(np.zeros((1024, 1024), dtype=np.uint8))

    is_single_channel = (len(target_channels) == 1)
    if is_single_channel:
        active_ch = channels_data[0]
        channels_data = [active_ch, active_ch, active_ch]

    rgb = np.stack(channels_data, axis=-1)
    return normalize_to_uint8(rgb)

def tile_image(rgb: np.ndarray, tile_size: int = TILE_SIZE, overlap: float = DEFAULT_OVERLAP):
    H, W = rgb.shape[:2]
    stride = int(tile_size * (1 - overlap))

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            crop = rgb[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] < tile_size // 4 or crop.shape[1] < tile_size // 4:
                continue
            if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            yield crop, x, y

def extract_species(czi_path: Path) -> str:
    filename = czi_path.name
    match = re.search(r'pol_pro_([A-Za-z]{3}_[A-Za-z]{3})', filename)
    if match:
        return match.group(1)
    parts = czi_path.resolve().parts
    skip = {"Source", "source", "Data", "data", "Raw", "raw", "Scans", "scans"}
    for part in reversed(parts[:-1]):
        if part not in skip and not part.startswith("/"):
            return part
    return czi_path.parent.name

def build_class_registry(root: str) -> dict[str, int]:
    codes = sorted({extract_species(p) for p in Path(root).rglob("*.czi")})
    return {code: idx for idx, code in enumerate(codes)}

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

def upload_zip_to_s3(s3_resource, bucket: str, zip_path: str, s3_key: str) -> None:
    import urllib.request, ssl
    client = s3_resource.meta.client
    url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600,
    )
    size = os.path.getsize(zip_path)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with open(zip_path, "rb") as fh:
        req = urllib.request.Request(url, data=fh, method="PUT")
        req.add_header("Content-Length", str(size))
        with urllib.request.urlopen(req, context=ctx) as resp:
            if resp.status == 200:
                print(f"   ✅ Uploaded {os.path.basename(zip_path)} → s3://{bucket}/{s3_key}")
            else:
                print(f"   ⚠️ Upload returned status {resp.status}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--limit",    type=int, default=300, help="Number of tiles to extract per species")
    args = ap.parse_args()
    
    target_per_species = args.limit

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    registry = build_class_registry(args.root)
    print(f"📋 Species registry: {registry}")

    all_czis = sorted(Path(args.root).rglob("*.czi"))
    print(f"🔍 Found {len(all_czis)} .czi files")
    
    # Track counts per species
    counts = {s: 0 for s in registry.keys()}
    manifest = []

    for czi_path in all_czis:
        species = extract_species(czi_path)
        class_id = registry.get(species)
        
        if counts[species] >= target_per_species:
            continue

        try:
            img = AICSImage(str(czi_path))
            ps  = img.physical_pixel_sizes
            scale_um_px = float(ps.X) if ps.X else None
            
            rgb = get_mip_rgb(img, [0, 1, 2])
            
            # Shuffle tiles so we cover the image uniformly or just randomly select from it
            tiles_list = list(tile_image(rgb, overlap=0.20))
            random.shuffle(tiles_list)
            
            for tile, tx, ty in tiles_list:
                if counts[species] >= target_per_species:
                    break
                    
                stem  = f"{species}_{czi_path.stem}_x{tx:06d}_y{ty:06d}"
                img_path = out_dir / "images" / f"{stem}.jpg"
                lbl_path = out_dir / "labels"  / f"{stem}.txt"

                cv2.imwrite(str(img_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                lbl_path.write_text("") # Empty labels for now, just extracting the images!

                manifest.append({
                    "tile_id":       stem,
                    "species_code":  species,
                    "class_id":      class_id,
                    "source_czi":    czi_path.name
                })

                counts[species] += 1

            print(f"✅ Extracted {counts[species]} tiles for {species} so far.")
            
            all_done = True
            for s in registry.keys():
                if counts[s] < target_per_species:
                    all_done = False
            if all_done:
                print("🎉 Reached 300 tiles for all species!")
                break

        except Exception as exc:
            print(f"  ❌ Failed: {exc}")

    names_yaml = "\n".join(f"  {i}: {name}" for name, i in sorted(registry.items(), key=lambda kv: kv[1]))
    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(registry)}\n"
        f"names:\n{names_yaml}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_content)
    
    manifest_path = out_dir / "tile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    print("\n📦 Zipping dataset…")
    zip_path = str(out_dir) + ".zip"
    shutil.make_archive(str(out_dir), "zip", str(out_dir))
    
    s3_resource, bucket = setup_s3()
    if s3_resource:
        dataset_name = out_dir.name
        s3_key = f"PEG/Colorado/staging_area/{dataset_name}.zip"
        print(f"☁️  Uploading to s3://{bucket}/{s3_key}…")
        upload_zip_to_s3(s3_resource, bucket, zip_path, s3_key)
    else:
        print("⚠️  S3 credentials not found. Skipping upload.")

if __name__ == "__main__":
    main()
