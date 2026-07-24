import os
import cv2
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from ultralytics import YOLO
import random
import boto3
from botocore.config import Config

def extract_species(filename):
    if 'Ran_ado' in filename: return 'Ran_ado'
    if 'Vio_adu' in filename: return 'Vio_adu'
    return 'Unknown'

def tile_image_random(dask_data, tile_size=640):
    """
    Yields 640x640 random tiles lazily using dask arrays to prevent RAM crashes.
    dask_data shape should be (Y, X, C)
    """
    H, W = dask_data.shape[:2]
    stride = tile_size
    coords = []
    
    # Generate non-overlapping grid to avoid duplicates
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            coords.append((x, y))
            
    # Shuffle to get diverse negatives from all parts of the slide
    random.shuffle(coords)
    
    for x, y in coords:
        # We compute() here so that we only load the small 640x640 crop into RAM
        crop = dask_data[y:y + tile_size, x:x + tile_size, :].compute()
        
        # Ignore tiny edge slivers entirely
        if crop.shape[0] < tile_size // 2 or crop.shape[1] < tile_size // 2:
            continue
            
        if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
            
        yield crop, x, y

def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    # Boto3 client with custom endpoint and timeouts to prevent hanging
    config = Config(connect_timeout=60, retries={'max_attempts': 5})
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
        config=config
    )

def main():
    s3_bucket = os.environ.get("S3_BUCKET", "bucket")
    source_prefix = "PEG/Colorado/Source/"
    out_dir = Path("/app/Staged_negatives")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading YOLO model to verify negatives...")
    # Assume best.pt is passed via volume or downloaded via k8s init
    model = YOLO("best.pt")
    
    s3_client = get_s3_client()
    print("Fetching file list from S3...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=source_prefix)
    
    czi_keys = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.czi'):
                    czi_keys.append(obj['Key'])
                    
    print(f"Found {len(czi_keys)} CZI files on S3.")
    czi_keys.sort()
    
    species_targets = {'Ran_ado': 150, 'Vio_adu': 150} # 300 total
    species_counts = {'Ran_ado': 0, 'Vio_adu': 0}
    
    for s3_key in czi_keys:
        filename = s3_key.split('/')[-1]
        species = extract_species(filename)
        if species not in species_targets:
            continue
            
        if species_counts[species] >= species_targets[species]:
            continue
            
        local_path = Path(f"/tmp/{filename}")
        print(f"\nDownloading {filename} from S3...")
        
        try:
            s3_client.download_file(s3_bucket, s3_key, str(local_path))
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            continue

        print(f"Reading {filename} lazily...")
        img = AICSImage(str(local_path))
        
        # Determine the RGB dask array correctly
        try:
            if getattr(img.dims, 'S', 1) == 3:
                # Shape (Y, X, S=3)
                dask_data = img.get_image_dask_data("YXS", T=0, Z=0, C=0)
            else:
                dask_data = img.get_image_dask_data("YXS", T=0, Z=0)
        except Exception:
            try:
                # Fallback if YXS fails
                dask_data = img.get_image_dask_data("YXC", T=0, Z=0)
            except Exception as e:
                print(f"Failed to get dask data for {filename}: {e}")
                local_path.unlink(missing_ok=True)
                continue
        
        for crop, tx, ty in tile_image_random(dask_data):
            if species_counts[species] >= species_targets[species]:
                break
                
            # Run YOLO strict to ensure it's completely empty of pollen
            results = model(crop, conf=0.15, verbose=False)
            
            # Criteria for negative: 0 detections, and not pure black padding
            if len(results[0].boxes) == 0 and (results[0].masks is None or len(results[0].masks) == 0):
                if np.mean(crop) > 20:  # Exclude areas that are completely black/empty scanner background
                    base_stem = filename.replace('.czi', '')
                    out_path = out_dir / f"negative_{base_stem}_x{tx:06d}_y{ty:06d}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    species_counts[species] += 1
                    total = sum(species_counts.values())
                    print(f"Collected negative {total}/300 [{species}: {species_counts[species]}/{species_targets[species]}]", end='\r')

        # Clean up huge file immediately to prevent disk space issues
        local_path.unlink(missing_ok=True)

    print(f"\n✨ Done! Negatives saved to {out_dir}")

if __name__ == '__main__':
    main()
