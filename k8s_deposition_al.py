import sys
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import random
import boto3
from botocore.config import Config

sys.path.append("/scripts") # This will be mounted in K8s
sys.path.append("/app/src") # Ensure src is accessible if needed, but actually let's just copy focus_check and build_species_dataset into the container?
# Wait! In extract-negatives-job, the docker image has /app/src containing the python files because it's built from Dockerfile.train which does `COPY . /app`!
# So we can just append /app/src
sys.path.append("/app/src")

from build_species_dataset import tile_image, extract_general_pollen, get_mip_rgb, AICSImage
from focus_check import compute_focus_score

def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
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
    out_dir = Path("/app/Trainig_data") # We will sync this directory to S3
    model_path = "/app/best.pt" # Downloaded by init script or just before python runs
    s3_bucket = os.environ.get("S3_BUCKET", "bucket")
    source_prefix = "PEG/Colorado/Source/Pollen_deposition/"
    conf_thresh = 0.65
    class_id = 0 # Using 0 since these are for the general pollen model
    species = "Deposition_Stigmas"

    print("🤖 Loading YOLO model...")
    model = YOLO(model_path)

    spec_img_dir = out_dir / species / "Images"
    spec_lbl_dir = out_dir / species / "Labels"
    spec_viz_dir = out_dir / species / "Vizualization"
    neg_dir = out_dir / "Negatives"

    for d in [spec_img_dir, spec_lbl_dir, spec_viz_dir, neg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    s3_client = get_s3_client()
    print("Fetching file list from S3...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=source_prefix)
    
    all_czis = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.czi'):
                    all_czis.append(obj['Key'])
                    
    print(f"Found {len(all_czis)} total CZI files in {source_prefix}.")

    random.seed(42)
    random.shuffle(all_czis)

    success_count = 0
    t_tiles = 0
    t_hits = 0
    t_negs = 0

    print(f"\n🌸 Processing {species}")
    for s3_key in all_czis:
        if success_count >= 10:
            break
            
        filename = s3_key.split('/')[-1]
        local_path = Path(f"/tmp/{filename}")
        
        print(f"\n📥 Downloading {filename} from S3...")
        try:
            s3_client.download_file(s3_bucket, s3_key, str(local_path))
        except Exception as e:
            print(f"  ❌ Error downloading {filename}: {e}")
            continue

        try:
            img = AICSImage(str(local_path))
            rgb = get_mip_rgb(img)
            
            # Resolution Adaptation Check (2x downsampling if deposition scan at ~0.44 um/px)
            px_size = getattr(img.physical_pixel_sizes, 'X', None)
            scale_str = "1.0x"
            if px_size is not None and px_size < 0.65:
                rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
                scale_str = "0.5x (2x downsampled)"
                print(f"   -> 📐 Applied spatial resolution scale: {scale_str} (original: {px_size:.4f} µm/px)")
                
            # 1. Blur/Focus Check (Laplacian variance scales quadratically with spatial resolution: ~15.0 for 2x downsampled)
            blur_threshold = 15.0 if (px_size is not None and px_size < 0.65) else 150.0
            blur_score = compute_focus_score(rgb)
            if blur_score < blur_threshold:
                print(f"   -> ⚠️ Skipping due to blur (score: {blur_score:.2f} < threshold {blur_threshold})")
                continue
                
            print(f"   -> ✅ Image is focused (score: {blur_score:.2f} >= threshold {blur_threshold}). Processing into tiles...")
                
            if len(rgb.shape) < 3 or rgb.shape[0] < 640 or rgb.shape[1] < 640:
                raise Exception(f"Invalid RGB Extracted: {rgb.shape}")
                
            n_tiles, n_hits, n_negs = 0, 0, 0
            for tile, tx, ty in tile_image(rgb):
                n_tiles += 1
                stem = f"{species}_{local_path.stem}_x{tx:06d}_y{ty:06d}"
                detections = extract_general_pollen(tile, model, conf_thresh)
                
                if len(detections) == 0:
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(neg_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n_negs += 1
                else:
                    n_hits += 1
                    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(spec_img_dir / f"{stem}.jpg"), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    lbl_lines = []
                    H, W = tile_bgr.shape[:2]
                    for d in detections:
                        poly_px = d['poly_px'].reshape((-1, 1, 2))
                        mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask, [poly_px], 255)
                        
                        hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
                        S = hsv[:,:,1]
                        S_blurred = cv2.GaussianBlur(S, (5, 5), 0)
                        _, binary = cv2.threshold(S_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        binary = cv2.bitwise_and(binary, binary, mask=mask)
                        kernel = np.ones((5,5), np.uint8)
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            epsilon = 0.003 * cv2.arcLength(largest_contour, True)
                            smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                            final_poly_px = smoothed_contour.reshape(-1, 2)
                        else:
                            final_poly_px = d['poly_px']
                            
                        norm_xy = final_poly_px.astype(float)
                        norm_xy[:, 0] /= W
                        norm_xy[:, 1] /= H
                        
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_xy)
                        lbl_lines.append(f"{class_id} {coords}")
                        
                    (spec_lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines))
                    
                    viz_bgr = tile_bgr.copy()
                    for d in detections:
                        poly_px = d['poly_px'].reshape((-1, 1, 2))
                        cv2.polylines(viz_bgr, [poly_px], True, (255, 0, 255), 2)
                        text_str = f"{d['conf']:.2f}"
                        px, py = poly_px[0][0]
                        cv2.putText(viz_bgr, text_str, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(viz_bgr, text_str, (int(px)-5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                        
                    cv2.imwrite(str(spec_viz_dir / f"{stem}_viz.jpg"), viz_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
            print(f"   ✓ {n_tiles} tiles -> {n_hits} Overlays | {n_negs} Negatives")
            t_tiles += n_tiles
            t_hits += n_hits
            t_negs += n_negs
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error parsing {filename}: {e}")
        finally:
            if local_path.exists():
                local_path.unlink()
                
    print(f"   => TOTAL: {t_tiles} tiles -> {t_hits} Overlays | {t_negs} Negatives")
    print("✅ ALL SETUP COMPLETED. You can now use the UI to curate 'Deposition_Stigmas'.")

if __name__ == '__main__':
    main()
