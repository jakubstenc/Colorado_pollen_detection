"""
extract_spores.py
-----------------
Auto-labels Lycopodium spore tiles from S3 as class 46 (Lyc_spo).

Strategy
--------
The 3 CZI files under PEG/Colorado/Source/Spores/ contain *only* spores —
no pollen. We run the current general model at low confidence (conf=0.25) to
detect all particle-like blobs, then relabel every detection as class 46
(Lyc_spo) instead of class 0 (pollen).

Output goes to:
  Species_model/Trainig_data/Lyc_spo/Images/
  Species_model/Trainig_data/Lyc_spo/Labels/
  Species_model/Trainig_data/Lyc_spo/Vizualization/

These paths are picked up automatically by the Active Learning UI
(active_learning_ui.py scrapes all of Species_model/Trainig_data/).

Usage (local):
  cd /home/meow/Documents/Antigravity/Colorado_pollen_detection
  .venv/bin/python extract_spores.py [--conf 0.25] [--max-tiles 200]

Usage (k8s / container): set S3_BUCKET, S3_ENDPOINT, AWS_ACCESS_KEY_ID,
                          AWS_SECRET_ACCESS_KEY as env vars.
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
import cv2
import numpy as np
import urllib3
from botocore.client import Config

urllib3.disable_warnings()

sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append("/app/src")
sys.path.append("src")

from build_species_dataset import get_mip_rgb, tile_image, AICSImage

# ── Constants ────────────────────────────────────────────────────────────────

SPORE_CLASS_ID   = 46
SPORE_CLASS_NAME = "Lyc_spo"

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET   = os.environ.get("S3_BUCKET",   "bucket")
S3_ACCESS   = os.environ.get("AWS_ACCESS_KEY_ID",     "1Y920BKC0SAWPNDE8RD6")
S3_SECRET   = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")

SOURCE_PREFIX = "PEG/Colorado/Source/Spores/"
OUT_PREFIX    = "PEG/Colorado/Species_model/Trainig_data/Lyc_spo"

LOCAL_SRC = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source/Spores")
LOCAL_OUT = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Species_model/Trainig_data/Lyc_spo")

TMP_DIR = Path("/tmp/spore_extract")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── S3 helpers ───────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS,
        aws_secret_access_key=S3_SECRET,
        config=Config(
            signature_version="s3v4",
            connect_timeout=60,
            retries={"max_attempts": 5},
            s3={"payload_signing_enabled": False},
        ),
        verify=False,
    )


def list_spore_czis(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".czi"):
                keys.append(obj["Key"])
    return keys


# ── Core extraction ──────────────────────────────────────────────────────────

def snap_mask_to_color_contour(tile_bgr, poly_px):
    """Refine the raw YOLO mask to the dominant colour contour inside it."""
    H, W = tile_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px.reshape((-1, 1, 2))], 255)

    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    S   = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, binary = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        lc  = max(contours, key=cv2.contourArea)
        eps = 0.003 * cv2.arcLength(lc, True)
        return cv2.approxPolyDP(lc, eps, True).reshape(-1, 2)
    return poly_px


def process_czi(czi_path: Path, model, out_dirs: dict, conf_thresh: float, max_tiles: int):
    """
    Tile one CZI, run the model, relabel all detections as Lyc_spo (46),
    save image + label + viz.
    Returns (n_tiles_saved, n_detections_total).
    """
    img_dir = out_dirs["images"]
    lbl_dir = out_dirs["labels"]
    viz_dir = out_dirs["viz"]

    for d in [img_dir, lbl_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"   📖 Reading {czi_path.name} via AICSImage...")
    aics = AICSImage(str(czi_path))
    rgb  = get_mip_rgb(aics)

    if len(rgb.shape) < 3 or rgb.shape[0] < 640 or rgb.shape[1] < 640:
        raise ValueError(f"Image too small for tiling: {rgb.shape}")

    n_saved = 0
    n_dets  = 0

    for tile_rgb, tx, ty in tile_image(rgb):
        if n_saved >= max_tiles:
            break
        if np.mean(tile_rgb) < 15:   # skip black scanner background
            continue

        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
        results  = model(tile_bgr, conf=conf_thresh, verbose=False, retina_masks=True)

        if results[0].masks is None or results[0].boxes is None:
            continue
        if len(results[0].masks.xy) == 0:
            continue

        H, W      = tile_bgr.shape[:2]
        lbl_lines = []
        viz_bgr   = tile_bgr.copy()

        for mask_xy, box in zip(results[0].masks.xy, results[0].boxes):
            if mask_xy.shape[0] < 3:
                continue
            det_conf = float(box.conf[0])
            poly_raw = mask_xy.astype(np.int32)
            poly     = snap_mask_to_color_contour(tile_bgr, poly_raw)
            if len(poly) < 3:
                continue

            norm_xy       = poly.astype(float)
            norm_xy[:, 0] /= W
            norm_xy[:, 1] /= H
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_xy)
            lbl_lines.append(f"{SPORE_CLASS_ID} {coords}")
            n_dets += 1

            # Draw detection — orange colour to distinguish from pollen (green)
            cv2.polylines(viz_bgr, [poly.reshape((-1, 1, 2))], True, (0, 140, 255), 2)
            text = f"Lyc {det_conf:.2f}"
            px, py = int(poly[0][0]), int(poly[0][1])
            cv2.putText(viz_bgr, text, (px - 5, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(viz_bgr, text, (px - 5, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if not lbl_lines:
            # No detections → this tile is likely scanner background; skip
            continue

        stem = f"Lyc_spo_{czi_path.stem}_x{tx:06d}_y{ty:06d}"

        cv2.imwrite(str(img_dir / f"{stem}.jpg"),     tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(viz_dir / f"{stem}_viz.jpg"), viz_bgr,  [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines))

        n_saved += 1
        print(f"   [{n_saved:3d}] x={tx} y={ty} → {len(lbl_lines)} spore dets", end="\r")

    print()
    return n_saved, n_dets


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label Lycopodium spore tiles as class 46 (Lyc_spo)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25). "
             "Lower catches more spores; raise to 0.35+ to suppress false blobs."
    )
    parser.add_argument(
        "--max-tiles", type=int, default=200,
        help="Max labelled tiles to save per CZI (default: 200)."
    )
    parser.add_argument(
        "--model", type=str,
        default="/home/meow/Documents/Antigravity/Colorado_pollen_detection/models/general_pollen/latest.pt",
        help="Path to the current general pollen model (best.pt / latest.pt)."
    )
    args = parser.parse_args()

    from ultralytics import YOLO
    print(f"🤖 Loading model: {args.model}")
    model = YOLO(args.model)

    # Prefer local CESNET mount; fall back to S3 download
    use_local_mount = LOCAL_OUT.parent.parent.exists()
    if use_local_mount:
        out_base = LOCAL_OUT
        print(f"📂 Output → local mount: {out_base}")
    else:
        out_base = TMP_DIR / "output" / "Lyc_spo"
        print(f"📂 Output → /tmp (will upload to S3): {out_base}")

    out_dirs = {
        "images": out_base / "Images",
        "labels": out_base / "Labels",
        "viz":    out_base / "Vizualization",
    }

    # ── Locate CZI files ─────────────────────────────────────────────────────
    use_s3_download = False
    if LOCAL_SRC.exists():
        czi_paths = list(LOCAL_SRC.rglob("*.czi"))
        print(f"🔍 Found {len(czi_paths)} CZI(s) in local mount: {LOCAL_SRC}")
    else:
        # Check if we are running inside a k8s container (no HOME-based CESNET mount).
        # Downloading CZIs locally is intentionally blocked to avoid filling the user's disk.
        # Run via k8s instead: ./deploy_extract_spores.sh
        in_k8s = os.path.exists("/var/run/secrets/kubernetes.io") or os.environ.get("KUBERNETES_SERVICE_HOST")
        if not in_k8s:
            print("❌  Local CESNET mount not found at:", LOCAL_SRC)
            print()
            print("   CZI files are too large to download to a local machine.")
            print("   Run this as a Kubernetes job instead:")
            print()
            print("     ./deploy_extract_spores.sh")
            print()
            print("   The job runs on the cluster with plenty of disk, downloads")
            print("   the CZIs there, processes them, and uploads tiles to S3.")
            sys.exit(1)

        print("🌐 Running in k8s — fetching CZI list from S3…")
        s3 = get_s3_client()
        czi_keys  = list_spore_czis(s3, S3_BUCKET, SOURCE_PREFIX)
        print(f"🔍 Found {len(czi_keys)} CZI(s) at S3:{SOURCE_PREFIX}")
        czi_paths = []
        for key in czi_keys:
            local = TMP_DIR / Path(key).name
            print(f"   ⬇️  Downloading {Path(key).name}…")
            s3.download_file(S3_BUCKET, key, str(local))
            czi_paths.append(local)
        use_s3_download = True

    if not czi_paths:
        print("❌ No CZI files found. Check that Source/Spores/ contains .czi files.")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    total_tiles = 0
    total_dets  = 0

    for czi_path in czi_paths:
        print(f"\n🧪 Processing: {czi_path.name}")
        try:
            n_tiles, n_dets = process_czi(
                czi_path, model, out_dirs, args.conf, args.max_tiles
            )
            print(f"   ✓ {n_tiles} tiles saved, {n_dets} spore annotations")
            total_tiles += n_tiles
            total_dets  += n_dets
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        finally:
            if use_s3_download and czi_path.exists():
                czi_path.unlink()

    # ── Upload to S3 if we could not write directly ───────────────────────────
    if not use_local_mount:
        print("\n⬆️  Uploading tiles to S3…")
        s3 = get_s3_client()
        for subdir_key, subdir_path in [
            ("Images",        out_dirs["images"]),
            ("Labels",        out_dirs["labels"]),
            ("Vizualization", out_dirs["viz"]),
        ]:
            for fpath in subdir_path.glob("*"):
                s3_key = f"{OUT_PREFIX}/{subdir_key}/{fpath.name}"
                s3.upload_file(str(fpath), S3_BUCKET, s3_key)
        print("✅ S3 upload complete.")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✨ Lycopodium Spore Extraction Complete                 ║
║                                                          ║
║  Total tiles saved  : {total_tiles:<6d}                         ║
║  Total annotations  : {total_dets:<6d} (all class 46 Lyc_spo)   ║
║                                                          ║
║  Next step: open the Active Learning UI and review       ║
║  tiles in the 'Lyc_spo' species folder.                  ║
║                                                          ║
║  💡 Too many false blobs? Re-run with --conf 0.35        ║
║     Too few spores detected? Try --conf 0.15             ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
