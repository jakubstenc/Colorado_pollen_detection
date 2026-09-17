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

# Typical single-spore area at the working resolution (~0.88 µm/px).  Used as a
# hard floor so tiles where every spore is touching still trigger the split.
# Lycopodium spore ≈ 25 µm diameter → radius ≈ 14 px → area ≈ π·14² ≈ 615 px²
SINGLE_SPORE_AREA_PX = 700


def _watershed_split_blob(binary_mask: np.ndarray,
                           tile_bgr: np.ndarray = None,
                           min_area_px: int = 300) -> list:
    """
    Given a binary mask (uint8, single blob region already isolated),
    use distance-transform + watershed to split touching spores.

    Returns a list of contour arrays (each shape Nx2, dtype int32).
    Returns an empty list if splitting is not productive.

    Parameters
    ----------
    binary_mask : uint8 ndarray, same H×W as the tile
    tile_bgr    : optional real BGR tile — used as the watershed proxy image
                  instead of a blank mask so the algorithm has genuine image
                  gradients to delineate grain boundaries.
    min_area_px : minimum contour area to keep after splitting
    """
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # 20 % threshold — the saddle between two closely-touching spores
    # sits well above 35 %, so the lower value is needed to find two distinct peaks.
    peak_thresh = max(dist.max() * 0.20, 3.0)
    _, sure_fg = cv2.threshold(dist, peak_thresh, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    if num_labels < 3:   # fewer than 2 foreground peaks → can't split
        return []

    markers = markers + 1   # background → 1, grains → 2…N
    sure_bg = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers[unknown == 255] = 0

    # Build the proxy image for watershed.
    # Using the real BGR tile (inside the blob region) gives watershed genuine
    # intensity gradients at grain boundaries → cleaner split lines.
    H, W = binary_mask.shape
    if tile_bgr is not None and tile_bgr.shape[:2] == (H, W):
        # Mask out everything outside the blob so watershed doesn't wander
        proxy = tile_bgr.copy()
        proxy[binary_mask == 0] = 0
        # Enhance edges with a small Laplacian so watershed "snaps" to grain borders
        gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap = np.clip(np.abs(lap) * 0.5, 0, 255).astype(np.uint8)
        proxy[:, :, 0] = np.clip(proxy[:, :, 0].astype(np.int32) + lap, 0, 255).astype(np.uint8)
    else:
        proxy = np.zeros((H, W, 3), dtype=np.uint8)
        proxy[:, :, 0] = binary_mask

    cv2.watershed(proxy, markers)

    sub_contours = []
    for label in range(2, num_labels + 1):
        grain_mask = np.zeros((H, W), dtype=np.uint8)
        grain_mask[markers == label] = 255

        cnts, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area_px:
            continue
        sub_contours.append(largest)

    return sub_contours


def detect_spore_blobs(
    tile_bgr: np.ndarray,
    min_area_px: int = 300,
    max_area_px: int = 18000,
    min_circularity: float = 0.30,
    min_solidity: float = 0.55,
    cluster_ratio: float = 1.3,   # lowered from 1.7: catches moderately-merged pairs
) -> list:
    """
    Detect Lycopodium spore blobs using classical CV (no model needed).

    Spores are dark gray/blue objects against a very light pink background.
    Strategy:
      1. Convert to grayscale
      2. Otsu threshold → binary mask of dark objects
      3. Morphological cleanup (close small holes, remove tiny noise)
      4. Contour extraction with shape filters (area, circularity, solidity)
      5. **Watershed splitting** for any contour whose area exceeds
         ``cluster_ratio`` × median single-spore area — handles touching spores
         that Otsu merges into one large blob.

    Returns a list of (contour_np_array, score_float) tuples where
    score is a quality proxy in [0, 1] (higher = more circular/solid).
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)

    # Mild blur to suppress fine texture noise before threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu on inverted image so dark objects → white foreground
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup: close internal holes, remove pepper noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── Collect raw candidates (area only) to estimate single-spore size ──────
    raw_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area_px <= area <= max_area_px * cluster_ratio * 3:
            raw_candidates.append((cnt, area))

    # Estimate single-spore reference area from the lower half of detections.
    # IMPORTANT: always clamp to SINGLE_SPORE_AREA_PX so that tiles where every
    # spore is touching (no isolated single spores) still produce a sensible
    # baseline instead of using the cluster blob area as the reference.
    if len(raw_candidates) > 1:
        sorted_areas = sorted(a for _, a in raw_candidates)
        reference_area = float(np.median(sorted_areas[:max(1, len(sorted_areas) // 2)]))
    elif raw_candidates:
        reference_area = float(raw_candidates[0][1])
    else:
        reference_area = float(SINGLE_SPORE_AREA_PX)
    reference_area = max(reference_area, float(SINGLE_SPORE_AREA_PX))

    # ── Build final detections, splitting clusters via watershed ──────────────
    H, W = tile_bgr.shape[:2]
    detections = []

    def _score_and_append(cnt):
        """Apply shape filters and append (poly, score) if the contour passes."""
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > max_area_px:
            return
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            return
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0:
            return
        solidity = area / hull_area
        if solidity < min_solidity:
            return
        score = (circularity + solidity) / 2.0
        eps  = 0.005 * perimeter
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(poly) >= 3:
            detections.append((poly, score))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        if area > cluster_ratio * reference_area:
            # Likely a cluster of touching spores — try watershed split
            blob_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            sub = _watershed_split_blob(blob_mask, tile_bgr=tile_bgr, min_area_px=min_area_px)
            if len(sub) >= 2:
                for sub_cnt in sub:
                    _score_and_append(sub_cnt)
                continue   # successfully split; skip the merged contour
            # Splitting didn't help — fall through to normal filter

        _score_and_append(cnt)

    return detections



def is_label_region(tile_rgb: np.ndarray, white_thresh: float = 0.92, uniformity_thresh: float = 18.0) -> bool:
    """
    Returns True if the tile looks like the physical slide label sticker rather
    than actual microscopy content.

    Heuristics (any one is sufficient):
    - >92 % of pixels are near-white  (mean > 230 in all channels)
    - Very low std-dev across the tile (uniformly white/blank)
    """
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    white_fraction = np.mean(gray > 230)
    if white_fraction > white_thresh:
        return True
    # Also reject extremely uniform light tiles (blank label background)
    if gray.mean() > 200 and gray.std() < uniformity_thresh:
        return True
    return False


def process_czi(czi_path: Path, model, out_dirs: dict, conf_thresh: float, max_tiles: int):
    """
    Tile one CZI, detect Lycopodium spore blobs with classical CV,
    relabel all detections as Lyc_spo (46), save image + label + viz.
    `model` and `conf_thresh` are kept for API compatibility but unused.
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

    # ── Resolution normalisation ──────────────────────────────────────────────
    # The general pollen model was trained on ~0.88 µm/px slides.  Spore slides
    # are sometimes acquired at ~0.44 µm/px (2x higher), making spores appear
    # half the size.  Downsample so the apparent object size matches training.
    px_size = getattr(aics.physical_pixel_sizes, 'X', None)
    scale_note = "1.0x"
    if px_size is not None and px_size < 0.65:
        rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2),
                         interpolation=cv2.INTER_AREA)
        scale_note = f"0.5x (was {px_size:.3f} µm/px → 2× downsampled)"
    print(f"   🔬 Resolution: {px_size} µm/px  |  scale applied: {scale_note}")

    if len(rgb.shape) < 3 or rgb.shape[0] < 640 or rgb.shape[1] < 640:
        raise ValueError(f"Image too small for tiling: {rgb.shape}")

    n_saved = 0
    n_dets  = 0
    n_skipped_label = 0
    n_skipped_blank = 0

    for tile_rgb, tx, ty in tile_image(rgb):
        if n_saved >= max_tiles:
            break
        if np.mean(tile_rgb) < 15:        # skip black scanner background
            n_skipped_blank += 1
            continue
        if is_label_region(tile_rgb):     # skip physical slide label sticker
            n_skipped_label += 1
            continue

        tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

        H, W      = tile_bgr.shape[:2]
        lbl_lines = []
        viz_bgr   = tile_bgr.copy()

        # ── Classical blob detection (replaces YOLO) ──────────────────────────
        # Spores are dark circular blobs on a light background.
        # Adjust min/max area for the effective pixel size after downsampling.
        # At 0.88 µm/px, a 25 µm spore is ~28px diameter → area ≈ 615 px²
        # At 0.44 µm/px (before 2× down), effective ≈ same after downsampling.
        blobs = detect_spore_blobs(tile_bgr)
        for poly, score in blobs:
            norm_xy       = poly.astype(float)
            norm_xy[:, 0] /= W
            norm_xy[:, 1] /= H
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_xy)
            lbl_lines.append(f"{SPORE_CLASS_ID} {coords}")
            n_dets += 1

            # Draw detection — orange outline
            cv2.polylines(viz_bgr, [poly.reshape((-1, 1, 2))], True, (0, 140, 255), 2)
            text = f"Lyc {score:.2f}"
            px, py = int(poly[0][0]), int(poly[0][1])
            cv2.putText(viz_bgr, text, (px - 5, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(viz_bgr, text, (px - 5, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Save ALL content tiles regardless of detection count.
        # These are spore-only slides, so blank label files are intentional
        # (they tell the trainer: this tile has no detectable objects).
        stem = f"Lyc_spo_{czi_path.stem}_x{tx:06d}_y{ty:06d}"

        cv2.imwrite(str(img_dir / f"{stem}.jpg"),     tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(viz_dir / f"{stem}_viz.jpg"), viz_bgr,  [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines))

        n_saved += 1
        n_ann = len(lbl_lines)
        suffix = f" → {n_ann} spore det{'s' if n_ann != 1 else ''}" if n_ann > 0 else " (no detections — saved as context)"
        print(f"   [{n_saved:3d}] x={tx} y={ty}{suffix}", end="\r")

    print()
    print(f"   (skipped: {n_skipped_blank} blank, {n_skipped_label} label-sticker tiles)")
    return n_saved, n_dets


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label Lycopodium spore tiles as class 46 (Lyc_spo)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.30,
        help="Minimum blob circularity threshold for classical detector "
             "(0.0–1.0, default: 0.30). Lower catches more irregular shapes."
    )
    parser.add_argument(
        "--max-tiles", type=int, default=200,
        help="Max tiles to save per CZI (default: 200)."
    )
    parser.add_argument(
        "--model", type=str,
        default=None,
        help="[Unused] Path to a YOLO model. Classical CV detection is used instead."
    )
    args = parser.parse_args()

    from ultralytics import YOLO  # noqa: F401  (kept for container compatibility)
    print("🔬 Using classical OpenCV blob detection (no YOLO model needed)")
    model = None   # unused

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
║  💡 Too many false blobs? Re-run with --conf 0.25        ║
║     Too few spores detected? Try --conf 0.10             ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
