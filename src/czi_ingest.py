#!/usr/bin/env python3
"""
czi_ingest.py — Convert .czi Zeiss scans into YOLO-ready training tiles.

Pipeline:
  1. Recursively scan a species root directory for .czi files
  2. Extract species code from folder path (e.g. .../Ran_ado/Source/file.czi → Ran_ado)
  3. Perform Max-Intensity Projection (MIP) across Z-slices (lazy via Dask)
  4. Compose an RGB image from chosen fluorescence channels
  5. Tile the composite into 640×640 patches with configurable overlap
  6. Optionally run an existing YOLO model to auto-generate polygon annotations
  7. Write YOLO dataset (images + labels + data.yaml) and a tile_manifest.json
  8. Optionally upload the zipped dataset to CESNET S3

Usage:
    python czi_ingest.py \\
        --root  /path/to/species_root_dir   \\
        --out   /path/to/output_dataset_dir \\
        --model /path/to/viability_best.pt  \\  # optional — for pseudo-labelling
        --channels 0 1 2                    \\  # fluorescence channel indices
        --z mip                             \\  # z-reduction strategy
        --upload                               # push zipped dataset to S3

Requirements:
    pip install aicsimageio aicspylibczi "aicsimageio[czi]" pylibCZIrw
    pip install ultralytics boto3 opencv-python-headless numpy
"""

import argparse
import json
import os
import random
import re
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

# ── Optional imports (fail gracefully so unit tests can run without them) ──
try:
    from aicsimageio import AICSImage
except ImportError:
    AICSImage = None  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

try:
    import boto3
    from botocore.client import Config
except ImportError:
    boto3 = None  # type: ignore

# ── Constants ──────────────────────────────────────────────────────────────
TILE_SIZE           = 640
DEFAULT_OVERLAP     = 0.20   # 20 % overlap between tiles
DEFAULT_SPLIT_RATIO = 0.80   # 80 % train / 20 % val
CONF_THRESHOLD      = 0.35   # Minimum confidence for pseudo-labels
# Biologically reasonable pollen grain area in µm² (filters debris & dust)
MIN_POLLEN_AREA_UM2 = 50.0
MAX_POLLEN_AREA_UM2 = 8000.0


# ── Image utilities ────────────────────────────────────────────────────────

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-channel 0.5–99.5 percentile contrast stretch → uint8 RGB."""
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
    """
    Compute Maximum Intensity Projection across Z for the given channels.
    Returns uint8 RGB array (H, W, 3).

    Memory note: Dask computes each channel independently to keep RAM usage low.
    Peak RAM ≈ (H × W × 4 bytes × n_z_slices) per channel.
    """
    dask_czyx = img.get_image_dask_data("CZYX", S=0)   # (C, Z, H, W)
    channels_data = []
    for c in channels[:3]:                              # cap at 3 channels
        mip = dask_czyx[c].max(axis=0).compute()       # (H, W) — Z collapsed
        channels_data.append(mip)

    # Pad to 3 channels if fewer provided
    while len(channels_data) < 3:
        channels_data.append(np.zeros_like(channels_data[0]))

    rgb = np.stack(channels_data, axis=-1)              # (H, W, 3)
    return normalize_to_uint8(rgb)


def tile_image(rgb: np.ndarray, tile_size: int = TILE_SIZE,
               overlap: float = DEFAULT_OVERLAP):
    """
    Yield (tile_uint8, x_offset, y_offset) patches from an RGB image.
    Pads incomplete border tiles with zeros.
    """
    H, W = rgb.shape[:2]
    stride = int(tile_size * (1 - overlap))

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            crop = rgb[y:y + tile_size, x:x + tile_size]
            # Skip tiny slivers at the far edges
            if crop.shape[0] < tile_size // 4 or crop.shape[1] < tile_size // 4:
                continue
            if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            yield crop, x, y


# ── Species metadata ───────────────────────────────────────────────────────

def extract_species(czi_path: Path) -> str:
    """
    Infer species code from folder structure or filename.
    1. Check for species patterns in filename (e.g. _Ran_ado_).
    2. Check parent folders, skipping generic names like 'Source'.
    """
    # 1. Try filename pattern extraction (e.g. pol_pro_SPECIES_...)
    filename = czi_path.name
    # Common pattern in these files: ...pol_pro_([A-Za-z]+_[A-Za-z]+)...
    match = re.search(r'pol_pro_([A-Za-z]{3}_[A-Za-z]{3})', filename)
    if match:
        return match.group(1)

    # 2. Fallback to folder structure
    parts = czi_path.resolve().parts
    # Walk upward looking for a non-trivial folder that isn't 'Source', etc.
    skip = {"Source", "source", "Data", "data", "Raw", "raw", "Scans", "scans"}
    for part in reversed(parts[:-1]):
        if part not in skip and not part.startswith("/"):
            return part
    return czi_path.parent.name


def build_class_registry(root: str) -> dict[str, int]:
    """
    Recursively scan for .czi files and build a sorted species→class_id dict.
    """
    codes = sorted({extract_species(p) for p in Path(root).rglob("*.czi")})
    return {code: idx for idx, code in enumerate(codes)}


def extract_qr_code(filename: str) -> str | None:
    """
    Extract QR numeric code embedded in the ZEISS filename convention.
    Pattern: YYYYMMDD_NNN_pol_pro_... → NNN
    """
    m = re.search(r'_(\d{3})_', filename)
    return m.group(1) if m else None


# ── YOLO pseudo-labelling ──────────────────────────────────────────────────

def pseudo_label_tile(tile: np.ndarray, model,
                      class_id: int,
                      scale_um_px: float | None,
                      conf: float = CONF_THRESHOLD) -> list[str]:
    """
    Run existing YOLO model on `tile`, re-assign class to `class_id`.
    Optionally filter detections by physical size in µm².
    Returns list of YOLO polygon annotation strings.
    """
    if model is None:
        return []
    H, W = tile.shape[:2]
    results = model(tile, verbose=False, conf=conf)
    lines = []
    if results[0].masks is None:
        return lines

    for mask_xy in results[0].masks.xy:
        if mask_xy.shape[0] < 3:
            continue
        # Physical size filter
        if scale_um_px is not None:
            pts = np.array(mask_xy, dtype=np.int32).reshape(-1, 1, 2)
            area_px  = cv2.contourArea(pts)
            area_um2 = area_px * (scale_um_px ** 2)
            if not (MIN_POLLEN_AREA_UM2 < area_um2 < MAX_POLLEN_AREA_UM2):
                continue

        # Normalize coordinates to [0, 1]
        norm = mask_xy.copy().astype(float)
        norm[:, 0] /= W
        norm[:, 1] /= H
        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        lines.append(f"{class_id} {coords_str}")

    return lines


# ── S3 helpers ─────────────────────────────────────────────────────────────

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


def upload_zip_to_s3(s3_resource, bucket: str, zip_path: str,
                     s3_key: str) -> None:
    """Upload zipped dataset to CESNET S3 staging area."""
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


# ── Main pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert .czi Zeiss scans to YOLO training tiles."
    )
    ap.add_argument("--root",     required=True, help="Species root directory")
    ap.add_argument("--out",      required=True, help="Output YOLO dataset directory")
    ap.add_argument("--model",    default=None,
                    help="Path to existing best.pt for pseudo-labelling (optional)")
    ap.add_argument("--channels", nargs="+", type=int, default=[0, 1, 2],
                    help="CZI channel indices to map to RGB (default: 0 1 2)")
    ap.add_argument("--z",        default="mip", choices=["mip"],
                    help="Z-reduction method (only 'mip' supported)")
    ap.add_argument("--overlap",  type=float, default=DEFAULT_OVERLAP,
                    help="Tile overlap fraction (default 0.20)")
    ap.add_argument("--split",    type=float, default=DEFAULT_SPLIT_RATIO,
                    help="Train split fraction (default 0.80)")
    ap.add_argument("--conf",     type=float, default=CONF_THRESHOLD,
                    help="YOLO confidence threshold for pseudo-labels")
    ap.add_argument("--upload",   action="store_true",
                    help="Upload zipped dataset to CESNET S3 after processing")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print plan but do not write files")
    args = ap.parse_args()

    if AICSImage is None:
        raise ImportError("aicsimageio is required. Run: pip install aicsimageio 'aicsimageio[czi]'")

    # ── Setup output directories ──────────────────────────────────────────
    out_dir = Path(args.out)
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # ── Build species registry ────────────────────────────────────────────
    registry = build_class_registry(args.root)
    print(f"📋 Species registry: {registry}")

    # ── Load optional YOLO model for pseudo-labelling ─────────────────────
    yolo_model = None
    if args.model and YOLO is not None:
        print(f"🤖 Loading YOLO model: {args.model}")
        yolo_model = YOLO(args.model)

    # ── Process each .czi file ────────────────────────────────────────────
    all_czis = sorted(Path(args.root).rglob("*.czi"))
    print(f"🔍 Found {len(all_czis)} .czi files\n")

    manifest = []
    tile_counts = {"train": 0, "val": 0}
    t0 = time.time()

    for czi_path in all_czis:
        species = extract_species(czi_path)
        class_id = registry.get(species)
        if class_id is None:
            print(f"  ⚠️ Unknown species '{species}' for {czi_path.name}, skipping.")
            continue

        qr = extract_qr_code(czi_path.name)
        print(f"  🌸 {czi_path.name}")
        print(f"     Species: {species} → class {class_id}  |  QR: {qr}")

        if args.dry_run:
            print("     [dry-run] Skipping file.")
            continue

        try:
            img = AICSImage(str(czi_path))
            ps  = img.physical_pixel_sizes
            scale_um_px = float(ps.X) if ps.X else None
            print(f"     Scale: {scale_um_px} µm/px  |  Dims: {img.dims}  |  Shape: {img.shape}")

            print("     Computing Max-Z projection…", end="", flush=True)
            rgb = get_mip_rgb(img, args.channels)
            print(f" ✓  ({rgb.shape[0]}×{rgb.shape[1]} px)")

            n_tiles = 0
            for tile, tx, ty in tile_image(rgb, overlap=args.overlap):
                split = "train" if random.random() < args.split else "val"
                stem  = f"{species}_{czi_path.stem}_x{tx:06d}_y{ty:06d}"

                img_path = out_dir / split / "images" / f"{stem}.jpg"
                lbl_path = out_dir / split / "labels"  / f"{stem}.txt"

                # BGR for OpenCV
                cv2.imwrite(str(img_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

                annotation_lines = pseudo_label_tile(
                    tile, yolo_model, class_id, scale_um_px, conf=args.conf
                )
                lbl_path.write_text("\n".join(annotation_lines))

                manifest.append({
                    "tile_id":       stem,
                    "species_code":  species,
                    "class_id":      class_id,
                    "source_czi":    czi_path.name,
                    "x_offset_px":   tx,
                    "y_offset_px":   ty,
                    "um_per_px_x":   scale_um_px,
                    "um_per_px_y":   scale_um_px,
                    "z_method":      args.z,
                    "channels_used": args.channels,
                    "n_annotations": len(annotation_lines),
                    "split":         split,
                    "qr_code":       qr,
                })

                tile_counts[split] += 1
                n_tiles += 1

            print(f"     ✅ {n_tiles} tiles extracted.")

        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ Failed: {exc}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Write data.yaml ───────────────────────────────────────────────────
    names_yaml = "\n".join(
        f"  {i}: {name}"
        for name, i in sorted(registry.items(), key=lambda kv: kv[1])
    )
    yaml_content = (
        f"path: {out_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: {len(registry)}\n"
        f"names:\n{names_yaml}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_content)

    # ── Write manifest ────────────────────────────────────────────────────
    manifest_path = out_dir / "tile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    total = sum(tile_counts.values())
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Done in {elapsed:.1f}s")
    print(f"   {tile_counts['train']} train tiles  |  {tile_counts['val']} val tiles  |  {total} total")
    print(f"   data.yaml     → {out_dir / 'data.yaml'}")
    print(f"   tile_manifest → {manifest_path}")

    # ── Optional S3 upload ────────────────────────────────────────────────
    if args.upload and boto3 is not None:
        zip_path = str(out_dir) + ".zip"
        print(f"\n📦 Zipping dataset to {zip_path}…")
        shutil.make_archive(str(out_dir), "zip", str(out_dir))
        s3_resource, bucket = setup_s3()
        if s3_resource:
            dataset_name = out_dir.name
            s3_key = f"Ostatni/Colorado_pollen_detection/staging_area/{dataset_name}.zip"
            print(f"☁️  Uploading to s3://{bucket}/{s3_key}…")
            upload_zip_to_s3(s3_resource, bucket, zip_path, s3_key)
        else:
            print("⚠️  S3 credentials not found. Skipping upload.")


if __name__ == "__main__":
    main()
