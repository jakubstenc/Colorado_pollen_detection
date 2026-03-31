import os
import cv2
from pathlib import Path
from aicsimageio import AICSImage
import numpy as np
import re

def get_mip_rgb(img):
    if 'S' in img.dims.order and getattr(img.dims, 'S', 1) == 3:
        rgb = img.get_image_data("YXS", T=0, C=0, Z=0)
        if rgb.dtype != np.uint8:
            if rgb.dtype in (np.uint16, np.float32, np.float64) and rgb.max() > 255:
                rgb = (rgb / 256).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        return rgb
    return None

def tile_image(rgb, tile_size=640, overlap=0.0):
    H, W = rgb.shape[:2]
    stride = int(tile_size * (1 - overlap))
    
    # Sample from the center 50% to avoid edges
    start_y = int(H * 0.25)
    end_y = int(H * 0.75)
    start_x = int(W * 0.25)
    end_x = int(W * 0.75)

    for y in range(start_y, end_y, stride):
        for x in range(start_x, end_x, stride):
            crop = rgb[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] < tile_size // 4 or crop.shape[1] < tile_size // 4:
                continue
            if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            yield crop, x, y

def extract_species(czi_path):
    match = re.search(r'pol_pro_([A-Za-z]{3}_[A-Za-z]{3})', czi_path.name)
    if match:
        return match.group(1)
    parts = czi_path.resolve().parts
    skip = {"Source", "source", "Data", "data", "Raw", "raw", "Scans", "scans"}
    for part in reversed(parts[:-1]):
        if part not in skip and not part.startswith("/"):
            return part
    return czi_path.parent.name

def main():
    source_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source")
    out_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/preview_rgb_300")
    if not source_dir.exists():
        print("Source not found")
        return

    czis = sorted(list(source_dir.rglob("*.czi")))
    species_tiles = {}
    
    for czi_path in czis:
        species = extract_species(czi_path)

        if species_tiles.get(species, 0) >= 300:
            continue
            
        print(f"Reading {czi_path.name}...")
        local_path = Path("/tmp") / czi_path.name
        if not local_path.exists():
            print(f"Copying {czi_path.name} to local /tmp to speed up FUSE reads...")
            import shutil
            shutil.copyfile(czi_path, local_path)

        img = AICSImage(str(local_path))
        rgb = get_mip_rgb(img)
        
        # Cleanup local copy
        local_path.unlink()
        
        if rgb is None:
            print("Not native RGB, skipping")
            continue

        (out_dir / species).mkdir(parents=True, exist_ok=True)

        for tile, tx, ty in tile_image(rgb):
            if species_tiles.get(species, 0) >= 300:
                break
            
            stem = f"{species}_{czi_path.stem}_x{tx:06d}_y{ty:06d}.jpg"
            img_path = out_dir / species / stem
            cv2.imwrite(str(img_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            species_tiles[species] = species_tiles.get(species, 0) + 1
            
        print(f"Species {species} now has {species_tiles.get(species, 0)} tiles.")

if __name__ == "__main__":
    main()
