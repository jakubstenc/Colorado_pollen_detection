import os
import cv2
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from ultralytics import YOLO
import shutil
import random

def extract_species(czi_path):
    if 'Ran_ado' in czi_path.name: return 'Ran_ado'
    if 'Vio_adu' in czi_path.name: return 'Vio_adu'
    return 'Unknown'

def get_mip_rgb(img):
    if getattr(img.dims, 'S', 1) == 3:
        try:
            return img.get_image_data("YXS", T=0, Z=0, C=0)
        except Exception:
            return img.get_image_data("YXS", T=0, Z=0)
    return None

def tile_image_random(rgb, tile_size=640):
    H, W = rgb.shape[:2]
    stride = tile_size
    coords = []
    
    # Generate non-overlapping grid to avoid duplicates
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            coords.append((x, y))
            
    # Shuffle to get diverse negatives from all parts of the slide
    random.shuffle(coords)
    
    for x, y in coords:
        crop = rgb[y:y + tile_size, x:x + tile_size]
        # Ignore tiny edge slivers entirely
        if crop.shape[0] < tile_size // 2 or crop.shape[1] < tile_size // 2:
            continue
            
        if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
            
        yield crop, x, y

def main():
    source_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source")
    out_dir = Path("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Staged_negatives")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading YOLO model to verify negatives...")
    model = YOLO("best.pt")
    
    czis = sorted(list(source_dir.rglob("*.czi")))
    species_targets = {'Ran_ado': 150, 'Vio_adu': 150} # 300 total
    species_counts = {'Ran_ado': 0, 'Vio_adu': 0}
    
    for czi_path in czis:
        species = extract_species(czi_path)
        if species not in species_targets:
            continue
            
        if species_counts[species] >= species_targets[species]:
            continue
            
        print(f"\nReading {czi_path.name}...")
        local_path = Path("/tmp") / czi_path.name
        if not local_path.exists():
            print(f"Copying to local /tmp...")
            shutil.copyfile(czi_path, local_path)

        img = AICSImage(str(local_path))
        rgb = get_mip_rgb(img)
        local_path.unlink()
        
        if rgb is None:
            continue

        for crop, tx, ty in tile_image_random(rgb):
            if species_counts[species] >= species_targets[species]:
                break
                
            # Run YOLO strict to ensure it's completely empty of pollen
            results = model(crop, conf=0.15, verbose=False)
            
            # Criteria for negative: 0 detections, and not pure black padding
            if len(results[0].boxes) == 0 and (results[0].masks is None or len(results[0].masks) == 0):
                if np.mean(crop) > 20:  # Exclude areas that are completely black/empty scanner background
                    out_path = out_dir / f"negative_{czi_path.stem}_x{tx:06d}_y{ty:06d}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    species_counts[species] += 1
                    total = sum(species_counts.values())
                    print(f"Collected negative {total}/300 [{species}: {species_counts[species]}/{species_targets[species]}]", end='\r')

    print(f"\n✨ Done! Negatives saved to {out_dir}")

if __name__ == '__main__':
    main()
