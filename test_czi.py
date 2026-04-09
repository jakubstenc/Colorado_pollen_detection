import traceback
import sys
from pathlib import Path
from src.build_species_dataset import AICSImage, get_mip_rgb, tile_image

local_path = "test.czi"

print(f"Testing extraction natively on {local_path} ...")
try:
    img = AICSImage(local_path)
    print("AICSImage loaded, dimensions:", img.dims.order)
    
    rgb = get_mip_rgb(img)
    print("MIP RGB generated, shape:", rgb.shape)
    
    for tile, tx, ty in tile_image(rgb):
        print("Tiled!", tile.shape)
        break
        
except Exception as e:
    print(f"CRASH:")
    traceback.print_exc()
