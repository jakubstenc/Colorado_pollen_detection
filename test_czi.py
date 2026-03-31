from aicsimageio import AICSImage
import numpy as np
import cv2

path = "/mnt/czi_data/20251031_001_pol_pro_Ran_ado_2025_32_Colorado2025.czi"
# If czi_data isn't mounted, maybe we look in /home/meow/cesnet_cloud/bucket/PEG/Colorado/Source/Ran_ado/
import os
alt_path = "/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source/Ran_ado/20251031_001_pol_pro_Ran_ado_2025_32_Colorado2025.czi"

target = path if os.path.exists(path) else alt_path
print(f"Reading {target}")

if not os.path.exists(target):
    # let's find a czi file
    import glob
    files = glob.glob("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Source/**/*.czi", recursive=True)
    if files:
        target = files[0]
        print(f"Found {target}")
    else:
        print("No czi found")
        exit()

img = AICSImage(target)
print("Dims:", img.dims)
print("Shape:", img.shape)
print("Dim order:", img.dims.order)

if 'S' in img.dims.order and img.dims.S == 3:
    print("RGB image detected!")
    # Try reading YXS
    data = img.get_image_data("YXS", T=0, C=0, Z=0)
    print("Data shape:", data.shape, "dtype:", data.dtype)
    print("Min:", data.min(), "Max:", data.max())
    
    # Save a tile to check
    tile = data[5000:5640, 5000:5640, :]
    cv2.imwrite("test_rgb_tile.jpg", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    print("Saved test_rgb_tile.jpg")
