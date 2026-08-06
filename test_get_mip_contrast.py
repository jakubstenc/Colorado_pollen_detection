import numpy as np
from aicsimageio import AICSImage
from src.build_species_dataset import get_mip_rgb, normalize_to_uint8

def test():
    img = AICSImage("test.czi")
    print("Loading raw...")
    d = np.squeeze(img.data)
    if len(d.shape) == 3 and d.shape[0] == 3:
        d = np.transpose(d, (1, 2, 0))
    print(f"Raw shape: {d.shape}, dtype: {d.dtype}, min: {d.min()}, max: {d.max()}")
    
    rgb_no_norm = get_mip_rgb(img, grayscale=False)
    print(f"get_mip_rgb output dtype: {rgb_no_norm.dtype}, min: {rgb_no_norm.min()}, max: {rgb_no_norm.max()}")
    
    # What it used to do
    rgb_norm = normalize_to_uint8(d)
    print(f"normalize_to_uint8 output dtype: {rgb_norm.dtype}, min: {rgb_norm.min()}, max: {rgb_norm.max()}")

if __name__ == "__main__":
    test()
