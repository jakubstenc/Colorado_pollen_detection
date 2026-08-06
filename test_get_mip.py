from aicsimageio import AICSImage
from src.build_species_dataset import get_mip_rgb

def test():
    img = AICSImage("test.czi")
    print(f"Shape: {img.shape}")
    print(f"Dims: {img.dims}")
    try:
        rgb = get_mip_rgb(img, grayscale=False)
        print(f"RGB shape: {rgb.shape}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
