from aicsimageio import AICSImage
import numpy as np
img = AICSImage('test.czi')
data = img.get_image_data("YXS", T=0, C=0, Z=0)
H, W = data.shape[:2]
patch = data[H//2:H//2+1000, W//2:W//2+1000]
print("Patch mean:", patch.mean(axis=(0,1)))
