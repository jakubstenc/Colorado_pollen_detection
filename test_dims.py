import sys
import urllib3
urllib3.disable_warnings()
from aicsimageio import AICSImage
img = AICSImage('/tmp/20260701_001_Dep_Ran_Ado_24_7_161b_Colorado2025.czi')
print("Dims:", img.dims.order)
print("Shape:", img.shape)
if 'S' in img.dims.order:
    print("S size:", getattr(img.dims, 'S', 1))
