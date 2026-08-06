import sys
import warnings
warnings.filterwarnings('ignore')
from aicsimageio import AICSImage
img = AICSImage('/tmp/20260701_001_Dep_Ran_Ado_24_7_161b_Colorado2025.czi')
print('Dims:', img.dims.order)
print('Shape:', img.shape)
print('Data shape:', img.data.shape)
