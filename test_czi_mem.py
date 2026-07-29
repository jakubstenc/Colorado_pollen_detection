from aicsimageio import AICSImage
import psutil
import os

img = AICSImage("/tmp/20260702_142_Dep_Ran_Ado_20_7_146b_Colorado2025.czi")
print(img.dims)
print("Dask shape:", img.dask_data.shape)
print("Mem usage:", psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB")
