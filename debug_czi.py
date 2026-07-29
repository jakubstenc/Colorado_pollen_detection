from aicsimageio import AICSImage
import numpy as np
img = AICSImage("/tmp/test.czi")
rgb = img.get_image_data("YXS", T=0, C=0, Z=0)
print("RGB dtype:", rgb.dtype, "max:", rgb.max(), "mean:", rgb.mean(), "median:", np.median(rgb))
print("Percentiles:", np.percentile(rgb, [0.5, 95, 99.5, 99.9]))
