from aicsimageio import AICSImage
import numpy as np
import cv2

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

img = AICSImage('test.czi')
dask_czyx = img.get_image_dask_data("CZYX")
num_c = dask_czyx.shape[0]

# test.czi has num_c = 1, but spatial dims ending in 3.
# Let's see what it does.
target_channels = [c for c in [0, 1, 2] if c < num_c]
channels_data = []
for c in target_channels:
    mip = dask_czyx[c].max(axis=0).compute()
    if len(mip.shape) == 3 and mip.shape[-1] == 3:
        # The underlying reader returned an RGB array directly
        res = normalize_to_uint8(mip)
        channels_data.append(res)
        break
    else:
        channels_data.append(normalize_to_uint8(mip))

if len(channels_data) == 1 and len(channels_data[0].shape) == 3:
    rgb = channels_data[0]
else:
    rgb = np.stack(channels_data, axis=-1)

# Write it directly to see how it looks
cv2.imwrite("mip_raw.jpg", rgb)
cv2.imwrite("mip_swapped.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
