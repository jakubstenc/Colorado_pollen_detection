from aicsimageio import AICSImage
import cv2
img = AICSImage('test.czi')
# get_image_data returns (Y, X, S)
data = img.get_image_data("YXS", T=0, C=0, Z=0)
# Let's get a 1000x1000 patch from the middle
H, W = data.shape[:2]
patch = data[H//2:H//2+1000, W//2:W//2+1000]

# Write it directly with cv2
cv2.imwrite("patch_raw.jpg", patch)
cv2.imwrite("patch_swapped.jpg", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
