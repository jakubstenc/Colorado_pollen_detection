import cv2
import numpy as np

# Let's create a red image (RGB: 255, 0, 0)
img_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
img_rgb[:, :, 0] = 255 # Red

# OpenCV imwrite expects BGR. So the 0th channel (Red) will be written as Blue byte.
# So the saved image will be physically BLUE.
cv2.imwrite("test_red.jpg", img_rgb)

# Now what if I do cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)?
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
# Now img_bgr has 255 at channel 2 (Red).
cv2.imwrite("test_red_cvt.jpg", img_bgr)
# When imwrite writes img_bgr, channel 2 is written as Red byte.
# So the saved image is physically RED.
