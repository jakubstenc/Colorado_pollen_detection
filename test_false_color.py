import cv2
import numpy as np
from ultralytics import YOLO

rgb = cv2.imread("patch_raw.jpg")
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# Apply a false color map
# Bright background -> Blue (255, 0, 0 in BGR)
# Medium tissue -> Purple (128, 0, 128 in BGR)
# Dark pollen -> Dark Blue (100, 0, 0 in BGR)

false_color = np.zeros_like(rgb)

# Normalize gray to 0-1
gray_norm = gray.astype(np.float32) / 255.0

# B channel: high for background, low for pollen
false_color[:,:,0] = (gray_norm * 255).astype(np.uint8) 

# G channel: near 0
false_color[:,:,1] = 0

# R channel: high for tissue (mid intensities), low for background and pollen
# tissue is roughly around 100-180 intensity. Let's make a Gaussian bump for red
bump = np.exp(-0.5 * ((gray_norm - 0.5) / 0.2)**2)
false_color[:,:,2] = (bump * 128).astype(np.uint8)

cv2.imwrite("patch_false_color.jpg", false_color)

model = YOLO("models/general_pollen/latest.pt")
results = model(false_color, verbose=False, retina_masks=True, conf=0.1)
detections = []
if results[0].masks is not None and results[0].boxes is not None:
    for box in results[0].boxes:
        detections.append(box)

print(f"Total general pollen detected in false color patch: {len(detections)}")

# Draw them
overview = false_color.copy()
if results[0].masks is not None:
    for mask_xy in results[0].masks.xy:
        poly_px = np.array(mask_xy, dtype=np.int32)
        cv2.polylines(overview, [poly_px.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

cv2.imwrite("patch_false_color_detected.jpg", overview)

