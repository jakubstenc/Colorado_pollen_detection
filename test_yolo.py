from ultralytics import YOLO
import cv2

model = YOLO('models/general_pollen/latest.pt')

for img_path in ['mip_raw.jpg', 'mip_swapped.jpg']:
    try:
        img = cv2.imread(img_path)
        if img is None: continue
        res = model(img, verbose=False, conf=0.25)
        print(f"Results for {img_path}: {len(res[0].boxes)} detections")
    except Exception as e:
        print(e)
