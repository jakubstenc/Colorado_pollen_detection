from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset_general_v1/data.yaml',
    epochs=500,
    imgsz=640,
    batch=16,
    project='/app/results/general_pollen_v1',
    name='train_run1',
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    patience=100
)

print("Training completed! Running evaluation on the test dataset...")
model.val(
    data='dataset_general_v1/data.yaml',
    split='test',
    project='/app/results/general_pollen_v1',
    name='test_run'
)
print("Test evaluation completed.")
