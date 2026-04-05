from ultralytics import YOLO

# Resume training from the previous general model weights
model = YOLO('models/general_pollen/latest.pt')

# Train on the new amalgamated active learning dataset
results = model.train(
    data='dataset_general_v2/data.yaml',
    epochs=150,     # Capped at 150 since it is pre-trained
    imgsz=640,
    batch=16,
    project='/app/results/general_pollen_v2',
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
    patience=50     # Early stopping
)

print("Training completed! Running evaluation on the test dataset...")
model.val(
    data='dataset_general_v2/data.yaml',
    split='test',
    project='/app/results/general_pollen_v2',
    name='test_run'
)
print("Test evaluation completed.")
