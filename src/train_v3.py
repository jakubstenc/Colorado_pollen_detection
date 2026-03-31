from ultralytics import YOLO

# Initialize the model from the existing best.pt weights
model = YOLO('/app/best.pt')

results = model.train(
    data='/app/dataset_v3/data.yaml',
    epochs=300, # Increased to 500 for the new training run
    imgsz=640,
    batch=16,
    project='/app/results/general_pollen_v3',
    name='train_run1',
)

print("Training completed! The new V3 model is saved.")
