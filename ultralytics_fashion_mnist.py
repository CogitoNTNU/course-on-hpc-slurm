from ultralytics import YOLO
import wandb
import os

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
# Initialize your Weights & Biases environment
wandb.login(key=WANDB_API_KEY)

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
print(f"Training results: {results}")

metrics = model.val()
precision, recall, mAP50, mAP50_95, fitness = metrics.results_dict.values()
print(
    f"Precision: {precision}, Recall: {recall}, mAP50: {mAP50}, mAP50_95: {mAP50_95}, Fitness: {fitness}"
)

path = model.export(format="onnx")
print(f"Model exported to: {path}")
