import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip3", "install", package])

# List of required packages
required_packages = [
    "argparse",
    "torch",
    "torchvision",
    "PIL",
    "numpy",
    "glob"
]

# Install each package if not already installed
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import argparse
import torch as th
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from glob import glob

# Set up device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Load class names
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(class_names)

# Load the trained model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(th.load("garbage_classification_modelL2ES.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Loads an image, preprocesses it, and predicts its class."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = data_transforms(image).unsqueeze(0).to(device)

    with th.no_grad():
        output = model(image_tensor)
        _, predicted = th.max(output, 1)

    return class_names[predicted.item()]

def batch_inference(folder, output_folder):
    """Runs inference on all images in the given folder and saves results."""
    image_paths = glob(os.path.join(folder, "*"))
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_paths:
        print("No images found in the specified folder.")
        return

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "predictions.txt")
    with open(output_file, "w") as f:
        for img_path in image_paths:
            prediction = predict_image(img_path)
            image_name = os.path.basename(img_path)
            f.write(f"{image_name}: {prediction}\n")
            print(f"Image: {image_name} → Predicted: {prediction}")

    print(f"\n✅ Predictions saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a folder of images and save results.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder for predictions.")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"❌ Error: {args.dir} is not a valid directory.")
    else:
        batch_inference(args.dir, args.output)
