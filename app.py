from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(class_names)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load("garbage_classification_modelL2ES.pth", map_location=device))
model.to(device)
model.eval()

# Image transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Handle multiple file uploads
@app.route("/", methods=["GET", "POST"])
def upload_files():
    if request.method == "POST":
        files = request.files.getlist("files")  # Get multiple files
        predictions = {}

        for file in files:
            if file:
                image = Image.open(file).convert("RGB")
                prediction = predict_image(image)
                predictions[file.filename] = prediction

        return jsonify({"predictions": predictions})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
