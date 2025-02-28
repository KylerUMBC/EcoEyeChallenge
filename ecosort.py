import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import subprocess
import sys
import os

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        print("Ensure 'pip' is installed and available in the system PATH.")
        print("You may need to run:")
        print(f"  {sys.executable} -m ensurepip --upgrade")
        print(f"  {sys.executable} -m pip install --upgrade pip")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Python executable not found or pip not callable")
        sys.exit(1)


# List of required packages
required_packages = [
    "torch",
    "torchvision",
    "Pillow",
    "numpy",
]

# Install each package if not already installed
for package in required_packages:
    try:
        if(package == "Pillow"):
            __import__("PIL")
        else:
            __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install(package)

import argparse
import torch as th
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import PIL.ImageTk as ImageTk
import numpy as np
import tkinter as tk
from glob import glob   

# Set up device
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Load class names
class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
num_classes = len(class_names)

#Try to load the trained model
try:
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(th.load("garbage_classification_modelL2ES.pth", map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print("❌ Error: Model file not found. Please ensure 'garbage_classification_modelL2ES.pth' is in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Define image transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Tries to predict an image
def predict_image(image_path):
    """Loads an image, preprocesses it, and predicts its class."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = data_transforms(image).unsqueeze(0).to(device)

        with th.no_grad():
            output = model(image_tensor)
            _, predicted = th.max(output, 1)

        return class_names[predicted.item()]
    except Exception as e:
        return f"❌ Error processing image {image_path}: {e}"
    
#Tries to predict a batch of images (better performance)
def batch_predict(images):
    try:
        image_tensors = th.stack([data_transforms(Image.open(img).convert("RGB")) for img in images]).to(device)
        with th.no_grad():
            outputs = model(image_tensors)
            _, preds = th.max(outputs, 1)
            return [class_names[pred.item()] for pred in preds]
    except Exception as e:
        return [f"❌ Error {e}"] * len(images)

def batch_inference(folder, output_folder=None, print_preds=False):
    """Runs inference on all images in the given folder and saves results."""
    image_paths = glob(os.path.join(folder, "*"))
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not image_paths:
        print("No images found in the specified folder.")
        return

    batch_size = 16
    predictions = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_preds = batch_predict(batch_paths)
        predictions.extend(zip(batch_paths, batch_preds))

    if print_preds:
        for img_path, pred in predictions:
            print(f"Image: {os.path.basename(img_path)} → Predicted: {pred}")

    # Ensure output directory exists
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(output_folder, "predictions.txt")
        with open(output_file, "w") as f:
            for img_path in image_paths:
                prediction = predict_image(img_path)
                image_name = os.path.basename(img_path)
                f.write(f"{image_name}: {prediction}\n")
                print(f"Image: {image_name} → Predicted: {prediction}")
        print(f"\n✅ Predictions saved to: {output_file}")

        #Save to CSV file
        csv_file = os.path.join(output_folder, "predictions.csv")
        with open(csv_file, "w") as f:
            f.write("Image,Prediction\n")
            for img_path, pred in predictions:
                image_name = os.path.basename(img_path)
                f.write(f"{image_name},{pred}\n")
        print(f"✅ Predictions saved to: {csv_file}")
    return predictions #We need a return for GUI use

'''Gui functionality'''
def run_gui(folder):
    #Gets the images in the folder
    image_paths = [p for p in glob(os.path.join(folder, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        print("No images found in the specified folder.")
        return

    #Gethering the predictions
    predictions = batch_inference(folder)
    current_idx = [0] #Navigation index
    
    root = tk.Tk()
    root.title("EcoSort Image Classifier")
    root.geometry("600x650")

    img_label = tk.Label(root)
    img_label.pack(pady=10)

    pred_label = tk.Label(root, text="", font=("Arial", 20))
    pred_label.pack(pady=10)

    #Update the display with the current image and prediction
    #Retrieves the image path and prediction for the current index
    #Resizes and updates the image label with the new PhotoImage object
    def update_display():
        img_path, pred = predictions[current_idx[0]]
        img = Image.open(img_path).resize((400, 400), Image.LANCZOS) #Using Lanczos for better quality
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        img_label.image = photo #Keeps a reference to the image
        pred_label.config(text=f"Image: {os.path.basename(img_path)}\nPredicted: {pred}")
        root.title(f"EcoSort Classification Viewer ({current_idx[0] + 1}/{len(predictions)})")

    #Navigation buttons to move to the previous or next image
    def prev_image():
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_display()

    def next_image():
        if current_idx[0] < len(image_paths) - 1:
            current_idx[0] += 1
            update_display()
    
    #Bind keys to the navigation functions (they pass the event object not call) 
    root.bind("<Left>", prev_image)
    root.bind("<Right>", next_image)

    update_display()
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Previous", command=prev_image).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Next", command=next_image).pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a folder of images and save results.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output", type=str, help="Path to the output folder for predictions.")
    parser.add_argument("-p", "--print", action="store_true", help="Print the predictions to the console.")
    parser.add_argument("--gui", action="store_true", help="Force GUI mode to view image predictions.")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"❌ Error: {args.dir} is not a valid directory.")
    elif os.path.isfile(args.dir):
        pred = predict_image(args.dir)
        #print(f"Image: {os.path.basename(args.dir)} → Predicted: {pred}")
    elif args.gui or (not args.output and not args.print):
        run_gui(args.dir)
    elif args.output:
        batch_inference(args.dir, args.output, args.print)
    else:
        batch_inference(args.dir, print_preds=args.print)
