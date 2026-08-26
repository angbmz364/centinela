"""
validate_custom_image.py

Classifies a single image using the trained model
and prints the prediction to the console.

Usage:
    python src/validate_custom_image.py path/to/image.jpg
"""

import sys

import torch
from torchvision import transforms
from PIL import Image

import model

# ----------------------------------------------------
# Class names (must match the training folder order)
# ----------------------------------------------------

CLASS_NAMES = ["fraterculus", "not_fraterculus"]

# ----------------------------------------------------
# Preprocessing (same as validation)
# ----------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------------------------------
# Select the device
# ----------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ----------------------------------------------------
# Load the trained weights
# ----------------------------------------------------

model.model.load_state_dict(
    torch.load(
        "models/centinela_stage1.pth",
        map_location=device,
        weights_only=True
    )
)

model.model.to(device)
model.model.eval()

# ----------------------------------------------------
# Read the image path from the command line
# ----------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python src/validate_custom_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# ----------------------------------------------------
# Open and preprocess the image
# ----------------------------------------------------

image = Image.open(image_path).convert("RGB")
tensor = preprocess(image).unsqueeze(0).to(device)
    
# ----------------------------------------------------
# Run the prediction
# ----------------------------------------------------

with torch.no_grad():
    output = model.model(tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_class = probabilities.max(dim=1)

# ----------------------------------------------------
# Print the result
# ----------------------------------------------------

class_name = CLASS_NAMES[predicted_class.item()]
confidence_pct = confidence.item() * 100

print(f"\nPrediction: {class_name}")
print(f"Confidence: {confidence_pct:.2f}%")
print(f"\nProbabilities:")
for i, name in enumerate(CLASS_NAMES):
    prob = probabilities[0][i].item() * 100
    print(f"  {name}: {prob:.2f}%")
