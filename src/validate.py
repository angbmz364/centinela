"""
validate.py

Evaluates the trained model on the validation dataset.
"""

import torch

import dataset
import model

from evaluate import evaluate

# ----------------------------------------------------
# Select the device
# ----------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

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

# ----------------------------------------------------
# Evaluate the model
# ----------------------------------------------------

validation_loss, validation_accuracy, predictions, labels = evaluate(
    model.model,
    dataset.val_loader,
    device
)

print(f"\nValidation Loss: {validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.2f}%")