"""
test_evaluate.py

Evaluates the trained model on the test dataset and prints
accuracy, precision, recall, F1-score and the confusion matrix.

Usage:
    python src/test_evaluate.py
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import config
import dataset
import model

from evaluate import evaluate

# ----------------------------------------------------
# Class order (must match the ImageFolder folder order)
# ImageFolder sorts classes alphabetically.
# ----------------------------------------------------

CLASS_NAMES = dataset.test_dataset.classes

print("Classes:", CLASS_NAMES)
print("Class to index:", dataset.test_dataset.class_to_idx)

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
        config.MODEL_PATH / "centinela_stage1.pth",
        map_location=device,
        weights_only=True
    )
)

model.model.to(device)

# ----------------------------------------------------
# Evaluate the model on the test set
# ----------------------------------------------------

test_loss, test_accuracy, predictions, labels = evaluate(
    model.model,
    dataset.test_loader,
    device
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# ----------------------------------------------------
# Precision, Recall, F1-score
# ----------------------------------------------------

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    predictions,
    labels=[0, 1],
    zero_division=0
)

for i, name in enumerate(CLASS_NAMES):
    print(f"\n{name}:")
    print(f"  Precision: {precision[i] * 100:.2f}%")
    print(f"  Recall:    {recall[i] * 100:.2f}%")
    print(f"  F1-score:  {f1[i] * 100:.2f}%")

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    labels,
    predictions,
    average="macro",
    zero_division=0
)

print("\nMacro:")
print(f"  Precision: {macro_precision * 100:.2f}%")
print(f"  Recall:    {macro_recall * 100:.2f}%")
print(f"  F1-score:  {macro_f1 * 100:.2f}%")

# ----------------------------------------------------
# Confusion matrix
# ----------------------------------------------------

print("\nConfusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(labels, predictions))

print("\nClassification Report:")
print(classification_report(
    labels,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))

# ----------------------------------------------------
# Number of test images (used later for the README)
# ----------------------------------------------------

print(f"\nTotal test images: {len(labels)}")