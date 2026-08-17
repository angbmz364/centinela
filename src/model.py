"""
model.py

Creates our AI model.
"""

# PyTorch - nn means neutral network
import torch.nn as nn 

# TorchVision models
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# ----------------------------------------------------
# Load pretrained MobileNetV3
# ----------------------------------------------------

model = mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.DEFAULT
)

# ----------------------------------------------------
# Freeze every parameter in the model.
#
# This means the model will still perform predictions,
# but its weights won't be updated during training.
# ----------------------------------------------------

for parameter in model.parameters():
    parameter.requires_grad = False

# ----------------------------------------------------
# Replace the last classification layer
# ----------------------------------------------------

model.classifier[3] = nn.Linear(
    in_features=1024,
    out_features=2
)

# ----------------------------------------------------
# Unfreeze the classifier.
#
# These are the only parameters that will learn during
# the first stage of training.
# ----------------------------------------------------

for parameter in model.classifier.parameters():
    parameter.requires_grad = True

#TEST
print("\nTrainable parameters:\n")

for name, parameter in model.named_parameters():
    print(f"{name}: {parameter.requires_grad}")