"""
dataset.py

This file creates the PyTorch datasets and DataLoaders.
"""

# PyTorch
import torch

# TorchVision contains tools for Computer Vision
from torchvision import datasets
from torchvision import transforms

# DataLoader loads images in batches
from torch.utils.data import DataLoader

# Our configuration file
import config

# ---------------------------------------------------
# Training transformations
# ---------------------------------------------------

train_transform = transforms.Compose([

    # Resize every image to 224x224
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),

    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # Random rotation
    transforms.RandomRotation(15),

    # Small brightness/contrast changes
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    # Convert image into a Tensor
    transforms.ToTensor(),

])

# Without rotation or augmentation because model needs to see the reality
val_transform = transforms.Compose([

    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),

    transforms.ToTensor()

])


train_dataset = datasets.ImageFolder(

    root=config.TRAIN_PATH,

    transform=train_transform

)

val_dataset = datasets.ImageFolder(

    root=config.VAL_PATH,

    transform=val_transform

)

test_dataset = datasets.ImageFolder(

    root=config.TEST_PATH,

    transform=val_transform

)

train_loader = DataLoader(

    train_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=True

)

val_loader = DataLoader(
  
    val_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=False
)

test_loader = DataLoader(
  
    test_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=False

)

# ---------------------------------------------------
# Test one batch
# ---------------------------------------------------

# Get one batch from the training loader
images, labels = next(iter(train_loader))

# Print information about the batch
"""
print("\nFirst batch")

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

print("Labels:", labels)
print(train_dataset.class_to_idx)

"""