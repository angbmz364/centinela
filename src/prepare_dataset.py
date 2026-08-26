# Used to navigate through folders
from pathlib import Path

# Used to copy files
import shutil

# Used to randomly shuffle images
import random

# ----------------------------------------------------
# Copies a list of images into a destination folder.
# ----------------------------------------------------

def copy_images(image_list, destination_folder):
    """
    Copies every image from image_list into destination_folder.

    Parameters:
        image_list: List of image paths.
        destination_folder: Folder where images will be copied.
    """

    # Go through every image in the list.
    for image in image_list:

        # Copy the file.
        shutil.copy(image, destination_folder)

# ----------------------------------------------------
# IMPORTANT
#
# If we use the same random seed every time,
# we always obtain the EXACT same split.
#
# This makes our experiments reproducible.
# ----------------------------------------------------

random.seed(42)

# ----------------------------------------------------
# Dataset paths
#
# Path() simply creates an object that points
# to a folder on your computer.
# ----------------------------------------------------

DRYAD_PATH = Path("datasets/raw/Dryad")

IP102_IMAGES_PATH = Path("datasets/raw/IP102/ip102_v1.1/images")

IP102_ANNOTATIONS = Path("datasets/raw/IP102/ip102_v1.1/train.txt")

OUTPUT_PATH = Path("datasets/processed")

# ----------------------------------------------------
# Create the folders if they don't exist.
#
# exist_ok=True means:
# "Don't crash if the folder already exists."
# ----------------------------------------------------

for split in ["train", "val", "test"]:

    for class_name in ["fraterculus", "not_fraterculus"]:

        folder = OUTPUT_PATH / split / class_name

        folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Read every image from the Dryad dataset
# ----------------------------------------------------

# Create an empty list.
# We'll store every image path inside this list.
positive_images = []

# Search for every JPG image inside the Dryad folder.
# "*.jpg" means:
# "Give me every file that ends with .jpg"
for image in DRYAD_PATH.glob("*.jpg"):

    # Add the image path to the list.
    positive_images.append(image)

# Print how many images we found.
print(f"Positive images found: {len(positive_images)}")

# ----------------------------------------------------
# Negative classes
#
# These are the classes we want to use
# from IP102.
# ----------------------------------------------------

NEGATIVE_CLASSES = [
    "41",
    "55",
    "72",
    "84",
    "85",
    "86"
]

# ----------------------------------------------------
# Read every negative image
# ----------------------------------------------------

negative_images = []

# Read the annotation file and select
# images that belong to our target classes.
with open(IP102_ANNOTATIONS, "r") as f:

    for line in f:

        filename, class_id = line.strip().split()

        if class_id in NEGATIVE_CLASSES:

            image_path = IP102_IMAGES_PATH / filename

            negative_images.append(image_path)

print(f"Negative images found: {len(negative_images)}")

# ----------------------------------------------------
# Shuffle the images
#
# This randomizes their order.
# ----------------------------------------------------

random.shuffle(positive_images)
random.shuffle(negative_images)

""""
print()

print("First positive image:")
print(positive_images[0])

print()

print("First negative image:")
print(negative_images[0])
"""

# ----------------------------------------------------
# Dataset split percentages
#
# We store these values in variables so that
# changing the split later is easy.
# ----------------------------------------------------

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ----------------------------------------------------
# Calculate the indexes where each split ends.
#
# Example:
#
# 100 images
#
# train_end = 70
#
# val_end = 85
# ----------------------------------------------------

positive_train_end = int(len(positive_images) * TRAIN_SPLIT)

positive_val_end = positive_train_end + int(len(positive_images) * VAL_SPLIT)

negative_train_end = int(len(negative_images) * TRAIN_SPLIT)

negative_val_end = negative_train_end + int(len(negative_images) * VAL_SPLIT)

# ----------------------------------------------------
# Positive dataset
# ----------------------------------------------------

positive_train = positive_images[:positive_train_end]

positive_val = positive_images[positive_train_end:positive_val_end]

positive_test = positive_images[positive_val_end:]

# ----------------------------------------------------
# Negative dataset
# ----------------------------------------------------

negative_train = negative_images[:negative_train_end]

negative_val = negative_images[negative_train_end:negative_val_end]

negative_test = negative_images[negative_val_end:]

print()

print("Positive split")

print("Train:", len(positive_train))
print("Validation:", len(positive_val))
print("Test:", len(positive_test))

print()

print("Negative split")

print("Train:", len(negative_train))
print("Validation:", len(negative_val))
print("Test:", len(negative_test))

# Positive images
copy_images(
    positive_train,
    OUTPUT_PATH / "train" / "fraterculus"
)

copy_images(
    positive_val,
    OUTPUT_PATH / "val" / "fraterculus"
)

copy_images(
    positive_test,
    OUTPUT_PATH / "test" / "fraterculus"
)

# Negative images
copy_images(
    negative_train,
    OUTPUT_PATH / "train" / "not_fraterculus"
)

copy_images(
    negative_val,
    OUTPUT_PATH / "val" / "not_fraterculus"
)

copy_images(
    negative_test,
    OUTPUT_PATH / "test" / "not_fraterculus"
)

print()
print("Dataset successfully prepared! 🎉")