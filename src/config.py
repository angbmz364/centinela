from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_CLASSES = 2

# Training

EPOCHS = 30

LEARNING_RATE = 0.001

# Paths

TRAIN_PATH = PROJECT_ROOT / "datasets/processed/train"

VAL_PATH = PROJECT_ROOT / "datasets/processed/val"

TEST_PATH = PROJECT_ROOT / "datasets/processed/test"

MODEL_PATH = PROJECT_ROOT / "models"