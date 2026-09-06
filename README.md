# Centinela

An AI model trained to identify the South American fruit fly (*Anastrepha fraterculus*) using a **MobileNetV3-small** backbone adapted with **Transfer Learning**.

Centinela is designed as an edge-AI solution: images are captured with a camera and analyzed locally, without requiring an Internet connection. The repository contains everything needed to prepare the dataset, train the model, and evaluate it on validation, test, and custom images.

Two classes are supported:

| Class | Meaning |
|---|---|
| `fraterculus` | *Anastrepha fraterculus* (target pest) |
| `not_fraterculus` | Anything else |

---

## Architecture

```txt
/
  datasets/
    raw/          # Read-only source datasets (Dryad + IP102)
    processed/    # Datasets split into train/val/test (copy of raw, organized)
    custom/       # Hand-picked images for quick manual checks
  models/
    centinela_stage1.pth   # Saved trained model
  reports/
    # School-project documentation (EUREKA 2026)
  src/
    config.py               # Paths and hyperparameters
    prepare_dataset.py      # Raw -> processed split
    dataset.py              # PyTorch datasets and DataLoaders
    model.py                # MobileNetV3-small + transfer-learning setup
    train.py                # One training epoch + save model
    validate.py             # Loss/accuracy on the validation split
    test_evaluate.py        # Full metrics on the test split
    validate_custom_image.py# Predict a single image
    evaluate.py             # Evaluation loop shared by validate / test
  requirements.txt
```

> Note: processed data lives under `datasets/processed` (the `outputs/` folder mentioned in earlier drafts does not exist).

---

## Pipeline

```txt
prepare_dataset.py → train.py → validate.py → test_evaluate.py → validate_custom_image.py
       (split)          (train)      (val metrics)   (test metrics)      (single image)
```

1. **Prepare** — split the raw images into `train/val/test` in a reproducible way.
2. **Train** — adapt `MobileNetV3-small` with a frozen backbone for 2 classes.
3. **Validate** — quick loss/accuracy check on the validation split.
4. **Evaluate** — accuracy, precision, recall, F1 and confusion matrix on the test split.
5. **Classify** — run the model on any single image.

---

## Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 1. Prepare the dataset

```bash
python src/prepare_dataset.py
```

Uses `random.seed(42)`, so the exact same split is produced every time (reproducible experiments).

### 2. Train the model

```bash
python src/train.py
```

Trains one epoch on the frozen-backbone classifier and saves the weights to `models/centinela_stage1.pth`. If a GPU is available it is used automatically, otherwise it falls back to CPU.

### 3. Validate

```bash
python src/validate.py
```

Prints validation loss and accuracy.

### 4. Evaluate on the test split (full metrics)

```bash
python src/test_evaluate.py
```

Prints test loss, accuracy, per-class and macro precision / recall / F1, plus the confusion matrix.

### 5. Classify a single image

```bash
python src/validate_custom_image.py path/to/image.jpg
```

---

## Dataset

The training data combines two public sources:

| Source | Usage | Images |
|---|---|---|
| **Dryad — A. fraterculus** | Positive class (`fraterculus`) | 2122 |
| **IP102** (classes 41, 55, 72, 84, 85, 86) | Negative class (`not_fraterculus`) | 1460 |

Total: **3582 images**, split 70 / 15 / 15 with `random.seed(42)`:

| Split | `fraterculus` | `not_fraterculus` | Total |
|---|---|---|---|
| Train | 1485 | 1021 | 2506 |
| Validation | 318 | 219 | 537 |
| Test | 319 | 220 | 539 |
| **Total** | **2122** | **1460** | **3582** |

Input images are resized to **224 × 224**. Training applies horizontal flip, rotation (±15°) and color jitter for augmentation; validation and test use only resizing (the model must see reality undistorted while being measured).

---

## Results (measured on this repository)

Metrics below were produced by running the scripts above (1 training epoch, batch size 32, Adam, `lr=0.001`, CrossEntropyLoss, CPU).

### Training

| Run | Average loss |
|---|---|
| Epoch 1 | 0.0491 |

### Validation split

| Metric | Value |
|---|---|
| Loss | 0.0267 |
| Accuracy | **98.88 %** |

### Test split

| Metric | `fraterculus` | `not_fraterculus` | Macro |
|---|---|---|---|
| Precision | 97.26 % | 100.00 % | 98.63 % |
| Recall | 100.00 % | 95.91 % | 97.95 % |
| F1-score | 98.61 % | 97.91 % | 98.26 % |
| Accuracy | — | — | **98.33 %** |

Test loss: **0.0324** · Test images: **539**

#### Confusion matrix (rows = actual, columns = predicted)

| Actual \ Predicted | `fraterculus` | `not_fraterculus` |
|---|---|---|
| `fraterculus` | **319** | 0 |
| `not_fraterculus` | 9 | **211** |

Interpretation: the model never missed a real *A. fraterculus* (0 false negatives) and flagged 9 non-target images as pests (false positives).

### Custom images

Manual predictions on `datasets/custom/` (unlabeled test images — shown as-is):

| Image | Prediction | Confidence |
|---|---|---|
| `bonito.webp` | `not_fraterculus` | 99.74 % |
| `mosca1.jpg` | `not_fraterculus` | 87.89 % |
| `mosca2.jpg` | `not_fraterculus` | 99.74 % |
| `mosca_fruta.jpg` | `fraterculus` | 91.50 % |
| `oruga.jpg` | `not_fraterculus` | 99.76 % |
| `prueba.jpeg` | `not_fraterculus` | 100.00 % |
| `vasco.jpeg` | `not_fraterculus` | 73.54 % |

---

## Configuration

Key settings live in `src/config.py`:

| Setting | Value |
|---|---|
| `IMAGE_SIZE` | 224 |
| `BATCH_SIZE` | 32 |
| `NUM_CLASSES` | 2 |
| `EPOCHS` | 30 (the train script currently runs 1 epoch) |
| `LEARNING_RATE` | 0.001 |

---

## Dependencies

* torch
* torchvision
* numpy
* opencv-python
* matplotlib
* scikit-learn
* pandas
* tqdm
* Pillow

---

## Notes and limitations

- Reported results reflect the current checkpoint (1 epoch, frozen backbone). Training longer or fine-tuning deeper layers can improve them.
- High test accuracy does not guarantee perfect performance in the field: lighting, angle, distance and background can change results, as seen in the custom-image section.
- Treat a detection as a monitoring signal: verify before applying any agricultural measure.