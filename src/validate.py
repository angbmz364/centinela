"""
validate.py

Evalúa el modelo entrenado sobre el conjunto de validación,
imprime un informe detallado de métricas en español y lo guarda
en logs/latest.txt (rotando el archivo anterior con su fecha).
"""

import torch

import config
import dataset
import model

from evaluate import evaluate
from utils import capture_output_to_log, log_detailed_metrics

# ----------------------------------------------------
# Seleccionar el dispositivo
# ----------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ----------------------------------------------------
# Cargar los pesos entrenados
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
# Evaluar el modelo y guardar el informe
# ----------------------------------------------------

with capture_output_to_log():

    print("=" * 74)
    print("          VALIDACIÓN DEL MODELO — CONJUNTO DE VALIDACIÓN")
    print("=" * 74)

    print(f"\nUsando dispositivo: {device}")
    print(f"Clases: {dataset.val_dataset.classes}")

    validation_loss, validation_accuracy, predictions, labels, probabilities, confidences = evaluate(
        model.model,
        dataset.val_loader,
        device
    )

    print(f"\nPérdida de validación (Loss): {validation_loss:.4f}")
    print(f"Exactitud de validación:      {validation_accuracy:.2f}%")

    log_detailed_metrics(
        labels,
        predictions,
        probabilities,
        confidences,
        dataset.val_dataset.classes,
        dataset_name="validación",
    )