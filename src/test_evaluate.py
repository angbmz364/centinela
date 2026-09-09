"""
test_evaluate.py

Evalúa el modelo entrenado sobre el conjunto de prueba, imprime
un informe detallado de métricas en español y lo guarda en
logs/latest.txt (rotando el archivo anterior con su fecha).

Uso:
    python src/test_evaluate.py
"""

import torch

import config
import dataset
import model

from evaluate import evaluate
from utils import capture_output_to_log, log_detailed_metrics

# ----------------------------------------------------
# Orden de las clases (debe coincidir con ImageFolder)
# ----------------------------------------------------

CLASS_NAMES = dataset.test_dataset.classes

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
# Evaluar el modelo en el conjunto de prueba
# ----------------------------------------------------

with capture_output_to_log():

    print("=" * 74)
    print("           EVALUACIÓN DEL MODELO — CONJUNTO DE PRUEBA")
    print("=" * 74)

    print(f"\nUsando dispositivo: {device}")
    print("Clases:", CLASS_NAMES)
    print("Índice de clases:", dataset.test_dataset.class_to_idx)

    test_loss, test_accuracy, predictions, labels, probabilities, confidences = evaluate(
        model.model,
        dataset.test_loader,
        device
    )

    print(f"\nPérdida de prueba (Loss): {test_loss:.4f}")
    print(f"Exactitud de prueba:      {test_accuracy:.2f}%")
    print(f"Total de imágenes de prueba: {len(labels)}")

    log_detailed_metrics(
        labels,
        predictions,
        probabilities,
        confidences,
        CLASS_NAMES,
        dataset_name="prueba",
    )