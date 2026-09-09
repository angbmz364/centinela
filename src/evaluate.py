"""
evaluate.py

Contains the evaluation function used for both
validation and testing.
"""

import torch
import torch.nn as nn


def evaluate(model, dataloader, device):
    """
    Evalúa un modelo entrenado.

    Parámetros
    ----------
    model : torch.nn.Module
        La red neuronal entrenada.

    dataloader : DataLoader
        DataLoader de validación o de prueba.

    device : torch.device
        CPU o GPU.

    Retorna
    -------
    average_loss : float
        Pérdida promedio (Cross-Entropy).
    accuracy : float
        Exactitud en porcentaje.
    all_predictions : list
        Clases predichas para cada imagen.
    all_labels : list
        Etiquetas reales para cada imagen.
    all_probabilities : list
        Distribución de probabilidades Softmax (n x num_clases).
    all_confidences : list
        Confianza (probabilidad máxima) de cada predicción.
    """

    # Poner el modelo en modo evaluación
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    correct_predictions = 0

    total_images = 0

    # Guardamos cada predicción: son útiles para la matriz
    # de confusión, precisión/recall, F1 y análisis de confianza.
    all_predictions = []

    all_labels = []

    all_probabilities = []

    all_confidences = []

    # Desactivar gradientes
    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Probabilidad de cada clase y confianza de la predicción
            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = torch.max(probabilities, dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_images += labels.size(0)

            # Guardar predicciones
            all_predictions.extend(predictions.cpu().tolist())

            all_labels.extend(labels.cpu().tolist())

            all_probabilities.extend(probabilities.cpu().tolist())

            all_confidences.extend(confidences.cpu().tolist())

    average_loss = running_loss / len(dataloader)

    accuracy = 100 * correct_predictions / total_images

    return (
        average_loss,
        accuracy,
        all_predictions,
        all_labels,
        all_probabilities,
        all_confidences,
    )