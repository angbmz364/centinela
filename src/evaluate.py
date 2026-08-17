"""
evaluate.py

Contains the evaluation function used for both
validation and testing.
"""

import torch
import torch.nn as nn


def evaluate(model, dataloader, device):
    """
    Evaluates a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained neural network.

    dataloader : DataLoader
        Validation or test DataLoader.

    device : torch.device
        CPU or GPU.

    Returns
    -------
    average_loss : float
    accuracy : float
    all_predictions : list
    all_labels : list
    """

    # Put the model into evaluation mode
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    correct_predictions = 0

    total_images = 0

    # We'll save every prediction.
    # These will be useful later for the
    # confusion matrix and precision/recall.
    all_predictions = []

    all_labels = []

    # Disable gradients.
    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()

            total_images += labels.size(0)

            # Save predictions
            all_predictions.extend(predictions.cpu().tolist())

            all_labels.extend(labels.cpu().tolist())

    average_loss = running_loss / len(dataloader)

    accuracy = 100 * correct_predictions / total_images

    return (
        average_loss,
        accuracy,
        all_predictions,
        all_labels,
    )