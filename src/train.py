import torch
import torch.nn as nn
from torch.optim import Adam

import dataset
import model

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model.model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.model.classifier.parameters(),
    lr=0.001
)

model.model.train()

NUM_EPOCHS = 1


for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    for images, labels in dataset.train_loader:

        # Move the batch to the selected device
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model.model(images)
        
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(dataset.train_loader)
    print(f"Average Loss: {average_loss:.4f}")

# Save the trained model
torch.save(model.model.state_dict(), "models/centinela_stage1.pth")

print("Model saved successfully!")