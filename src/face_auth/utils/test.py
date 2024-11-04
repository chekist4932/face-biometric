import torch

from model import model, device
from dataset import test_loader


def test_model():
    correct = 0.0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            for predicted, label in zip(outputs, labels):
                predicted = torch.round(predicted)
                total += 1
                correct += 1 if (predicted == label).sum().item() == 512 else 0

    print(f"Accuracy: {100 * correct / total}")


