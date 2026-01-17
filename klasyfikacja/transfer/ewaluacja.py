import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix

from przygotowanie_danych.przygotowanie_danych import (
    get_data_loaders, get_data_path
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


_, _, test_loader, classes = get_data_loaders(
    get_data_path(),
    batch_size=BATCH_SIZE
)
num_classes = len(classes)


def evaluate(model, weights_path, name):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(targets.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"{name}: {acc*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    mask = np.tril(np.ones_like(cm, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        mask=mask
    )
    plt.title(f"{name} – macierz pomyłek")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion.png")
    plt.close()

    return acc


#  RESNET
resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

acc_resnet = evaluate(
    resnet,
    "best_resnet18.pt",
    "ResNet18"
)

# MOBILENET
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(
    mobilenet.classifier[1].in_features,
    num_classes
)

acc_mobilenet = evaluate(
    mobilenet,
    "best_mobilenetv2.pt",
    "MobileNetV2"
)

print(f"ResNet18    : {acc_resnet*100:.2f}%")
print(f"MobileNetV2 : {acc_mobilenet*100:.2f}%")
