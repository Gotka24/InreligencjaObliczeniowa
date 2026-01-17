import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from struktura import DeepAnimalNet
from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader, classes = get_data_loaders(get_data_path(), batch_size=32)

# Wczytanie modelu
model = DeepAnimalNet(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("model1.pt"))
model.eval()

# Wykresy z historii
history = pd.read_csv("history1.csv")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['acc'], label='Train Acc', color='blue')
plt.plot(history['val_acc'], label='Val Acc', color='orange')
plt.title('Dokładność w czasie treningu')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', color='blue')
plt.plot(history['val_loss'], label='Val Loss', color='orange')
plt.title('Strata w czasie treningu')
plt.legend()
plt.savefig("training_plot1.png")
plt.show()

# macierz pomyłek na zbiorze testowym
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        y_true.extend(targets.numpy())
        y_pred.extend(logits.argmax(1).cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.title('Macierz pomyłek')
plt.savefig("matrix1.png")
plt.show()


final_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
print(f"\n[!] Wynik ostateczny na zbiorze TESTOWYM: {final_acc:.2f}%")
print(f"Ewaluacja zakończona.")