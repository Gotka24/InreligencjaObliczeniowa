import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from struktura2 import DeepAnimalNet
from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader, classes = get_data_loaders(get_data_path(), batch_size=32)

# Ładowanie modelu
model = DeepAnimalNet(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("model2.pt"))
model.eval()

#wykresy
history = pd.read_csv("history2.csv")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history['acc'], label='Train Accuracy', color='#2ca02c', linewidth=2)
plt.plot(history['val_acc'], label='Val Accuracy', color='#d62728', linewidth=2)
plt.title('Krzywa Dokładności', fontsize=14)
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', color='#2ca02c', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', color='#d62728', linewidth=2)
plt.title('Krzywa Straty', fontsize=14)
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_plot2.png")
plt.show()

# Testowanie i Macierz Pomyłek
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu().numpy()
        y_true.extend(targets.numpy())
        y_pred.extend(preds)

print("\n" + "="*50)
print("RAPORT KLASYFIKACJI")
print("="*50)
print(classification_report(y_true, y_pred, target_names=classes))

cm = confusion_matrix(y_true, y_pred)
mask = np.tril(np.ones_like(cm, dtype=bool))

plt.figure(figsize=(20, 16))
sns.heatmap(
    cm,
    mask=mask,
    xticklabels=classes,
    yticklabels=classes,
    cmap='Blues',
    annot=False,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title('Macierz pomyłek – Final Animal Net (zbiór testowy)', fontsize=18)
plt.xlabel('Przewidziane')
plt.ylabel('Rzeczywiste')
plt.tight_layout()
plt.savefig("matrix2.png")
plt.close()

final_acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
print(f"\n[!] Wynik ostateczny na zbiorze TESTOWYM: {final_acc:.2f}%")