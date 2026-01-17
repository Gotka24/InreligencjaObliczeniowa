import torch
import torch.nn as nn
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os

from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path

os.makedirs("plots", exist_ok=True)

#Feature extractor ResNet18
resnet = models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # bez warstwy, która decyduje o klasie
feature_extractor.eval()

# zwracanie wektorów cech
def extract_features(loader):
    X, y = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = feature_extractor(imgs).view(imgs.size(0), -1)
            X.append(feats)
            y.append(lbls)
    return torch.cat(X).numpy(), torch.cat(y).numpy()

# ładowanie danych
batch_size = 64
train_loader, val_loader, test_loader, classes = get_data_loaders(get_data_path(), batch_size=batch_size)
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)
X_test, y_test = extract_features(test_loader)

# skalowanie cech, zeby modelom łatwiej było
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# sprawdzam jak teraz sobie radzi knn
best_knn = KNeighborsClassifier(n_neighbors=11, weights='distance', metric='cosine', n_jobs=-1)
best_knn.fit(X_train_scaled, y_train)
y_pred_knn = best_knn.predict(X_test_scaled)

# Dokładność dla knn
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"k-NN (k=11) - dokładność: {acc_knn*100:.2f}%")

# Trójkątna macierz pomyłek
cm_knn = confusion_matrix(y_test, y_pred_knn)
mask = np.tril(np.ones_like(cm_knn, dtype=bool))
plt.figure(figsize=(12,10))
sns.heatmap(cm_knn, annot=False, fmt="d", xticklabels=classes, yticklabels=classes,
            cmap="Blues", mask=mask)
plt.title("k-NN (k=11) -  macierz pomyłek")
plt.xlabel("Przewidziane")
plt.ylabel("Rzeczywiste")
plt.tight_layout()
plt.savefig("plots/knn_k11_confusion_triangle.png")
plt.close()


# knn szukanie najlepszego k
k_values = [1, 3, 5, 11, 21, 31]
knn_acc_list = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_acc_list.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_knn_idx = np.argmax(knn_acc_list)
best_knn_k = k_values[best_knn_idx]
best_knn_acc = knn_acc_list[best_knn_idx]

plt.figure(figsize=(7,5))
plt.plot(k_values, knn_acc_list, marker='o', color='orange')
plt.xlabel("k w k-NN")
plt.ylabel("Dokładność na zbiorze testowym")
plt.title(f"k-NN - najlepsze k={best_knn_k} ({best_knn_acc*100:.2f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/knn_k_values.png")
plt.close()


# Random Forest - optymalizacja liczby drzew
estimators_range = [100, 200, 500, 600]
rf_acc_list = []
for n in estimators_range:
    rf = RandomForestClassifier(n_estimators=n, max_depth=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_acc_list.append(accuracy_score(y_test, rf.predict(X_test_scaled)))

best_rf_idx = np.argmax(rf_acc_list)
best_rf_n = estimators_range[best_rf_idx]
best_rf_acc = rf_acc_list[best_rf_idx]

plt.figure(figsize=(7,5))
plt.plot(estimators_range, rf_acc_list, marker='o', color='green')
plt.xlabel("Liczba drzew w Random Forest")
plt.ylabel("Dokładność na zbiorze testowym")
plt.title(f"Random Forest - najlepsze n={best_rf_n} ({best_rf_acc*100:.2f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/rf_estimators.png")
plt.close()



# SVM - wpływ parametru C
C_values = [0.001, 0.01, 0.1, 1, 10]
svm_acc_list = []
for C in C_values:
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train_scaled, y_train)
    svm_acc_list.append(accuracy_score(y_test, svm.predict(X_test_scaled)))

best_svm_idx = np.argmax(svm_acc_list)
best_svm_C = C_values[best_svm_idx]
best_svm_acc = svm_acc_list[best_svm_idx]

plt.figure(figsize=(7,5))
plt.plot(C_values, svm_acc_list, marker='o', color='red')
plt.xscale('log')
plt.xlabel("C w SVM")
plt.ylabel("Dokładność na zbiorze testowym")
plt.title(f"SVM - najlepsze C={best_svm_C} ({best_svm_acc*100:.2f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/svm_C_values.png")
plt.close()

# Naive Bayes - wpływ var_smoothing

var_values = [1e-9, 1e-8, 1e-7, 1e-6]
nb_acc_list = []
for v in var_values:
    nb = GaussianNB(var_smoothing=v)
    nb.fit(X_train_scaled, y_train)
    nb_acc_list.append(accuracy_score(y_test, nb.predict(X_test_scaled)))

best_nb_idx = np.argmax(nb_acc_list)
best_nb_v = var_values[best_nb_idx]
best_nb_acc = nb_acc_list[best_nb_idx]

plt.figure(figsize=(7,5))
plt.plot(var_values, nb_acc_list, marker='o', color='purple')
plt.xscale('log')
plt.xlabel("var_smoothing w Naive Bayes")
plt.ylabel("Dokładność na zbiorze testowym")
plt.title(f"Naive Bayes - najlepsze var_smoothing={best_nb_v} ({best_nb_acc*100:.2f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/nb_var_smoothing.png")
plt.close()

# Podsumowanie najlepszych modeli
results_df = pd.DataFrame([
    {"model": "Random Forest", "best_param": f"n_estimators={best_rf_n}", "test_accuracy": best_rf_acc},
    {"model": "k-NN", "best_param": f"k={best_knn_k}", "test_accuracy": best_knn_acc},
    {"model": "SVM", "best_param": f"C={best_svm_C}", "test_accuracy": best_svm_acc},
    {"model": "Naive Bayes", "best_param": f"var_smoothing={best_nb_v}", "test_accuracy": best_nb_acc}
])
print("\nPodsumowanie najlepszych modeli:")
print(results_df)

# Wykres dla porównania
plt.figure(figsize=(7,5))
bars = plt.bar(results_df['model'], results_df['test_accuracy'],
               color=['#D0C4FF','#B96EFA','#4D62FA','#303BFA'])
plt.ylabel("Dokładność na zbiorze testowym")
plt.ylim(0, 1)
plt.title("Porównanie najlepszych klasycznych modeli ML")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height*100:.1f}%",
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("plots/models_comparison_best.png")
plt.close()

#macierz pomyłek dla najlpeszego modelu - svm
best_svm_model = SVC(kernel='linear', C=best_svm_C)
best_svm_model.fit(X_train_scaled, y_train)
y_pred_svm = best_svm_model.predict(X_test_scaled)

cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(12,10))
sns.heatmap(cm_svm, annot=False, fmt="d", xticklabels=classes, yticklabels=classes,
            cmap="Reds")
plt.title(f"SVM (C={best_svm_C}) - Macierz pomyłek")
plt.xlabel("Przewidziane")
plt.ylabel("Rzeczywiste")
plt.tight_layout()
plt.savefig("plots/svm_best_confusion_matrix.png")
plt.close()