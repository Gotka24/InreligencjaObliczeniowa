import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path

def prepare_raw_pixels(loader):
    X, y = [], []
    #konwertowanie pikseli na wektory
    for imgs, lbls in loader:
        # trzeba płaszczyć: [batch, 3, 224, 224] -> [batch, 150528]
        X_flat = imgs.view(imgs.size(0), -1).numpy()
        X.append(X_flat)
        y.append(lbls.numpy())
    return np.vstack(X), np.concatenate(y)

train_loader, val_loader, test_loader, classes = get_data_loaders(get_data_path(), batch_size=32)

# wektory pikselowe
X_train, y_train = prepare_raw_pixels(train_loader)
X_test, y_test = prepare_raw_pixels(test_loader)


print("Trenowanie KNN")
knn = KNeighborsClassifier(n_neighbors=11,
                           weights='distance',
                           metric='cosine',
                           n_jobs=-1)
knn.fit(X_train, y_train)

# wyniki
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)


print(f"dokładność: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=classes))