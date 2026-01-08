from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path

def przygotuj_dane(loader):
    X, y = [], []
    for imgs, lbls in loader:
        X_batch = imgs.view(imgs.size(0), -1).numpy()
        X.append(X_batch)
        y.append(lbls.numpy())
    return np.vstack(X), np.concatenate(y)

train_loader, val_loader, test_loader, classes = get_data_loaders(get_data_path(), batch_size=64)

X_train, y_train = przygotuj_dane(train_loader)
X_test, y_test = przygotuj_dane(test_loader)


# automatyzacja procesu przetwarzania
pipeline_knn = Pipeline([
    ('pca', PCA(n_components=200, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=11, weights='distance', metric='cosine'))
])

pipeline_knn.fit(X_train, y_train)

pca_object = pipeline_knn.named_steps['pca']
cumulative_variance = np.cumsum(pca_object.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')

plt.axvline(x=200, color='r', linestyle=':', label='Wybrane 200 komponentów')
plt.axhline(y=cumulative_variance[199], color='g', linestyle=':', label=f'Wariancja: {cumulative_variance[199]:.2f}')

plt.title('Skumulowana Wyjaśniona Wariancja przez PCA')
plt.xlabel('Liczba Składowych Głównych (Components)')
plt.ylabel('Skumulowana Wariancja')
plt.grid(True)
plt.legend()
plt.savefig('Wariancja.png')
plt.show()

# wariancja wyjaśniona przez 200 komponentów
pca_object = pipeline_knn.named_steps['pca']
total_variance = np.sum(pca_object.explained_variance_ratio_)

print(f"Liczba komponentów: {pca_object.n_components_}")
print(f"Zachowana wariancja: {total_variance * 100:.2f}%")

y_pred = pipeline_knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Osiągnięta dokładność: {acc * 100:.2f}%")







