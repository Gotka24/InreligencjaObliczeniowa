import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import kagglehub
import os
import matplotlib.pyplot as plt
import pandas as pd

def get_data_path():
    path = kagglehub.dataset_download("asaniczka/mammals-image-classification-dataset-45-animals")
    data_dir = os.path.join(path, 'mammals')
    return data_dir


def siatka_klas(full_dataset, classes):
    images_to_show = []
    titles = []
    found_classes = set()

    for img, label in full_dataset:
        class_name = classes[label]
        if class_name not in found_classes and len(found_classes) < 16:

            images_to_show.append(np.array(img))
            titles.append(class_name)
            found_classes.add(class_name)

        if len(found_classes) >= 16:
            break

    plt.figure(figsize=(15, 10))
    for i in range(len(images_to_show)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images_to_show[i])
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("siatka_klas.png")
    plt.show()



def przed_po(data_dir, przyklady=4):

    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    #augumentacja treningowa
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    ds_raw = datasets.ImageFolder(data_dir, transform=raw_transform)
    ds_aug = datasets.ImageFolder(data_dir, transform=train_transform)

    fig, axes = plt.subplots(przyklady, 2, figsize=(10, 4 * przyklady))

    for i in range(przyklady):
        # losowy indeks
        idx = np.random.randint(len(ds_raw))

        # oryginal i zmieniaony obraz
        img_raw, label = ds_raw[idx]
        img_aug, _ = ds_aug[idx]

        # format matlibowy
        img_raw = img_raw.permute(1, 2, 0)
        img_aug = img_aug.permute(1, 2, 0)

        axes[i, 0].imshow(img_raw)
        axes[i, 0].set_title(f"Oryginał: {ds_raw.classes[label]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img_aug)
        axes[i, 1].set_title("Po augmentacji")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("przed_po.png")
    plt.show()




def get_data_loaders(data_dir, batch_size=32):
    # augumentacja danych treningowych
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),                     # rozmiar obrazków 224x224 piksele
        transforms.RandomHorizontalFlip(),                 # 50% szans na flip horyzontalny
        transforms.RandomRotation(10),                     # rotacja zdjęcia [-10, 10] (stopnie)
        transforms.ToTensor(),                             # transformacja do tensora[kanal][x][y]
        transforms.Normalize([0.485, 0.456, 0.406],  # normalizacja kanałów RGB (wartości z ResNet)
                             [0.229, 0.224, 0.225])
    ])

    # przygotowanie do walidacji i testu
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # wczytanie datasetu (do podziału)
    full_dataset = datasets.ImageFolder(data_dir)   # folder data
    classes = full_dataset.classes                  # podfoldery to klasy
    indices = list(range(len(full_dataset)))        # liczba próbek
    targets = full_dataset.targets                  # klasy w formie liczba całkowitych

    # 70% trening
    train_idx, temp_idx = train_test_split(
        indices, train_size=0.7, stratify=targets, random_state=42
    )

    # podział pozostałych 30 na test i walidacje (po 15%)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, stratify=temp_targets, random_state=42
    )

    # tworzenie Subsetow z konkretnych indeksów
    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=test_transform), val_idx)
    test_ds = Subset(datasets.ImageFolder(data_dir, transform=test_transform), test_idx)

    # dataLoadery do iterowania po batchach
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, classes



if __name__ == "__main__":
    data_dir = get_data_path()

    # zliczanie plików w kazdej klasie
    stats = {}
    for class_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            stats[class_folder] = num_images

    # tworzenie DataFrame
    df_stats = pd.DataFrame(list(stats.items()), columns=['Gatunek', 'Liczba_zdjec'])
    df_stats = df_stats.sort_values(by='Liczba_zdjec', ascending=False)

    print(f"Całkowita liczba zdjęć: {df_stats['Liczba_zdjec'].sum()}")
    print(f"Średnia liczba zdjęć na klasę: {df_stats['Liczba_zdjec'].mean():.2f}")

    # rozkład klas
    plt.figure(figsize=(15, 20))
    plt.barh(df_stats['Gatunek'], df_stats['Liczba_zdjec'], color='skyblue')
    plt.xlabel('Liczba obrazów')
    plt.ylabel('Nazwa gatunku')
    plt.title('Liczebność poszczególnych klas w zbiorze danych')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("rozkład.png")

    plt.show()


    temp_ds = datasets.ImageFolder(data_dir)
    siatka_klas(temp_ds, temp_ds.classes)


    przed_po(data_dir)


    train_l, val_l, test_l, classes = get_data_loaders(data_dir)
    print(f"Batch treningowy: {len(train_l)} paczek")
    print(f"Batch walidacyjny: {len(val_l)} paczek")
    print(f"Batch testowy: {len(test_l)} paczek")
