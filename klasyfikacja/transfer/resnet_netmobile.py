import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models
from tqdm import tqdm

from przygotowanie_danych.przygotowanie_danych import (
    get_data_loaders, get_data_path
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 20

EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-4

MODELS = ["resnet18", "mobilenetv2"]

train_loader, val_loader, _, classes = get_data_loaders(
    get_data_path(),
    batch_size=BATCH_SIZE
)
num_classes = len(classes)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler()


def build_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        classifier_params = list(model.fc.parameters())

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        classifier_params = list(model.classifier[1].parameters())
    else:
        raise ValueError("Nieznany model")

    return model.to(device), classifier_params


def train_model(model_name):
    print(f"\n================ {model_name.upper()} =================")

    model, classifier_params = build_model(model_name)

    for p in model.parameters():
        p.requires_grad = False
    for p in classifier_params:
        p.requires_grad = True

    optimizer = AdamW(classifier_params, lr=3e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_STAGE1
    )

    for epoch in range(EPOCHS_STAGE1):
        model.train()
        for imgs, targets in tqdm(
            train_loader, desc=f"{model_name} S1 E{epoch+1}"
        ):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                loss = criterion(model(imgs), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    for p in model.parameters():
        p.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_STAGE2
    )

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS_STAGE2):
        model.train()
        for imgs, targets in tqdm(
            train_loader, desc=f"{model_name} S2 E{epoch+1}"
        ):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # walidacja
        model.eval()
        acc, n = 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs).argmax(1)
                acc += (preds == targets).float().mean().item()
                n += 1

        val_acc = acc / n
        print(f" >> Val Acc: {val_acc:.4f}")


        if val_acc > best_acc + MIN_DELTA:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_{model_name}.pt")
            print(" zapisano najlepszy model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping – koniec treningu")
            break

    print(f" Najlepsza walidacja {model_name}: {best_acc:.4f}")


for m in MODELS:
    train_model(m)
