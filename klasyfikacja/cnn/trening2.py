import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
from tqdm import tqdm
from struktura2 import DeepAnimalNet
from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path

torch.manual_seed(2411)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

batch_size = 32
train_loader, val_loader, _, classes = get_data_loaders(get_data_path(), batch_size=batch_size)
num_classes = len(classes)

model = DeepAnimalNet(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

epochs = 300
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0

scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}


for epoch in range(1, epochs + 1):
    model.train()
    r_loss, r_acc, n = 0.0, 0.0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for imgs, targets in loop:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        r_loss += loss.item()
        r_acc += (logits.argmax(1) == targets).float().mean().item()
        n += 1
        loop.set_postfix(loss=r_loss / n, acc=r_acc / n)

    model.eval()
    v_loss, v_acc, vn = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            v_loss += criterion(logits, targets).item()
            v_acc += (logits.argmax(1) == targets).float().mean().item()
            vn += 1

    history["loss"].append(r_loss / n)
    history["acc"].append(r_acc / n)
    history["val_loss"].append(v_loss / vn)
    history["val_acc"].append(v_acc / vn)

    print(
        f" >> [E{epoch}] Val Loss: {v_loss / vn:.4f} | Val Acc: {v_acc / vn:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # early stopping
    if v_loss / vn < best_val_loss:
        best_val_loss = v_loss / vn
        epochs_no_improve = 0
        torch.save(model.state_dict(), "model2.pt")
        print(f"  [+] Nowy rekord! Model zapisany.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping po {epoch} epokach.")
            break

# Zapis statystyk
pd.DataFrame(history).to_csv("history2.csv", index=False)
print("[*] Sukces. Pliki gotowe do ewaluacji.")