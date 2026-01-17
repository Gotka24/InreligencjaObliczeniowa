import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import pandas as pd
from tqdm import tqdm
from struktura import DeepAnimalNet
from przygotowanie_danych.przygotowanie_danych import get_data_loaders, get_data_path


torch.manual_seed(2411)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


batch_size = 32
train_loader, val_loader, _, classes = get_data_loaders(get_data_path(), batch_size=batch_size)


model = DeepAnimalNet(num_classes=len(classes)).to(device)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


epochs = 300
patience = 20
best_val_loss = float('inf')
epochs_no_improve = 0
history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}


#trening
for epoch in range(1, epochs + 1):
    model.train()
    r_loss, r_acc, n = 0, 0, 0

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

        r_loss += loss.item()
        r_acc += (logits.argmax(1) == targets).float().mean().item()
        n += 1
        loop.set_postfix(loss=r_loss / n, acc=r_acc / n)

    # Walidacja
    model.eval()
    v_loss, v_acc, vn = 0, 0, 0
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

    print(f" -> Val Loss: {v_loss / vn:.4f}, Val Acc: {v_acc / vn:.4f}")

    # Early stopping i zapis najlepszego
    if v_loss / vn < best_val_loss:
        best_val_loss = v_loss / vn
        epochs_no_improve = 0
        torch.save(model.state_dict(), "model1.pt")
        print("  [+] Zapisano najlepszy model.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.");
            break

np.savez("history1.npz", **history)
pd.DataFrame(history).to_csv("history1.csv", index=False)