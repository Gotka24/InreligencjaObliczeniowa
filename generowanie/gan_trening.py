import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm
from gan_struktura import Generator, Discriminator
from przygotowanie_danych.przygotowanie_danych import get_gan_loader, get_data_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_DIM = 128
BATCH_SIZE = 32
EPOCHS = 300
LR = 2e-4
SAVE_EVERY = 10

SAMPLE_DIR = "samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

train_loader, classes = get_gan_loader(
    get_data_path(),
    batch_size=BATCH_SIZE,
    image_size=64
)

num_classes = len(classes)

# modele
G = Generator(Z_DIM, num_classes).to(device)
D = Discriminator(num_classes).to(device)

opt_G = Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))
opt_D = Adam(D.parameters(), lr=LR, betas=(0.0, 0.9))

fixed_z = torch.randn(16, Z_DIM, 1, 1, device=device)
fixed_labels = torch.arange(16, device=device) % num_classes


# trening
for epoch in range(EPOCHS):
    G.train()
    D.train()

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        bs = imgs.size(0)

        # trening D
        z = torch.randn(bs, Z_DIM, 1, 1, device=device)
        fake = G(z, labels).detach()

        d_real = D(imgs, labels)
        d_fake = D(fake, labels)

        loss_D = torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # trening G
        z = torch.randn(bs, Z_DIM, 1, 1, device=device)
        fake = G(z, labels)

        loss_G = -torch.mean(D(fake, labels))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"[E{epoch+1}] D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")

    # zapis obrazków
    if (epoch + 1) % SAVE_EVERY == 0:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_z, fixed_labels)
            save_image(
                samples,
                f"{SAMPLE_DIR}/epoch_{epoch+1}.png",
                nrow=4,
                normalize=True
            )
        torch.save(G.state_dict(), f"{SAMPLE_DIR}/G_{epoch+1}.pt")
        torch.save(D.state_dict(), f"{SAMPLE_DIR}/D_{epoch+1}.pt")
