import torch
from torchvision.utils import save_image
from gan_struktura import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_DIM = 128
NUM_CLASSES = 45
CLASS_TO_GENERATE = 23
N_IMAGES = 16

G = Generator(Z_DIM, NUM_CLASSES).to(device)
G.load_state_dict(torch.load("samples/G_300.pt", map_location=device))
G.eval()

z = torch.randn(N_IMAGES, Z_DIM, 1, 1, device=device)
labels = torch.full((N_IMAGES,), CLASS_TO_GENERATE, dtype=torch.long, device=device)

with torch.no_grad():
    imgs = G(z, labels)

save_image(imgs, f"generated_class_{CLASS_TO_GENERATE}.png", normalize=True)
print("✓ Wygenerowano obrazy")
