import os
import re
from PIL import Image

FOLDER_SAMPLES = "samples"
OUTPUT_GIF = "progres.gif"
CZAS_KLATKI = 600


def gif():
    pliki = [f for f in os.listdir(FOLDER_SAMPLES) if f.startswith("epoch_") and f.endswith(".png")]

    pliki.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    if not pliki:
        print("✗ Nie znaleziono plików epoch_*.png w folderze samples.")
        return

    klatki = []

    for nazwa_pliku in pliki:
        sciezka = os.path.join(FOLDER_SAMPLES, nazwa_pliku)
        img = Image.open(sciezka)

        img = img.convert("RGB")
        klatki.append(img)

    klatki[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=klatki[1:],
        duration=CZAS_KLATKI,
        loop=0,
        optimize=True
    )

    print("GIF zapisany")


if __name__ == "__main__":
    gif()