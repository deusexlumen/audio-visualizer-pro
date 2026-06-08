"""
Debug-Skript: Exportiert den SDF Font-Atlas als PNG
und zeigt die UV-Koordinaten fuer ein paar Zeichen an.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.gpu_text_renderer import SDFFontAtlas

# Finde eine verfuegbare Schriftart
import os
font_candidates = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
font_path = None
for p in font_candidates:
    if os.path.exists(p):
        font_path = p
        break

if not font_path:
    print("[FEHLER] Keine Schriftart gefunden!")
    exit(1)

print(f"Verwende Schriftart: {font_path}")

# Atlas bauen (ohne ModernGL Context)
atlas = SDFFontAtlas(font_path, font_size=64, sdf_size=64)
atlas._generate_atlas()

# Als PNG speichern
img = Image.fromarray((np.clip(atlas.texture_data, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
img.save("debug_font_atlas.png")
print("Atlas gespeichert: debug_font_atlas.png")

# Ein paar Zeichen pruefen
test_chars = ['A', 'g', '«', 'ä', 'T', 'W', 'H', 'E', 'L', 'O']
print("\nGlyphen-Metadaten:")
print(f"{'Char':>4} {'x':>4} {'y':>4} {'w':>4} {'h':>4} {'adv':>4}")
print("-" * 30)
for char in test_chars:
    g = atlas.get_glyph(char)
    if g:
        print(f"{char:>4} {g.x:>4} {g.y:>4} {g.w:>4} {g.h:>4} {g.advance:>4}")
    else:
        print(f"{char:>4} NICHT IM ATLAS")

# Zeichen auf dem Atlas markieren
draw = ImageDraw.Draw(img)
for char in test_chars:
    g = atlas.get_glyph(char)
    if g:
        draw.rectangle([g.x, g.y, g.x + g.w, g.y + g.h], outline=255, width=1)
        draw.text((g.x + 2, g.y + 2), char, fill=255)

img.save("debug_font_atlas_annotated.png")
print("\nAnnotierter Atlas gespeichert: debug_font_atlas_annotated.png")
print("Oeffne die Bilder und pruefe ob die Zeichen lesbar sind.")
