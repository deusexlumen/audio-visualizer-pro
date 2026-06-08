"""
Debug-Skript: Rendert Text mit dem GPU-Text-Renderer und speichert als PNG.
"""
import os
import numpy as np
from PIL import Image
import moderngl

from src.gpu_text_renderer import SDFFontAtlas, GPUTextRenderer

# Font finden
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

print(f"Schriftart: {font_path}")

# ModernGL Context erstellen (standalone, kein Fenster)
ctx = moderngl.create_standalone_context()
width, height = 800, 200
fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 3)])

# Atlas + Renderer bauen
atlas = SDFFontAtlas(font_path, font_size=64, sdf_size=64)
tex = atlas.build(ctx)
renderer = GPUTextRenderer(ctx, atlas, tex, width=width, height=height)

# Rendern
fbo.use()
ctx.clear(0.1, 0.1, 0.15)

renderer.render_text(
    "Hello World!", x=width/2, y=height/2, size=64,
    color=(1.0, 1.0, 1.0), align="center"
)

# Als PNG speichern
pixels = fbo.read(components=3)
img = Image.fromarray(np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3)))
img.save("debug_text_render.png")
print("Test-Bild gespeichert: debug_text_render.png")
print("Oeffne das Bild und pruefe ob 'Hello World!' lesbar ist.")

# Cleanup
renderer.release()
tex.release()
ctx.release()
