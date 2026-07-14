# 🎮 SKILL SPECIFICATION: GPU & OpenGL Development

## 1. ARCHITEKTUR-PRINZIPIEN
- **ModernGL Context:** Immer `create_standalone_context()` verwenden (kein GLFW/Window nötig).
- **Resource Management:** Alle FBOs, VAOs, VBOs, Texturen MÜSSEN `.release()` aufrufen.
- **Shader-Version:** `#version 330` als Minimum.
- **Alpha-Handling:** Visualizer rendern in RGBA, Background in RGB, Composite-Shader verrechnet korrekt.

## 2. HARDCODED CONSTRAINTS

- **[CONSTRAINT_GPU_1] Context-Lifetime:** Ein ModernGL-Context pro Renderer-Instanz. Niemals Context zwischen Threads teilen.
- **[CONSTRAINT_GPU_2] Texture-Format:**
  - Visualizer-Output: RGBA (4 Kanäle) für Alpha-Support
  - Background: RGB (3 Kanäle)
  - Final-Output: RGB (3 Kanäle) für FFmpeg
- **[CONSTRAINT_GPU_3] Pixel-Readback:** `fbo.read(components=3)` für FFmpeg-kompatibles RGB-Format.

## 3. FEHLERBEHANDLUNG
- `moderngl.Error` bei Shader-Compilation → Stacktrace loggen, Shader-Source mit Zeilennummern ausgeben.
- Context-Creation-Fail → Graceful Fallback auf CPU/PIL-Renderer (wenn implementiert).
