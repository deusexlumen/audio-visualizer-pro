#!/usr/bin/env python3
"""
Launcher für die Audio Visualizer Pro GUI (DearPyGui).
Funktioniert auf Windows, macOS und Linux.
"""

import subprocess
import sys
import os


def check_dearpygui():
    """Prüft ob DearPyGui installiert ist."""
    try:
        import dearpygui
        return True
    except ImportError:
        return False


def install_dearpygui():
    """Installiert DearPyGui."""
    print("📦 DearPyGui wird installiert...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dearpygui", "-q"])
    print("✅ DearPyGui installiert!")


def main():
    print("=" * 50)
    print("  🎵 Audio Visualizer Pro - GUI Launcher")
    print("=" * 50)
    print()
    
    # Prüfe DearPyGui
    if not check_dearpygui():
        install_dearpygui()
    else:
        print("✅ Alle Abhängigkeiten sind installiert")
    
    print()
    print("🚀 Starte GUI...")
    print()
    
    # Starte DearPyGui direkt
    try:
        subprocess.call([sys.executable, "gui.py"])
    except KeyboardInterrupt:
        print()
        print("👋 GUI beendet")


if __name__ == "__main__":
    main()
