#!/usr/bin/env python3
"""
Launcher für die Audio Visualizer Pro GUI.
Funktioniert auf Windows, macOS und Linux.
"""

import subprocess
import sys
import os


def check_streamlit():
    """Prüft ob Streamlit installiert ist."""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def install_streamlit():
    """Installiert Streamlit."""
    print("📦 Streamlit wird installiert...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
    print("✅ Streamlit installiert!")


def main():
    print("=" * 50)
    print("  🎵 Audio Visualizer Pro - GUI Launcher")
    print("=" * 50)
    print()
    
    # Prüfe Streamlit
    if not check_streamlit():
        install_streamlit()
    else:
        print("✅ Alle Abhängigkeiten sind installiert")
    
    print()
    print("🚀 Starte GUI...")
    print("   Die Anwendung öffnet sich in deinem Browser")
    print("   (Normalerweise unter http://localhost:8501)")
    print()
    
    # Starte Streamlit
    try:
        subprocess.call([
            sys.executable, "-m", "streamlit", "run", "gui.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print()
        print("👋 GUI beendet")


if __name__ == "__main__":
    main()
