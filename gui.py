"""Thin-Wrapper: Startet die neue PyQt6-GUI."""

import sys
from src.gui.app import run_app

if __name__ == "__main__":
    sys.exit(run_app(sys.argv))
