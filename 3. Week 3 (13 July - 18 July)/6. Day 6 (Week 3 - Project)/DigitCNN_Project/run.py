"""
run.py
------
Simplest way to start the app:

    python run.py

This makes sure the src/ folder is importable, then hands off to
src/main.py. Run this file from the project's root folder.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from main import main  # noqa: E402  (import after sys.path tweak, on purpose)

if __name__ == "__main__":
    main()
