"""
main.py
-------
Entry point: loads/trains the CNN, then launches the Tkinter GUI.
Run via run.py in the project root (recommended), e.g.:

    python run.py
"""

from model import get_model
from gui import launch


def main():
    model = get_model()
    launch(model)


if __name__ == "__main__":
    main()
