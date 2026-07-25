"""
check_setup.py
--------------
Run this FIRST if the app won't start:

    python check_setup.py

It checks your Python version and each required package one at a
time, and tells you exactly what's missing / what to do about it.
"""

import os
import sys


def check_python():
    major, minor = sys.version_info[0], sys.version_info[1]
    print(f"Python version: {sys.version.split()[0]}")
    if (major, minor) < (3, 8):
        print("  -> WARNING: Python 3.8+ is recommended.")
    else:
        print("  -> OK")


def check_tkinter():
    try:
        import tkinter  # noqa: F401
        print("tkinter: OK (built into Python)")
    except ImportError:
        print("tkinter: MISSING")
        print("  -> On Linux, install it with: sudo apt-get install python3-tk")
        print("  -> On Windows/Mac, reinstall Python from python.org and make")
        print("     sure 'tcl/tk' is included in the installer options.")


def check_package(name, pip_name=None):
    pip_name = pip_name or name
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown version")
        print(f"{name}: OK ({version})")
    except ImportError:
        print(f"{name}: MISSING")
        print(f"  -> Install with: pip install {pip_name}")


def check_bundled_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "mnist_subset.npz")
    if os.path.exists(data_path):
        size_mb = os.path.getsize(data_path) / 1e6
        print(f"bundled dataset: OK ({size_mb:.1f} MB)")
    else:
        print("bundled dataset: MISSING")
        print(f"  -> Expected at: {data_path}")
        print("  -> Make sure you extracted the WHOLE zip (including the")
        print("     'data' folder), not just individual files.")


def main():
    print("=" * 50)
    print("Digit Recognizer - Environment Check")
    print("=" * 50)
    check_python()
    check_tkinter()
    check_package("numpy")
    check_package("PIL", pip_name="Pillow")
    check_bundled_data()
    print("=" * 50)
    print("This project has NO TensorFlow/PyTorch dependency, so if the")
    print("checks above all say OK, it WILL run:  python run.py")


if __name__ == "__main__":
    main()
