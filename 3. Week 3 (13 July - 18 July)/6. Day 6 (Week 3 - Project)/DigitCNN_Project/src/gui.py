"""
gui.py
------
A lightweight Tkinter GUI (built into Python - nothing extra to
install) for drawing a digit and getting a CNN prediction.

Drawing is mirrored onto a Pillow Image in memory, which is what
actually gets resized and fed to the model.
"""

import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw

CANVAS_SIZE = 280      # on-screen drawing area (px)
IMAGE_SIZE = 28         # size the CNN expects
PEN_WIDTH = 18


class DigitApp:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        self.root.title("Handwritten Digit Recognizer (CNN)")
        self.root.resizable(False, False)

        # ---- title -----------------------------------------------------
        tk.Label(
            root, text="Draw a digit (0-9) below",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=(12, 6))

        # ---- drawing canvas ---------------------------------------------
        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="cross", highlightthickness=2,
            highlightbackground="#888"
        )
        self.canvas.pack(padx=12)

        # Off-screen image that mirrors exactly what's drawn, used for
        # prediction (this is what actually feeds the CNN).
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self._last_xy = None
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # ---- buttons ------------------------------------------------------
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="Predict", width=12, font=("Segoe UI", 11),
            command=self.predict_digit
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_frame, text="Clear", width=12, font=("Segoe UI", 11),
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=6)

        # ---- result --------------------------------------------------------
        self.result_label = tk.Label(
            root, text="Prediction: -", font=("Segoe UI", 20, "bold")
        )
        self.result_label.pack(pady=(6, 0))

        self.confidence_label = tk.Label(
            root, text="", font=("Segoe UI", 11), fg="#555"
        )
        self.confidence_label.pack(pady=(0, 12))

    # -- drawing handlers -----------------------------------------------------
    def _on_press(self, event):
        self._last_xy = (event.x, event.y)
        r = PEN_WIDTH / 2
        self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill="white", outline="white"
        )
        self.draw.ellipse(
            [event.x - r, event.y - r, event.x + r, event.y + r],
            fill=255
        )

    def _on_drag(self, event):
        if self._last_xy is not None:
            x0, y0 = self._last_xy
            self.canvas.create_line(
                x0, y0, event.x, event.y,
                width=PEN_WIDTH, fill="white",
                capstyle=tk.ROUND, smooth=True
            )
            self.draw.line([x0, y0, event.x, event.y], fill=255, width=PEN_WIDTH)
        self._last_xy = (event.x, event.y)

    def _on_release(self, event):
        self._last_xy = None

    # -- actions -----------------------------------------------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: -")
        self.confidence_label.config(text="")

    def predict_digit(self):
        arr = np.array(self.image)
        if arr.max() == 0:
            messagebox.showinfo("Nothing drawn", "Please draw a digit first.")
            return

        small = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        x = np.array(small, dtype="float32") / 255.0
        x = x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

        probs = self.model.predict_proba(x)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100

        self.result_label.config(text=f"Prediction: {digit}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")


def launch(model):
    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()
