This folder is a placeholder for visual assets used by the application.

Currently, the application draws its "company logo" directly on a Tkinter
Canvas widget inside gui.py (a simple circular badge with the text "EAP"),
so no image file is strictly required to run the project.

If you would like to use a real logo image instead:
1. Place a .png or .jpg logo file in this folder (e.g. assets/logo.png).
2. In gui.py, inside the _build_header() method, replace the Canvas-based
   placeholder logo with a Pillow-loaded image, for example:

    from PIL import Image, ImageTk
    logo_img = Image.open("assets/logo.png").resize((64, 64))
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(header, image=logo_photo, bg=COLOR_BG)
    logo_label.image = logo_photo  # keep a reference!
    logo_label.pack(side="left", padx=(0, 16))
