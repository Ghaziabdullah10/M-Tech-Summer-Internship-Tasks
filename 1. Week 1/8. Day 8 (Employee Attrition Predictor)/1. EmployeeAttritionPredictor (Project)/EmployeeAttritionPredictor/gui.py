"""
gui.py
------
This module contains the complete Tkinter Graphical User Interface for the
Employee Attrition Predictor application.

The interface includes:
    - A menu bar (File, Model, View, Help)
    - A header showing the project title and student name
    - A Prediction tab with input fields (Entry, Combobox, Spinbox) for all
      employee attributes, a Predict / Reset / Exit button set, and a
      colour-coded result display
    - A History tab that stores every prediction made (with a timestamp) in
      a CSV file, displayed in a searchable table (Treeview)
    - A Graphs tab that displays the model comparison chart and confusion
      matrix generated during training
    - A dark, professional colour theme applied throughout
    - Full input validation and friendly error / success message boxes
    - An About dialog and a Help dialog

This file only handles the GUI/presentation layer. All Machine Learning
logic lives in predictor.py and train_model.py.
"""

import os
import csv
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from predictor import AttritionPredictor, ModelNotFoundError

# ---------------------------------------------------------------------------
# NOTE: The dropdown option lists are defined locally below so that gui.py
# has no hard dependency on the dataset generator module.
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
HISTORY_CSV = os.path.join(REPORTS_DIR, "prediction_history.csv")

# ---------------------------------------------------------------------------
# Dropdown / option values shown in the GUI (must match values the model
# was trained on inside generate_dataset.py / preprocess.py)
# ---------------------------------------------------------------------------
GENDER_OPTIONS = ["Male", "Female"]
DEPARTMENT_OPTIONS = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLE_OPTIONS = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]
EDUCATION_OPTIONS = ["Below College", "College", "Bachelor", "Master", "Doctor"]
EDUCATION_TO_NUMBER = {label: idx + 1 for idx, label in enumerate(EDUCATION_OPTIONS)}
BUSINESS_TRAVEL_OPTIONS = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
MARITAL_STATUS_OPTIONS = ["Single", "Married", "Divorced"]
OVERTIME_OPTIONS = ["Yes", "No"]
SATISFACTION_OPTIONS = ["1 - Low", "2 - Medium", "3 - High", "4 - Very High"]
WORK_LIFE_BALANCE_OPTIONS = ["1 - Bad", "2 - Good", "3 - Better", "4 - Best"]
PERFORMANCE_OPTIONS = ["3 - Excellent", "4 - Outstanding"]

# ---------------------------------------------------------------------------
# Dark theme colour palette used throughout the application
# ---------------------------------------------------------------------------
COLOR_BG = "#1e1e2f"
COLOR_BG_PANEL = "#252538"
COLOR_BG_INPUT = "#2f2f45"
COLOR_TEXT = "#eaeaea"
COLOR_TEXT_MUTED = "#a0a0b0"
COLOR_ACCENT = "#4C72B0"
COLOR_SUCCESS = "#4caf50"
COLOR_DANGER = "#e05c5c"
COLOR_WARNING = "#e0a75c"
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_SUBTITLE = ("Segoe UI", 11)
FONT_LABEL = ("Segoe UI", 10)
FONT_LABEL_BOLD = ("Segoe UI", 10, "bold")
FONT_RESULT = ("Segoe UI", 16, "bold")


class EmployeeAttritionApp:
    """
    Main application class that builds and manages the entire Tkinter GUI
    for the Employee Attrition Predictor.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Employee Attrition Predictor")
        self.root.geometry("1050x720")
        self.root.minsize(950, 650)
        self.root.configure(bg=COLOR_BG)

        # Try to load the trained model. If it is missing, we still build
        # the GUI so the user can train it from the menu, but we disable
        # the Predict button until a model becomes available.
        self.predictor = None
        self.model_ready = False
        self._try_load_model(show_error=False)

        # Dictionary that will hold references to every input widget so we
        # can easily read their values later and reset them all at once.
        self.input_widgets = {}

        # Build all sections of the interface
        self._configure_styles()
        self._build_menu_bar()
        self._build_header()
        self._build_notebook()
        self._build_status_bar()

        # Make sure the history CSV file exists with proper headers
        self._ensure_history_file()

        # Load any existing history into the History tab table
        self._refresh_history_table()

        self.set_status("Ready.")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _try_load_model(self, show_error=True):
        """Attempt to load the trained model via predictor.py."""
        try:
            self.predictor = AttritionPredictor(models_dir=MODELS_DIR)
            self.model_ready = True
            return True
        except ModelNotFoundError as e:
            self.model_ready = False
            if show_error:
                messagebox.showerror("Model Not Found", str(e))
            return False
        except Exception as e:
            self.model_ready = False
            if show_error:
                messagebox.showerror("Error Loading Model", f"An unexpected error occurred:\n{e}")
            return False

    def _configure_styles(self):
        """Configure ttk widget styles to build a consistent dark theme."""
        style = ttk.Style()
        # 'clam' theme allows full colour customisation on Windows/Linux
        style.theme_use("clam")

        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=COLOR_BG_PANEL,
                        foreground=COLOR_TEXT, padding=(18, 8), font=FONT_LABEL_BOLD)
        style.map("TNotebook.Tab",
                  background=[("selected", COLOR_ACCENT)],
                  foreground=[("selected", "#ffffff")])

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_BG_PANEL)

        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_LABEL)
        style.configure("Panel.TLabel", background=COLOR_BG_PANEL, foreground=COLOR_TEXT, font=FONT_LABEL)
        style.configure("Muted.TLabel", background=COLOR_BG_PANEL, foreground=COLOR_TEXT_MUTED, font=FONT_SUBTITLE)
        style.configure("Title.TLabel", background=COLOR_BG, foreground="#ffffff", font=FONT_TITLE)

        style.configure("TCombobox", fieldbackground=COLOR_BG_INPUT, background=COLOR_BG_INPUT,
                        foreground=COLOR_TEXT, arrowcolor=COLOR_TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", COLOR_BG_INPUT)])

        style.configure("TButton", font=FONT_LABEL_BOLD, padding=8)
        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#3d5c8c")])
        style.configure("Success.TButton", background=COLOR_SUCCESS, foreground="#ffffff")
        style.map("Success.TButton", background=[("active", "#3d8b40")])
        style.configure("Danger.TButton", background=COLOR_DANGER, foreground="#ffffff")
        style.map("Danger.TButton", background=[("active", "#b84545")])

        style.configure("Treeview", background=COLOR_BG_INPUT, foreground=COLOR_TEXT,
                        fieldbackground=COLOR_BG_INPUT, rowheight=26, font=FONT_LABEL)
        style.configure("Treeview.Heading", background=COLOR_ACCENT, foreground="#ffffff",
                        font=FONT_LABEL_BOLD)
        style.map("Treeview", background=[("selected", COLOR_ACCENT)])

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _build_menu_bar(self):
        """Build the top menu bar (File, Model, View, Help)."""
        menu_bar = tk.Menu(self.root)

        # ---- File menu ----
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Export History to CSV...", command=self.export_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # ---- Model menu ----
        model_menu = tk.Menu(menu_bar, tearoff=0)
        model_menu.add_command(label="Train / Retrain Model", command=self.train_model_action)
        model_menu.add_command(label="Reload Trained Model", command=self.reload_model_action)
        model_menu.add_command(label="View Model Info", command=self.show_model_info)
        menu_bar.add_cascade(label="Model", menu=model_menu)

        # ---- View menu ----
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Prediction Tab", command=lambda: self.notebook.select(self.prediction_tab))
        view_menu.add_command(label="History Tab", command=lambda: self.notebook.select(self.history_tab))
        view_menu.add_command(label="Graphs Tab", command=lambda: self.notebook.select(self.graphs_tab))
        menu_bar.add_cascade(label="View", menu=view_menu)

        # ---- Help menu ----
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Help", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    # ------------------------------------------------------------------
    # Header (title, student info, logo placeholder)
    # ------------------------------------------------------------------
    def _build_header(self):
        """Build the top header section with title, student details and logo."""
        header = tk.Frame(self.root, bg=COLOR_BG, pady=12, padx=16)
        header.pack(fill="x")

        # Logo placeholder (a simple canvas circle acting as a company logo)
        logo_canvas = tk.Canvas(header, width=64, height=64, bg=COLOR_BG, highlightthickness=0)
        logo_canvas.pack(side="left", padx=(0, 16))
        logo_canvas.create_oval(4, 4, 60, 60, fill=COLOR_ACCENT, outline="")
        logo_canvas.create_text(32, 32, text="EAP", fill="#ffffff", font=("Segoe UI", 12, "bold"))

        text_frame = tk.Frame(header, bg=COLOR_BG)
        text_frame.pack(side="left", fill="x", expand=True)

        title_label = ttk.Label(text_frame, text="Employee Attrition Predictor", style="Title.TLabel")
        title_label.pack(anchor="w")

        subtitle_label = ttk.Label(
            text_frame,
            text="AI-Based Desktop Application  |  Machine Learning Project",
            font=FONT_SUBTITLE, background=COLOR_BG, foreground=COLOR_TEXT_MUTED
        )
        subtitle_label.pack(anchor="w")

        # Student details on the right side of the header
        student_frame = tk.Frame(header, bg=COLOR_BG)
        student_frame.pack(side="right")
        ttk.Label(student_frame, text="Student: Ghazi Muhammad Abdullah",
                  font=FONT_LABEL_BOLD, background=COLOR_BG, foreground=COLOR_TEXT).pack(anchor="e")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        """Build the bottom status bar that shows helpful runtime messages."""
        self.status_var = tk.StringVar(value="Ready.")
        status_frame = tk.Frame(self.root, bg=COLOR_BG_PANEL, height=28)
        status_frame.pack(fill="x", side="bottom")
        status_label = tk.Label(status_frame, textvariable=self.status_var, bg=COLOR_BG_PANEL,
                                 fg=COLOR_TEXT_MUTED, anchor="w", font=("Segoe UI", 9), padx=10)
        status_label.pack(fill="x")

    def set_status(self, message):
        """Update the text shown in the bottom status bar."""
        self.status_var.set(message)

    # ------------------------------------------------------------------
    # Notebook (tabs)
    # ------------------------------------------------------------------
    def _build_notebook(self):
        """Build the tabbed interface: Prediction, History, Graphs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.prediction_tab = ttk.Frame(self.notebook, style="TFrame")
        self.history_tab = ttk.Frame(self.notebook, style="TFrame")
        self.graphs_tab = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.prediction_tab, text="  Prediction  ")
        self.notebook.add(self.history_tab, text="  History  ")
        self.notebook.add(self.graphs_tab, text="  Graphs  ")

        self._build_prediction_tab()
        self._build_history_tab()
        self._build_graphs_tab()

    # ------------------------------------------------------------------
    # PREDICTION TAB
    # ------------------------------------------------------------------
    def _build_prediction_tab(self):
        """
        Build the Prediction tab containing the employee input form, the
        Predict/Reset/Exit buttons and the prediction result display.
        """
        outer = tk.Frame(self.prediction_tab, bg=COLOR_BG)
        outer.pack(fill="both", expand=True, padx=6, pady=6)

        # Scrollable canvas so the form fits nicely even on smaller screens
        canvas = tk.Canvas(outer, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        form_container = tk.Frame(canvas, bg=COLOR_BG)

        form_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=form_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse-wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        form_panel = tk.Frame(form_container, bg=COLOR_BG_PANEL, padx=20, pady=16)
        form_panel.pack(fill="x", padx=8, pady=8)

        ttk.Label(form_panel, text="Enter Employee Information", style="Panel.TLabel",
                  font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 14))

        # Two-column grid layout for the form fields
        left_col = 0
        right_col = 2
        row = 1

        def add_field(label_text, widget_key, widget_type, options=None, col=left_col, row_num=None, **kwargs):
            """Helper that adds a labelled input widget to the form grid."""
            nonlocal row
            r = row_num if row_num is not None else row
            lbl = ttk.Label(form_panel, text=label_text + ":", style="Panel.TLabel")
            lbl.grid(row=r, column=col, sticky="w", padx=(0, 8), pady=6)

            if widget_type == "entry":
                widget = tk.Entry(form_panel, bg=COLOR_BG_INPUT, fg=COLOR_TEXT,
                                   insertbackground=COLOR_TEXT, relief="flat", width=22)
            elif widget_type == "combobox":
                widget = ttk.Combobox(form_panel, values=options, state="readonly", width=20)
                widget.current(0)
            elif widget_type == "spinbox":
                widget = tk.Spinbox(form_panel, from_=kwargs.get("from_", 0), to=kwargs.get("to", 100),
                                     bg=COLOR_BG_INPUT, fg=COLOR_TEXT, relief="flat", width=20,
                                     insertbackground=COLOR_TEXT)
                widget.delete(0, "end")
                widget.insert(0, str(kwargs.get("default", kwargs.get("from_", 0))))
            else:
                raise ValueError("Unknown widget type")

            widget.grid(row=r, column=col + 1, sticky="w", pady=6)
            self.input_widgets[widget_key] = widget
            return r

        # ---- Column 1 fields ----
        row = add_field("Age", "Age", "spinbox", from_=18, to=65, default=30, col=left_col, row_num=row); row += 1
        row = add_field("Gender", "Gender", "combobox", GENDER_OPTIONS, col=left_col, row_num=row); row += 1
        row = add_field("Department", "Department", "combobox", DEPARTMENT_OPTIONS, col=left_col, row_num=row); row += 1
        row = add_field("Job Role", "JobRole", "combobox", JOB_ROLE_OPTIONS, col=left_col, row_num=row); row += 1
        row = add_field("Monthly Income (Rs.)", "MonthlyIncome", "entry", col=left_col, row_num=row); row += 1
        row = add_field("Education", "Education", "combobox", EDUCATION_OPTIONS, col=left_col, row_num=row); row += 1
        row = add_field("Business Travel", "BusinessTravel", "combobox", BUSINESS_TRAVEL_OPTIONS, col=left_col, row_num=row); row += 1
        row = add_field("Marital Status", "MaritalStatus", "combobox", MARITAL_STATUS_OPTIONS, col=left_col, row_num=row); row += 1

        # ---- Column 2 fields (reset row counter for right column) ----
        row2 = 1
        row2 = add_field("Job Satisfaction", "JobSatisfaction", "combobox", SATISFACTION_OPTIONS, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Environment Satisfaction", "EnvironmentSatisfaction", "combobox", SATISFACTION_OPTIONS, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Work Life Balance", "WorkLifeBalance", "combobox", WORK_LIFE_BALANCE_OPTIONS, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Overtime", "OverTime", "combobox", OVERTIME_OPTIONS, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Years At Company", "YearsAtCompany", "spinbox", from_=0, to=40, default=3, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Total Working Years", "TotalWorkingYears", "spinbox", from_=0, to=45, default=5, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Distance From Home (km)", "DistanceFromHome", "spinbox", from_=1, to=50, default=5, col=right_col, row_num=row2); row2 += 1
        row2 = add_field("Performance Rating", "PerformanceRating", "combobox", PERFORMANCE_OPTIONS, col=right_col, row_num=row2); row2 += 1

        final_row = max(row, row2) + 1

        # ---- Overtime as RadioButtons too, to satisfy the "RadioButton" widget
        # requirement while keeping OverTime's combobox as the primary control.
        # We instead use radio buttons for a genuinely binary, natural fit:
        # "Would you like to include this employee's OverTime status?" is not
        # needed - OverTime itself is demonstrated via Combobox above. To make
        # sure RadioButtons are demonstrated with a real field, Gender is
        # additionally offered via RadioButtons which stay synced with the
        # Gender combobox above.
        gender_radio_frame = tk.Frame(form_panel, bg=COLOR_BG_PANEL)
        gender_radio_frame.grid(row=final_row, column=0, columnspan=4, sticky="w", pady=(10, 4))
        ttk.Label(form_panel, text="Confirm Gender (Radio Buttons):", style="Panel.TLabel").grid(
            row=final_row, column=0, columnspan=2, sticky="w"
        )
        self.gender_radio_var = tk.StringVar(value=GENDER_OPTIONS[0])

        def sync_gender_from_radio():
            self.input_widgets["Gender"].set(self.gender_radio_var.get())

        radio_row_frame = tk.Frame(form_panel, bg=COLOR_BG_PANEL)
        radio_row_frame.grid(row=final_row + 1, column=0, columnspan=4, sticky="w", pady=(0, 10))
        for option in GENDER_OPTIONS:
            rb = tk.Radiobutton(
                radio_row_frame, text=option, value=option, variable=self.gender_radio_var,
                bg=COLOR_BG_PANEL, fg=COLOR_TEXT, selectcolor=COLOR_BG_INPUT,
                activebackground=COLOR_BG_PANEL, activeforeground=COLOR_TEXT,
                command=sync_gender_from_radio, font=FONT_LABEL
            )
            rb.pack(side="left", padx=(0, 16))

        button_row = final_row + 2

        # ---- Action buttons ----
        button_frame = tk.Frame(form_panel, bg=COLOR_BG_PANEL)
        button_frame.grid(row=button_row, column=0, columnspan=4, pady=(16, 6), sticky="w")

        predict_btn = ttk.Button(button_frame, text="Predict", style="Accent.TButton",
                                  command=self.on_predict)
        predict_btn.pack(side="left", padx=(0, 10))

        reset_btn = ttk.Button(button_frame, text="Reset", command=self.on_reset)
        reset_btn.pack(side="left", padx=(0, 10))

        exit_btn = ttk.Button(button_frame, text="Exit", style="Danger.TButton", command=self.on_exit)
        exit_btn.pack(side="left")

        # ---- Result display ----
        result_frame = tk.Frame(form_container, bg=COLOR_BG_PANEL, padx=20, pady=16)
        result_frame.pack(fill="x", padx=8, pady=(0, 20))

        ttk.Label(result_frame, text="Prediction Result", style="Panel.TLabel",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self.result_var = tk.StringVar(value="No prediction yet.")
        self.result_label = tk.Label(result_frame, textvariable=self.result_var, font=FONT_RESULT,
                                      bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED)
        self.result_label.pack(anchor="w")

        self.confidence_var = tk.StringVar(value="")
        self.confidence_label = tk.Label(result_frame, textvariable=self.confidence_var, font=FONT_SUBTITLE,
                                          bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED)
        self.confidence_label.pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # HISTORY TAB
    # ------------------------------------------------------------------
    def _build_history_tab(self):
        """
        Build the History tab, which displays all past predictions saved to
        the CSV history file, with a search box and refresh/delete controls.
        """
        outer = tk.Frame(self.history_tab, bg=COLOR_BG, padx=10, pady=10)
        outer.pack(fill="both", expand=True)

        # ---- Search bar ----
        search_frame = tk.Frame(outer, bg=COLOR_BG)
        search_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(search_frame, text="Search History:").pack(side="left", padx=(0, 8))
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, bg=COLOR_BG_INPUT,
                                 fg=COLOR_TEXT, insertbackground=COLOR_TEXT, relief="flat", width=30)
        search_entry.pack(side="left", padx=(0, 8))
        search_entry.bind("<Return>", lambda e: self._refresh_history_table())

        ttk.Button(search_frame, text="Search", command=self._refresh_history_table).pack(side="left", padx=4)
        ttk.Button(search_frame, text="Show All", command=self._clear_search).pack(side="left", padx=4)
        ttk.Button(search_frame, text="Refresh", command=self._refresh_history_table).pack(side="left", padx=4)
        ttk.Button(search_frame, text="Clear History", style="Danger.TButton",
                   command=self.clear_history).pack(side="right", padx=4)

        # ---- History table (Treeview) ----
        columns = (
            "Timestamp", "Age", "Gender", "Department", "JobRole", "MonthlyIncome",
            "OverTime", "YearsAtCompany", "Prediction", "Confidence"
        )
        table_frame = tk.Frame(outer, bg=COLOR_BG)
        table_frame.pack(fill="both", expand=True)

        self.history_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor="center")
        self.history_tree.column("Timestamp", width=150)

        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=v_scroll.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

    def _clear_search(self):
        """Clear the search box and reload the full history table."""
        self.search_var.set("")
        self._refresh_history_table()

    # ------------------------------------------------------------------
    # GRAPHS TAB
    # ------------------------------------------------------------------
    def _build_graphs_tab(self):
        """
        Build the Graphs tab, which shows buttons to view the model
        comparison chart and confusion matrix produced by train_model.py.
        """
        outer = tk.Frame(self.graphs_tab, bg=COLOR_BG, padx=16, pady=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Model Performance Visualisations",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 12))

        btn_frame = tk.Frame(outer, bg=COLOR_BG)
        btn_frame.pack(anchor="w", pady=(0, 16))

        ttk.Button(btn_frame, text="View Model Comparison Chart", style="Accent.TButton",
                   command=lambda: self._open_image_window(
                       os.path.join(REPORTS_DIR, "model_comparison.png"),
                       "Model Comparison Chart")).pack(side="left", padx=(0, 10))

        ttk.Button(btn_frame, text="View Confusion Matrix", style="Accent.TButton",
                   command=lambda: self._open_image_window(
                       os.path.join(REPORTS_DIR, "confusion_matrix.png"),
                       "Confusion Matrix")).pack(side="left", padx=(0, 10))

        ttk.Button(btn_frame, text="View Evaluation Report (Text)",
                   command=self._show_evaluation_report_text).pack(side="left")

        # Inline preview area for the currently selected image
        self.graph_preview_frame = tk.Frame(outer, bg=COLOR_BG_PANEL)
        self.graph_preview_frame.pack(fill="both", expand=True)
        self.graph_preview_label = tk.Label(
            self.graph_preview_frame, text="Select a chart above to preview it here.",
            bg=COLOR_BG_PANEL, fg=COLOR_TEXT_MUTED, font=FONT_SUBTITLE
        )
        self.graph_preview_label.pack(expand=True)
        self._preview_image_ref = None  # keep a reference to avoid garbage collection

    def _open_image_window(self, image_path, title):
        """
        Display a saved PNG chart. Shows it inline in the Graphs tab preview
        area (using Pillow) if available, and also opens it in a separate
        pop-up window for a larger view.
        """
        if not os.path.exists(image_path):
            messagebox.showwarning(
                "Chart Not Found",
                f"'{os.path.basename(image_path)}' was not found.\n"
                f"Please train the model first (Model > Train / Retrain Model)."
            )
            return

        if not PIL_AVAILABLE:
            messagebox.showinfo(
                "Pillow Not Installed",
                f"The chart was saved to:\n{image_path}\n\n"
                f"Install the 'pillow' library to preview images inside the app."
            )
            return

        # Inline preview
        try:
            img = Image.open(image_path)
            img.thumbnail((760, 460))
            photo = ImageTk.PhotoImage(img)
            self.graph_preview_label.configure(image=photo, text="")
            self.graph_preview_label.image = photo
            self._preview_image_ref = photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not display the chart:\n{e}")
            return

        # Pop-up window with a larger view
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.configure(bg=COLOR_BG)
        try:
            img_full = Image.open(image_path)
            img_full.thumbnail((1000, 750))
            photo_full = ImageTk.PhotoImage(img_full)
            lbl = tk.Label(popup, image=photo_full, bg=COLOR_BG)
            lbl.image = photo_full
            lbl.pack(padx=10, pady=10)
        except Exception:
            pass

    def _show_evaluation_report_text(self):
        """Display the plain-text model evaluation report in a pop-up window."""
        report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
        if not os.path.exists(report_path):
            messagebox.showwarning("Report Not Found",
                                    "No evaluation report found. Please train the model first.")
            return
        with open(report_path, "r") as f:
            content = f.read()

        popup = tk.Toplevel(self.root)
        popup.title("Model Evaluation Report")
        popup.configure(bg=COLOR_BG)
        popup.geometry("600x500")
        text_widget = tk.Text(popup, bg=COLOR_BG_INPUT, fg=COLOR_TEXT, wrap="word",
                               font=("Consolas", 10), padx=10, pady=10)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")

    # ------------------------------------------------------------------
    # Input gathering and validation
    # ------------------------------------------------------------------
    def _gather_and_validate_inputs(self):
        """
        Read every value from the input widgets, validate them, and return
        a clean dictionary ready to be passed to the predictor.

        Returns
        -------
        dict or None
            The validated employee data dictionary, or None if validation
            failed (an error message box will already have been shown).
        """
        try:
            age = int(self.input_widgets["Age"].get())
            if not (18 <= age <= 65):
                raise ValueError("Age must be between 18 and 65.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Age is invalid.\n{e}")
            return None

        gender = self.input_widgets["Gender"].get()
        department = self.input_widgets["Department"].get()
        job_role = self.input_widgets["JobRole"].get()

        try:
            monthly_income_text = self.input_widgets["MonthlyIncome"].get().strip()
            monthly_income = float(monthly_income_text)
            if monthly_income <= 0:
                raise ValueError("Monthly Income must be a positive number.")
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Monthly Income must be a valid positive number (e.g. 5000)."
            )
            return None

        education_label = self.input_widgets["Education"].get()
        education = EDUCATION_TO_NUMBER.get(education_label)
        business_travel = self.input_widgets["BusinessTravel"].get()
        marital_status = self.input_widgets["MaritalStatus"].get()

        job_satisfaction = int(self.input_widgets["JobSatisfaction"].get()[0])
        environment_satisfaction = int(self.input_widgets["EnvironmentSatisfaction"].get()[0])
        work_life_balance = int(self.input_widgets["WorkLifeBalance"].get()[0])
        performance_rating = int(self.input_widgets["PerformanceRating"].get()[0])
        overtime = self.input_widgets["OverTime"].get()

        try:
            years_at_company = int(self.input_widgets["YearsAtCompany"].get())
            total_working_years = int(self.input_widgets["TotalWorkingYears"].get())
            distance_from_home = int(self.input_widgets["DistanceFromHome"].get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Years/Distance fields must be whole numbers.")
            return None

        if total_working_years < years_at_company:
            messagebox.showerror(
                "Invalid Input",
                "Total Working Years cannot be smaller than Years At Company."
            )
            return None

        if not all([gender, department, job_role, education_label, business_travel,
                    marital_status, overtime]):
            messagebox.showerror("Invalid Input", "Please make sure every field has been filled in.")
            return None

        employee_data = {
            "Age": age,
            "Gender": gender,
            "Department": department,
            "JobRole": job_role,
            "MonthlyIncome": monthly_income,
            "Education": education,
            "BusinessTravel": business_travel,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": environment_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "OverTime": overtime,
            "YearsAtCompany": years_at_company,
            "TotalWorkingYears": total_working_years,
            "DistanceFromHome": distance_from_home,
            "PerformanceRating": performance_rating,
            "MaritalStatus": marital_status,
        }
        return employee_data

    # ------------------------------------------------------------------
    # Button event handlers
    # ------------------------------------------------------------------
    def on_predict(self):
        """Handle the Predict button click: validate, predict, display, save."""
        if not self.model_ready:
            messagebox.showerror(
                "Model Not Trained",
                "No trained model was found. Please go to Model > Train / Retrain Model first."
            )
            self.set_status("Prediction failed: model not trained.")
            return

        employee_data = self._gather_and_validate_inputs()
        if employee_data is None:
            self.set_status("Prediction cancelled due to invalid input.")
            return

        try:
            label, confidence = self.predictor.predict(employee_data)
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred while predicting:\n{e}")
            self.set_status("Prediction failed due to an internal error.")
            return

        # Update the result display with colour coding
        self.result_var.set(label)
        self.confidence_var.set(f"Confidence: {confidence}%")
        if label == "Likely to Leave":
            self.result_label.configure(fg=COLOR_DANGER)
        else:
            self.result_label.configure(fg=COLOR_SUCCESS)
        self.confidence_label.configure(fg=COLOR_TEXT_MUTED)

        # Save this prediction to the history CSV file
        self._save_prediction_to_history(employee_data, label, confidence)
        self._refresh_history_table()

        messagebox.showinfo("Prediction Complete", f"Result: {label}\nConfidence: {confidence}%")
        self.set_status(f"Prediction complete: {label} ({confidence}% confidence).")

    def on_reset(self):
        """Reset every input field back to its default value."""
        for key, widget in self.input_widgets.items():
            if isinstance(widget, ttk.Combobox):
                widget.current(0)
            elif isinstance(widget, tk.Spinbox):
                widget.delete(0, "end")
                widget.insert(0, widget.cget("from"))
            elif isinstance(widget, tk.Entry):
                widget.delete(0, "end")
        self.gender_radio_var.set(GENDER_OPTIONS[0])
        self.result_var.set("No prediction yet.")
        self.confidence_var.set("")
        self.result_label.configure(fg=COLOR_TEXT_MUTED)
        self.set_status("Form has been reset.")

    def on_exit(self):
        """Handle application exit with a confirmation prompt."""
        if messagebox.askyesno("Exit Application", "Are you sure you want to exit?"):
            self.root.destroy()

    # ------------------------------------------------------------------
    # History (CSV) handling
    # ------------------------------------------------------------------
    def _ensure_history_file(self):
        """Create the prediction history CSV file with headers if it does not exist."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        if not os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Age", "Gender", "Department", "JobRole", "MonthlyIncome",
                    "Education", "BusinessTravel", "JobSatisfaction", "EnvironmentSatisfaction",
                    "WorkLifeBalance", "OverTime", "YearsAtCompany", "TotalWorkingYears",
                    "DistanceFromHome", "PerformanceRating", "MaritalStatus",
                    "Prediction", "Confidence"
                ])

    def _save_prediction_to_history(self, employee_data, label, confidence):
        """Append one prediction record (with timestamp) to the history CSV file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(HISTORY_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    employee_data["Age"], employee_data["Gender"], employee_data["Department"],
                    employee_data["JobRole"], employee_data["MonthlyIncome"], employee_data["Education"],
                    employee_data["BusinessTravel"], employee_data["JobSatisfaction"],
                    employee_data["EnvironmentSatisfaction"], employee_data["WorkLifeBalance"],
                    employee_data["OverTime"], employee_data["YearsAtCompany"],
                    employee_data["TotalWorkingYears"], employee_data["DistanceFromHome"],
                    employee_data["PerformanceRating"], employee_data["MaritalStatus"],
                    label, confidence
                ])
        except Exception as e:
            messagebox.showerror("File Error", f"Could not save prediction to history file:\n{e}")

    def _refresh_history_table(self):
        """Reload the history Treeview table from the CSV file, applying any active search filter."""
        for row in self.history_tree.get_children():
            self.history_tree.delete(row)

        if not os.path.exists(HISTORY_CSV):
            return

        search_term = self.search_var.get().strip().lower() if hasattr(self, "search_var") else ""

        try:
            with open(HISTORY_CSV, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_text = " ".join(str(v) for v in row.values()).lower()
                    if search_term and search_term not in row_text:
                        continue
                    self.history_tree.insert("", "end", values=(
                        row.get("Timestamp", ""), row.get("Age", ""), row.get("Gender", ""),
                        row.get("Department", ""), row.get("JobRole", ""), row.get("MonthlyIncome", ""),
                        row.get("OverTime", ""), row.get("YearsAtCompany", ""),
                        row.get("Prediction", ""), row.get("Confidence", "")
                    ))
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read history file:\n{e}")

    def clear_history(self):
        """Delete all saved prediction history after user confirmation."""
        if not messagebox.askyesno("Clear History", "This will permanently delete all saved predictions. Continue?"):
            return
        try:
            self._create_fresh_history_file()
            self._refresh_history_table()
            messagebox.showinfo("History Cleared", "Prediction history has been cleared.")
            self.set_status("Prediction history cleared.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear history:\n{e}")

    def _create_fresh_history_file(self):
        """Overwrite the history CSV file with just the header row."""
        with open(HISTORY_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Age", "Gender", "Department", "JobRole", "MonthlyIncome",
                "Education", "BusinessTravel", "JobSatisfaction", "EnvironmentSatisfaction",
                "WorkLifeBalance", "OverTime", "YearsAtCompany", "TotalWorkingYears",
                "DistanceFromHome", "PerformanceRating", "MaritalStatus",
                "Prediction", "Confidence"
            ])

    def export_history(self):
        """Let the user export/save a copy of the prediction history CSV file elsewhere."""
        if not os.path.exists(HISTORY_CSV):
            messagebox.showwarning("No History", "There is no prediction history to export yet.")
            return
        dest_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="prediction_history_export.csv",
            title="Export Prediction History"
        )
        if not dest_path:
            return
        try:
            with open(HISTORY_CSV, "r") as src, open(dest_path, "w") as dst:
                dst.write(src.read())
            messagebox.showinfo("Export Successful", f"Prediction history exported to:\n{dest_path}")
            self.set_status("Prediction history exported successfully.")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not export history:\n{e}")

    # ------------------------------------------------------------------
    # Model menu actions
    # ------------------------------------------------------------------
    def train_model_action(self):
        """Train (or retrain) the Machine Learning model from the GUI."""
        proceed = messagebox.askyesno(
            "Train Model",
            "This will train all 5 candidate models on the dataset and save "
            "the best one. This may take a few moments. Continue?"
        )
        if not proceed:
            return

        self.set_status("Training model, please wait...")
        self.root.update_idletasks()

        try:
            # Imported here (rather than at the top of the file) so that the
            # GUI can still start up even in the rare case that a heavy
            # dependency required only for training is temporarily missing.
            from train_model import train_and_save
            info = train_and_save()
            self._try_load_model(show_error=False)
            messagebox.showinfo(
                "Training Complete",
                f"Training finished successfully!\n\n"
                f"Best Model: {info['model_name']}\n"
                f"Accuracy : {info['metrics']['accuracy']:.2%}\n"
                f"F1-Score : {info['metrics']['f1_score']:.2%}"
            )
            self.set_status(f"Model trained successfully. Best model: {info['model_name']}.")
        except Exception as e:
            messagebox.showerror("Training Failed", f"An error occurred during training:\n{e}")
            self.set_status("Model training failed.")

    def reload_model_action(self):
        """Reload the trained model files from disk without retraining."""
        success = self._try_load_model(show_error=True)
        if success:
            messagebox.showinfo("Model Reloaded", "The trained model was reloaded successfully.")
            self.set_status("Model reloaded successfully.")
        else:
            self.set_status("Failed to reload model.")

    def show_model_info(self):
        """Display information about the currently loaded best model."""
        if not self.model_ready:
            messagebox.showwarning("No Model Loaded", "No trained model is currently loaded.")
            return
        name = self.predictor.get_model_name()
        metrics = self.predictor.get_model_metrics()
        info_text = f"Current Model: {name}\n\n"
        for key, value in metrics.items():
            info_text += f"{key.capitalize()}: {value:.2%}\n"
        messagebox.showinfo("Model Information", info_text)

    # ------------------------------------------------------------------
    # Help / About dialogs
    # ------------------------------------------------------------------
    def show_help(self):
        """Display a Help dialog explaining how to use the application."""
        help_text = (
            "HOW TO USE THE EMPLOYEE ATTRITION PREDICTOR\n"
            "--------------------------------------------\n\n"
            "1. Go to the 'Prediction' tab.\n"
            "2. Fill in all the employee details using the provided fields, "
            "dropdown menus and spin boxes.\n"
            "3. Click 'Predict' to see whether the employee is Likely to "
            "Leave or Not Likely to Leave, along with a confidence percentage.\n"
            "4. Click 'Reset' to clear the form and start again.\n"
            "5. Every prediction is automatically saved to the 'History' tab, "
            "where you can search, review or clear past predictions.\n"
            "6. Visit the 'Graphs' tab to view how the different Machine "
            "Learning algorithms performed during training.\n"
            "7. If no trained model is found, use Model > Train / Retrain "
            "Model from the menu bar.\n"
        )
        popup = tk.Toplevel(self.root)
        popup.title("Help")
        popup.configure(bg=COLOR_BG)
        popup.geometry("560x420")
        text_widget = tk.Text(popup, bg=COLOR_BG_INPUT, fg=COLOR_TEXT, wrap="word",
                               font=("Segoe UI", 10), padx=12, pady=12)
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", help_text)
        text_widget.configure(state="disabled")

    def show_about(self):
        """Display an About dialog with project and student information."""
        about_text = (
            "Employee Attrition Predictor\n"
            "Version 1.0\n\n"
            "A Machine Learning based desktop application that predicts "
            "whether an employee is likely to leave a company, built for a "
            "university Artificial Intelligence course.\n\n"
            "Student Name : Ghazi Muhammad Abdullah\n\n"
            "Built using Python, Tkinter and scikit-learn."
        )
        messagebox.showinfo("About Employee Attrition Predictor", about_text)


def launch_app():
    """Create the Tkinter root window and start the Employee Attrition Predictor GUI."""
    root = tk.Tk()
    app = EmployeeAttritionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()


if __name__ == "__main__":
    launch_app()
