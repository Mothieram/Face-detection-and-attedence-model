"""
utils/file_picker.py
─────────────────────
File selection utility.

Opens a native OS file dialog (tkinter) for selecting an image.
Falls back gracefully to a console path prompt if tkinter is unavailable.
"""


def pick_image() -> str:
    """
    Open a native file dialog and return the selected image path.
    Falls back to console input if tkinter is unavailable.

    Returns:
        Absolute file path string, or "" if cancelled.
    """
    try:
        return _tkinter_picker()
    except Exception:
        print("[PICKER] GUI unavailable — falling back to console input.")
        return _console_picker()


def _tkinter_picker() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select a face image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return path or ""


def _console_picker() -> str:
    raw = input("Enter image path: ").strip().strip('"').strip("'")
    return raw
