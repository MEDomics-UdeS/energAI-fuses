from tkinter import *
from tkinter import filedialog
from src.utils.constants import FONT_PATH, MODELS_PATH, COLOR_PALETTE

class ModelLoader:

    def __init__(self, root) -> None:
        
        Label(root,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select your model",
              font=(FONT_PATH, 14)
              ).grid(row=0, column=0, padx=10, pady=10)
        
        Button(root,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Select",font=(FONT_PATH, 12),
               command=lambda: self.__select_model(root)
               ).grid(row=1, column=0, padx=10)
        
    def __select_model(self, root):
        root.filename = filedialog.askopenfile(
            initialdir=MODELS_PATH, title="Select a model")

        if root.filename:
            self.__model = root.filename.name

    @property
    def model(self):
        return self.__model
