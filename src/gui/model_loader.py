from tkinter import *
from tkinter import filedialog
from src.utils.constants import MODELS_PATH

class Model_Loader:

    def __init__(self, root) -> None:
        Label(root, text="Select your model").grid(row=0, column=0)
        model_button = Button(root, text="Select", command=lambda: self.__select_model(root))
        model_button.grid(row=1, column=0)
        
    def __select_model(self, root):
        root.filename = filedialog.askopenfile(
            initialdir=MODELS_PATH, title="Select a model")

        if root.filename:
            self.__model = root.filename.name

    def get_model(self):
        return self.__model