from tkinter import *
from tkinter import filedialog
from src.utils.constants import RESIZED_PATH

class Image_Loader:
    
    def __init__(self, root) -> None:
        Label(root, text="", padx=100).grid(row=0, column=1)
        Label(root, text="Select your image directory").grid(row=0, column=2, padx=20, pady=20)
        img_button = Button(root, text="Select",
                              command=lambda: self.__select_img_dir(root))
        img_button.grid(row=1, column=2, padx=20)
        
    def __select_img_dir(self, root):
        root.filename = filedialog.askdirectory(
            initialdir='.', title="Select a directory for inference pass")

        if root.filename:
            self.__img_dir = root.filename

    def get_img_dir(self):
        return self.__img_dir
