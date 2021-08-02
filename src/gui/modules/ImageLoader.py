from tkinter import *
from tkinter import filedialog

from src.utils.constants import COLOR_PALETTE, FONT_PATH

class ImageLoader:
    
    def __init__(self, root) -> None:
        
        Label(root,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select your image directory",
              font=(FONT_PATH, 14)
              ).grid(row=0, column=2, padx=10, pady=10)
        
        Button(root,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Select",
               font=(FONT_PATH, 12),
               command=lambda: self.__select_img_dir(root)
               ).grid(row=1, column=2, padx=10)
        
    def __select_img_dir(self, root):
        root.filename = filedialog.askdirectory(
            initialdir='.', title="Select a directory for inference pass")

        if root.filename:
            self.__img_dir = root.filename

    @property
    def img_dir(self):
        return self.__img_dir
