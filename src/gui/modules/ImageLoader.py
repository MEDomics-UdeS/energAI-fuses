from tkinter import *
from tkinter import filedialog
import json

from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS
from src.utils.helper_functions import cross_platform_path_split

class ImageLoader:
    
    def __init__(self, root: Tk) -> None:
        
        Label(root,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select your image directory",
              font=(FONT_PATH, 14),
              width=50
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

        with open(GUI_SETTINGS, "r") as f_obj:

            try:
                img_dir = cross_platform_path_split(json.load(f_obj)["imgdir"])

                self.__img_dir_label = Label(root,
                                             background=COLOR_PALETTE["bg"],
                                             foreground=COLOR_PALETTE["purple"],
                                             text=f'{".../" if len(img_dir) > 3 else ""}{"/".join(img_dir[-2:])} selected',
                                             font=(FONT_PATH, 14),
                                             height=2,
                                             width=50,
                                             justify=CENTER)

            except KeyError:
                self.__img_dir_label = Label(root,
                                             background=COLOR_PALETTE["bg"],
                                             foreground=COLOR_PALETTE["red"],
                                             text="No image directory selected",
                                             font=(FONT_PATH, 14),
                                             height=2,
                                             width=50,
                                             justify=CENTER)
            # Putting the label on screen
            self.__img_dir_label.grid(row=2, column=2)
        
    def __select_img_dir(self, root: Tk) -> None:
        root.filename = filedialog.askdirectory(
            initialdir='.', title="Select a directory for inference pass")

        if root.filename:
            img_dir = cross_platform_path_split(root.filename)
            self.__img_dir_label.config(foreground=COLOR_PALETTE["purple"],
                                        text=f'{".../" if len(img_dir) > 3 else ""}{"/".join(img_dir[-2:])} selected')

            with open(GUI_SETTINGS, "r+") as f_obj:
                settings_dict = json.load(f_obj)
                f_obj.seek(0)
                f_obj.truncate()

                settings_dict["imgdir"] = root.filename
                json.dump(settings_dict, f_obj)

    @property
    def img_dir_label(self):
        return self.__img_dir_label
