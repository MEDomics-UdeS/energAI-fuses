"""
File:
    src/gui/modules/ImageLoader.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Class responsible of handling the raw image directory path
"""

from tkinter import *
from tkinter import filedialog
import json

from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS
from src.utils.helper_functions import cp_split

class ImageLoader:
    """Class responsible of handling the raw image directory path"""
    
    def __init__(self, window: Tk) -> None:
        """Class constructor

        Args:
            window (Tk): Root window

        Notes:
            A image directory must be selected by the user in order to visualize the inference results.
            By default, none will be selected.
        """
        
        Label(window,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select your image directory",
              font=(FONT_PATH, 14),
              width=50
              ).grid(row=0, column=2, padx=10, pady=10)
        
        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Select",
               font=(FONT_PATH, 12),
               command=lambda: self.__select_img_dir(window)
               ).grid(row=1, column=2, padx=10)
        
        # Reading the settings JSON file
        with open(GUI_SETTINGS, "r") as f_obj:

            try:
                img_dir = cp_split(json.load(f_obj)["imgdir"])

                self.__img_dir_label = Label(window,
                                             background=COLOR_PALETTE["bg"],
                                             foreground=COLOR_PALETTE["purple"],
                                             text=f'{".../" if len(img_dir) > 3 else ""}{"/".join(img_dir[-2:])} selected',
                                             font=(FONT_PATH, 14),
                                             height=2,
                                             width=50,
                                             justify=CENTER)

            except KeyError:
                self.__img_dir_label = Label(window,
                                             background=COLOR_PALETTE["bg"],
                                             foreground=COLOR_PALETTE["red"],
                                             text="No image directory selected",
                                             font=(FONT_PATH, 14),
                                             height=2,
                                             width=50,
                                             justify=CENTER)
            # Putting the label on screen
            self.__img_dir_label.grid(row=2, column=2)
        
    def __select_img_dir(self, window: Tk) -> None:
        """Opens a file manager window to select the raw image directory

        Args:
            window (Tk): Root window
        """
        
        window.filename = filedialog.askdirectory(
            initialdir='.', title="Select a directory for inference pass")

        if window.filename:
            img_dir = cp_split(window.filename)
            self.__img_dir_label.config(foreground=COLOR_PALETTE["purple"],
                                        text=f'{".../" if len(img_dir) > 3 else ""}{"/".join(img_dir[-2:])} selected')

            with open(GUI_SETTINGS, "r+") as f_obj:
                settings_dict = json.load(f_obj)
                f_obj.seek(0)
                f_obj.truncate()

                settings_dict["imgdir"] = window.filename
                json.dump(settings_dict, f_obj)

    @property
    def img_dir_label(self):
        """Get the raw image directory path label widget"""
        
        return self.__img_dir_label
