"""
File:
    src/gui/modules/ModelLoader.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Class responsible of handling the trained model file path
"""

from tkinter import *
from tkinter import filedialog
import json

from src.utils.constants import FONT_PATH, MODELS_PATH, COLOR_PALETTE, GUI_SETTINGS
from src.utils.helper_functions import cp_split


class ModelLoader:
    """Class responsible of handling the trained model file path"""
    def __init__(self,
                 window: Tk) -> None:
        """Class constructor

        Args:
            window(Tk): Root window
        Notes:
            A model must be selected by the user in order to visualize the inference results.
            By default, none will be selected.
            
        """
        
        Label(window,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select your model",
              font=(FONT_PATH, 14),
              width=50
              ).grid(row=0, column=0, padx=10, pady=10)
        
        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Select",font=(FONT_PATH, 12),
               command=lambda: self.__select_model(window)
               ).grid(row=1, column=0, padx=10)

        # Reading the settings JSON file
        with open(GUI_SETTINGS, "r") as f_obj:
            try:
                self.__model_label = Label(window,
                                           background=COLOR_PALETTE["bg"],
                                           foreground=COLOR_PALETTE["purple"],
                                           text=f'{cp_split(json.load(f_obj)["model"])[-1]} selected',
                                           font=(FONT_PATH, 14),
                                           height=2,
                                           width=50,
                                           justify=CENTER)
                
            except KeyError:
                self.__model_label = Label(window,
                                           background=COLOR_PALETTE["bg"],
                                           foreground=COLOR_PALETTE["red"],
                                           text="No model selected",
                                           font=(FONT_PATH, 14),
                                           height=2,
                                           width=50,
                                           justify=CENTER)
            # Putting the label on screen
            self.__model_label.grid(row=2, column=0)

    def __select_model(self,
                       window: Tk) -> None:
        """Opens a file manager window to select the model

        Args:
            window(Tk): Root window

        """
        
        window.filename = filedialog.askopenfile(
            initialdir=MODELS_PATH, title="Select a model")

        if window.filename:
            self.__model_label.config(foreground=COLOR_PALETTE["purple"],
                                      text=f'{cp_split(window.filename.name)[-1]} selected')
        
            with open(GUI_SETTINGS, "r+") as f_obj:
                settings_dict = json.load(f_obj)
                f_obj.seek(0)
                f_obj.truncate()

                settings_dict["model"] = window.filename.name
                json.dump(settings_dict, f_obj)

    @property
    def model_label(self) -> Label:
        """Get the model label widget"""
        
        return self.__model_label
