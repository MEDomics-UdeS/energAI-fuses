"""
File:
    gui.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Small cross platform GUI application to visualize a model's inference results
"""

from tkinter import *
import os
import json
from tkinter.messagebox import showerror

from src.gui.modules.ModelLoader import ModelLoader
from src.gui.modules.ImageLoader import ImageLoader
from src.gui.modules.ReadOnlyTextBox import ReadOnlyTextBox
from src.gui.modules.OuputRedirector import OutputRedirector
from src.gui.modules.AdvancedOptionsWindow import AdvancedOptionsWindow

from src.utils.constants import INFERENCE_PATH, COLOR_PALETTE, FONT_PATH, GUI_SETTINGS
from src.utils.helper_functions import enter_default_json


class GUI(Tk):
    """Small cross platform GUI application to visualize a model's inference results

    Notes:
        Since the settings are stored in src/gui/gui_settings.json, they will be kept 
        across reruns of this script
    """

    def __init__(self) -> None:
        """Class constructor"""

        # Initializing the root window
        super().__init__()
        self.configure(background=COLOR_PALETTE["bg"])

        # Setting the title
        self.title("EnergAI-fuses GUI")
        
        # Looking for user settings
        if os.path.isfile(GUI_SETTINGS) is False:
            self.create_json_file()

        # Putting the widgets on screen
        self.__model_ld = ModelLoader(self)
        self.__img_dir = ImageLoader(self)

        Label(self,
              bg=COLOR_PALETTE["bg"],
              fg=COLOR_PALETTE["fg"],
              text="Start inference test",
              font=(FONT_PATH, 14),
              width=40
              ).grid(row=3, column=1)

        Button(self,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Start",
               font=(FONT_PATH, 14),
               command=self.__start_inference
               ).grid(row=4, column=1)

        Button(self,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Advanced options",
               font=(FONT_PATH, 12),
               command=lambda: AdvancedOptionsWindow(window=self, 
                                                     model=self.__model_ld, 
                                                     img_dir=self.__img_dir)
               ).grid(row=4, column=2, pady=10)

        self.__frame = LabelFrame(self,
                   background=COLOR_PALETTE["bg"],
                   foreground=COLOR_PALETTE["fg"],
                   text="Application output",
                   font=(FONT_PATH, 14),
                   width=860,
                   height=200
        )
        self.__frame.grid(row=5, column=0, columnspan=3, padx=20, pady=20)
        self.__frame.grid_propagate(False)
        
        self.__textbox = ReadOnlyTextBox(window=self.__frame)

    def create_json_file(self) -> None:
        """Creates the default GUI settings JSON file"""
        
        with open(GUI_SETTINGS, "a+") as f_obj:
            enter_default_json(f_obj)

    def __check_for_errors(self, settings: dict) -> str:
        """Checks if every mandatory settings are initialized by the user

        Args:
            settings (dict): The current state of the GUI settings JSON dictionnary

        Returns:
            str: An error message to be displayed if some settings are missing
        """
        
        error_message = ""
        
        if "model" not in settings:
            error_message += "No model selected.\n"
        if "imgdir" not in settings:
            error_message += "No image directory selected.\n"
        if "iou_treshold" not in settings:
            error_message += "No IoU treshold selected. Please see advanced options.\n"
        if "score_treshold" not in settings:
            error_message += "No score treshold selected. Please see advanced options.\n"
        if "device" not in settings:
            error_message += "No device selected. Please see advanced options.\n"
        
        return error_message
    
    def __start_inference(self) -> None:
        """Starts the inference"""
        
        # Load the user settings
        with open(GUI_SETTINGS, "r") as f_obj:
            settings_dict = json.load(f_obj)
        
        # Look for missing elements in the user settings
        if self.__check_for_errors(settings_dict):
            showerror(title="Error",
                      message=self.__check_for_errors(settings_dict))
        else:
            # Create the subprocess command
            cmd = [
                'python', 'final_product.py',
                '--image_path', settings_dict["imgdir"],
                '--inference_path', INFERENCE_PATH,
                '--model_file_name', settings_dict["model"],
                '--iou_threshold', settings_dict["iou_treshold"],
                '--score_threshold', settings_dict["score_treshold"],
                '--device', settings_dict["device"]
            ]

            # Adding the ground truth json file if one is entered by the user
            try:
                settings_dict["ground_truth"]
                cmd.extend(("--ground_truth_file", settings_dict["ground_truth"]))
            except KeyError:
                pass

            # Execute the current command in an output redirector
            OutputRedirector(self, self.__textbox, cmd)


if __name__ == '__main__':
    # This fixes a multithreading error with torch
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # Starts the GUI application
    app = GUI()
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()
