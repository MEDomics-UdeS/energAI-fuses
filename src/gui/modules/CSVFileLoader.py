"""
File:
    src/gui/modules/CSVFileLoader.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Class responsible of handling the ground truth CSV file that the user may provide
"""

from tkinter import *
from tkinter import filedialog
import json

from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS
from src.utils.helper_functions import cp_split


class CSVFileLoader:
    """
    Class responsible of handling the ground truth CSV file that the user may provide
    """
    def __init__(self,
                 window: Toplevel) -> None:
        """Class constructor

        Args:
            window (Toplevel): Root window

        Notes:
            By default, no CSV file will be selected
        """

        Label(window, 
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Ground truth file",
              font=(FONT_PATH, 14),
              width=30
              ).grid(row=0, column=1, padx=10, pady=10)
        
        Button(window,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Select",
               font=(FONT_PATH, 12),
               command=lambda: self.__select_file(window)
               ).grid(row=1, column=1, padx=10)

        # Reading the settings JSON file
        with open(GUI_SETTINGS, "r") as f_obj:

            try:
                self.__json_label = Label(window,
                                          background=COLOR_PALETTE["bg"],
                                          foreground=COLOR_PALETTE["purple"],
                                          text=f'{cp_split(json.load(f_obj)["ground_truth"])[-1]} selected',
                                          font=(FONT_PATH, 14),
                                          width=30,
                                          justify=CENTER)

            except KeyError:
                self.__json_label = Label(window,
                                          background=COLOR_PALETTE["bg"],
                                          foreground=COLOR_PALETTE["purple"],
                                          text="",
                                          font=(FONT_PATH, 14),
                                          width=30,
                                          justify=CENTER)

            # Creating a button to remove the json file if desired
            self.__remove_button = Button(window,
                                          background=COLOR_PALETTE["widgets"],
                                          foreground=COLOR_PALETTE["fg"],
                                          activebackground=COLOR_PALETTE["active"],
                                          activeforeground=COLOR_PALETTE["fg"],
                                          highlightbackground=COLOR_PALETTE["active"],
                                          text="Remove CSV",
                                          font=(FONT_PATH, 12),
                                          command=self.__remove_file)

            # Putting the button on screen if a json file is already given
            if self.__json_label["text"]:
                self.__remove_button.grid(row=4, column=1)
            
            # Putting the label on screen
            self.__json_label.grid(row=2, column=1)

    def __select_file(self,
                      window: Toplevel) -> None:
        """Opens a file manager window to select the ground truth CSV file

        Args:
            window (Toplevel): Root window

        Notes:
            The file manager will only display directories and CSV files. If a directory 
            contains other file types, they will be ignored
        """
        window.filename = filedialog.askopenfile(
            initialdir=".", title="Select a ground truth CSV file", filetypes=[("CSV files", "*.csv")])

        if window.filename:
            self.__json_label.config(text=f'{cp_split(window.filename.name)[-1]} selected')

            # Overwriting the settings json file
            with open(GUI_SETTINGS, "r+") as f_obj:
                settings_dict = json.load(f_obj)
                f_obj.seek(0)
                f_obj.truncate()

                settings_dict["ground_truth"] = window.filename.name
                json.dump(settings_dict, f_obj)

            # Putting the remove json file button on screen
            self.__remove_button.grid(row=4, column=1)

    def __remove_file(self) -> None:
        """Removes the selected ground truth CSV file"""
        
        # Overwriting the settings json file
        with open(GUI_SETTINGS, "r+") as f_obj:
            settings_dict = json.load(f_obj)
            f_obj.seek(0)
            f_obj.truncate()

            del settings_dict["ground_truth"]
            json.dump(settings_dict, f_obj)

            # Remove the displayed widgets
            self.reset()

    def reset(self) -> None:
        """Removes the CSV file label and the remove button"""
        
        # Removing the label since no json file is selected
        self.__json_label.config(text="")

        # Removing the button from screen
        self.__remove_button.grid_forget()
