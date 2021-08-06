from tkinter import *
from tkinter import filedialog
import json
from src.utils.constants import COLOR_PALETTE, FONT_PATH, GUI_SETTINGS

class JsonFileLoader:

    def __init__(self, window: Toplevel) -> None:

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

        with open(GUI_SETTINGS, "r") as f_obj:

            try:
                self.__json_label = Label(window,
                                           background=COLOR_PALETTE["bg"],
                                           foreground=COLOR_PALETTE["purple"],
                                           text=f'{json.load(f_obj)["ground_truth"].split(sep="/")[-1]} selected',
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
                                            text="Remove JSON",
                                            font=(FONT_PATH, 12),
                                            command=self.__remove_file)

            # Putting the button on screen if a json file is already given
            if self.__json_label["text"]:
                self.__remove_button.grid(row=4, column=1)
            
            # Putting the label on screen
            self.__json_label.grid(row=2, column=1)

    def __select_file(self, window: Toplevel) -> None:
        window.filename = filedialog.askopenfile(
            initialdir=".", title="Select a ground truth JSON file", filetypes=[("JSON files", "*.json")])

        if window.filename:
            self.__json_label.config(text=f'{window.filename.name.split(sep="/")[-1]} selected')

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
        # Removing the label since no json file is selected
        self.__json_label.config(text="")

        # Removing the button from screen
        self.__remove_button.grid_forget()
