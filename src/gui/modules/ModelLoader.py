from tkinter import *
from tkinter import filedialog
import json
from src.utils.constants import FONT_PATH, MODELS_PATH, COLOR_PALETTE, GUI_SETTINGS

class ModelLoader:

    def __init__(self, root) -> None:
        
        Label(root,
              background=COLOR_PALETTE["bg"],
              foreground=COLOR_PALETTE["fg"],
              text="Select your model",
              font=(FONT_PATH, 14),
              width=50
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

        with open(GUI_SETTINGS, "r") as f_obj:

            try:
                self.__model_label = Label(root,
                                           background=COLOR_PALETTE["bg"],
                                           foreground=COLOR_PALETTE["purple"],
                                           text=f'{json.load(f_obj)["model"].split(sep="/")[-1]} selected',
                                           font=(FONT_PATH, 14),
                                           height=2,
                                           width=50,
                                           justify=CENTER)
                
            except KeyError:
                self.__model_label = Label(root,
                                           background=COLOR_PALETTE["bg"],
                                           foreground=COLOR_PALETTE["red"],
                                           text="No model selected",
                                           font=(FONT_PATH, 14),
                                           height=2,
                                           width=50,
                                           justify=CENTER)
            # Putting the label on screen
            self.__model_label.grid(row=2, column=0)

        
    def __select_model(self, root):
        root.filename = filedialog.askopenfile(
            initialdir=MODELS_PATH, title="Select a model")

        if root.filename:
            self.__model_label.config(foreground=COLOR_PALETTE["purple"],
                                      text=f'{root.filename.name.split(sep="/")[-1]} selected')
            with open(GUI_SETTINGS, "r+") as f_obj:
                settings_dict = json.load(f_obj)
                f_obj.seek(0)
                f_obj.truncate()

                settings_dict["model"] = root.filename.name
                json.dump(settings_dict, f_obj)

    @property
    def model_label(self):
        return self.__model_label