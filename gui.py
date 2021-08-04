from tkinter import *
import os
import json
from tkinter.messagebox import showerror

from src.gui.modules.DeviceSelector import DeviceSelector
from src.gui.modules.ScoreSlider import ScoreSlider
from src.gui.modules.IOUSlider import IOUSlider
from src.gui.modules.ModelLoader import ModelLoader
from src.gui.modules.ImageLoader import ImageLoader
from src.gui.modules.JsonFileLoader import JsonFileLoader
from src.gui.modules.ResetButton import ResetButton
from src.gui.ImageViewer import ImageViewer
from src.gui.modules.OuputRedirector import OutputRedirector

from src.utils.constants import GUI_SETTINGS
from src.utils.constants import INFERENCE_PATH, COLOR_PALETTE, FONT_PATH
from src.utils.helper_functions import enter_default_json


class GUI(Tk):

    def __init__(self) -> None:

        # Initializing the root window
        super().__init__()
        self.geometry("+0+0")
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
               command=lambda: self.__start_inference()
               ).grid(row=4, column=1)

        Button(self,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Advanced options",
               font=(FONT_PATH, 12),
               command=self.open_advanced_options
               ).grid(row=4, column=2, pady=10)

        frame = LabelFrame(self,
                   background=COLOR_PALETTE["bg"],
                   foreground=COLOR_PALETTE["fg"],
                   text="Application output",
                   font=(FONT_PATH, 14),
                   width=860,
                   height=200
        )
        frame.grid(row=5, column=0, columnspan=3, padx=20, pady=20)
        frame.grid_propagate(False)
        
        scroll = Scrollbar(frame,
                           activebackground=COLOR_PALETTE["purple"],
                           background=COLOR_PALETTE["widgets"],
                           highlightbackground=COLOR_PALETTE["bg"],
                           troughcolor=COLOR_PALETTE["active"],
                           orient=VERTICAL)
        scroll.pack(side=RIGHT, fill=Y)
        
        self.__text = Text(frame,
                           background=COLOR_PALETTE["widgets"],
                           foreground=COLOR_PALETTE["green"],
                           highlightbackground=COLOR_PALETTE["bg"],
                           insertbackground=COLOR_PALETTE["fg"],
                           selectbackground=COLOR_PALETTE["purple"],
                           width=115,
                           height=12,
                           wrap=WORD,
                           yscrollcommand=scroll.set)
        self.__text.pack(side=LEFT, fill=BOTH)
        
        scroll.config(command=self.__text.yview)


    def create_json_file(self):
        with open(GUI_SETTINGS, "a+") as f_obj:
            enter_default_json(f_obj)


    def open_advanced_options(self):
        # Declaring the advanced options window
        advanced_options_window = Toplevel()
        advanced_options_window.title("Advanced options")
        advanced_options_window.geometry(f"+{self.winfo_screenwidth()}+0")
        advanced_options_window.config(background=COLOR_PALETTE["bg"])

        # Putting the options widgets on screen
        self.__iou = IOUSlider(advanced_options_window)
        self.__score = ScoreSlider(advanced_options_window)
        self.__device = DeviceSelector(advanced_options_window)
        self.__gt_json = JsonFileLoader(advanced_options_window)
        self.reset = ResetButton(advanced_options_window,
                                 model=self.__model_ld,
                                 imgdir=self.__img_dir,
                                 iou=self.__iou,
                                 score=self.__score,
                                 device=self.__device,
                                 gt_json=self.__gt_json)


    def __check_for_errors(self, settings):
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
    
    
    def __start_inference(self):

        with open(GUI_SETTINGS, "r") as f_obj:
            settings_dict = json.load(f_obj)
        
        
        if self.__check_for_errors(settings_dict):
            showerror(title="Error",
                      message=self.__check_for_errors(settings_dict))
        else:

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

            # Execute current command
            OutputRedirector(self, self.__text, cmd)
            
            # p = Popen(cmd)

            # # # Wait until the command finishes before continuing
            # p.wait()

            image_viewer_window = Toplevel()
            image_viewer_window.geometry("1600x926")
            image_viewer_window.config(background=COLOR_PALETTE["bg"])
            image_viewer_window.resizable(False, False)

            ImageViewer(window=image_viewer_window)


if __name__ == '__main__':
    app = GUI()
    
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()
