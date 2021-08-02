from tkinter import *
import subprocess as sp

from src.gui.modules.DeviceSelector import DeviceSelector
from src.gui.modules.ScoreSlider import ScoreSlider
from src.gui.modules.IOUSlider import IOUSlider
from src.gui.modules.ModelLoader import ModelLoader
from src.gui.modules.ImageLoader import ImageLoader
from src.gui.modules.JsonFileLoader import JsonFileLoader
from src.gui.ImageViewer import ImageViewer
import torch
import json
from src.utils.constants import GUI_SETTINGS

from src.utils.constants import INFERENCE_PATH, COLOR_PALETTE, FONT_PATH


class GUI(Tk):

    def __init__(self) -> None:

        # Initializing the root window
        super().__init__()
        self.geometry("630x400+0+0")
        self.configure(background=COLOR_PALETTE["bg"])

        # Setting the title
        self.title("Inference test")
        
        # Loading the defaults settings
        self.load_settings_json()

        # Putting the widgets on screen
        model_ld = ModelLoader(self)
        img_dir = ImageLoader(self)

        Label(self,
              bg=COLOR_PALETTE["bg"],
              fg=COLOR_PALETTE["fg"],
              text="Start inference test",
              font=(FONT_PATH, 14)
              ).grid(row=2, column=1)

        Button(self,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Start",
               font=(FONT_PATH, 14),
               command=lambda: self.__start_inference(model_ld, img_dir)
               ).grid(row=3, column=1)

        Button(self,
               background=COLOR_PALETTE["widgets"],
               foreground=COLOR_PALETTE["fg"],
               activebackground=COLOR_PALETTE["active"],
               activeforeground=COLOR_PALETTE["fg"],
               highlightbackground=COLOR_PALETTE["active"],
               text="Advanced options",
               font=(FONT_PATH, 12),
               command=self.open_advanced_options
               ).grid(row=3, column=2, pady=10)


    def load_settings_json(self):
        # Automatically set advanced options to default values
        with open(GUI_SETTINGS, "r+") as f_obj:
            # Replacing the file pointer at the start
            f_obj.seek(0)
            f_obj.truncate()

            # Loading in the default values for inference
            iou_treshold = "0.5"
            score_treshold = "0.5"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Creating the settings dictionnary
            settings_dict = {"iou_treshold": iou_treshold,
                             "score_treshold": score_treshold,
                             "device": device}

            # Saving the settings in json file
            json.dump(settings_dict, f_obj)


    def open_advanced_options(self):
        # Declaring the advanced options window
        advanced_options_window = Toplevel()
        advanced_options_window.title("Advanced options")
        advanced_options_window.geometry("450x400+630+0")
        advanced_options_window.config(background=COLOR_PALETTE["bg"])

        # Putting the options widgets on screen
        IOUSlider(advanced_options_window)
        ScoreSlider(advanced_options_window)
        DeviceSelector(advanced_options_window)
        JsonFileLoader(advanced_options_window)


    def click_exit_button(self, window):
        window.destroy()
        window.update()


    def __start_inference(self, model_ld, img_dir):

        with open(GUI_SETTINGS, "r") as f_obj:
            settings_dict = json.load(f_obj)

        cmd = [
            'python', 'final_product.py',
            '--image_path', img_dir.img_dir,
            '--inference_path', INFERENCE_PATH,
            '--model_file_name', model_ld.model,
            '--iou_threshold', settings_dict["iou_treshold"],
            '--score_threshold', settings_dict["score_treshold"],
            '--device', settings_dict["device"]
        ]

        # Adding the ground truth json file if one is entered by the user
        try:
            settings_dict["ground_truth"]
        except KeyError:
            pass
        else:
            cmd.extend(("--ground_truth_file", settings_dict["ground_truth"]))

        # Execute current command
        p = sp.Popen(cmd)

        # Wait until the command finishes before continuing
        p.wait()

        image_viewer_window = Toplevel()
        image_viewer_window.config(background=COLOR_PALETTE["bg"])
        image_viewer_window.resizable(False, False)

        ImageViewer(window=image_viewer_window)


if __name__ == '__main__':
    app = GUI()
    app.mainloop()
