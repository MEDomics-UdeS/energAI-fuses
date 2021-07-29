from tkinter import *
import subprocess as sp

from src.gui.modules.DeviceSelector import DeviceSelector
from src.gui.modules.ScoreSlider import ScoreSlider
from src.gui.modules.IOUSlider import IOUSlider
from src.gui.modules.ModelLoader import ModelLoader
from src.gui.modules.ImageLoader import ImageLoader
from src.gui.ImageViewer import ImageViewer
import torch
import json
from src.utils.constants import GUI_SETTINGS

from src.utils.constants import INFERENCE_PATH


def load_settings_json():
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


def open_advanced_options():
    # Declaring the advanced options window
    advanced_options_window = Toplevel()
    advanced_options_window.title("Advanced options")

    # Putting the options widgets on screen
    IOUSlider(advanced_options_window)
    ScoreSlider(advanced_options_window)
    DeviceSelector(advanced_options_window)
    Button(advanced_options_window, text="Exit", command=advanced_options_window.quit).grid(row=7, column=0, pady=10, padx=10)


def start_inference(model_ld, img_dir):

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

    # Execute current command
    p = sp.Popen(cmd)

    # Wait until the command finishes before continuing
    p.wait()

    image_viewer_window = Toplevel()
    image_viewer_window.geometry("1600x900")
    image_viewer_window.resizable(False, False)

    ImageViewer(window=image_viewer_window)


if __name__ == '__main__':
    
    # Declaring the main window
    root = Tk()
    root.title("Inference test")

    # Loading the defaults settings
    load_settings_json()

    # Putting the widgets on screen
    model_ld = ModelLoader(root)
    img_dir = ImageLoader(root)
    
    Label(root, text="Start inference test").grid(row=2, column=1, pady=10)
    Button(root, text="Start", padx=10, pady=10, command=lambda: start_inference(model_ld, img_dir)).grid(row=3, column=1, pady=10)
    Button(root, text="Advanced options", pady=10, padx=10, command=open_advanced_options).grid(row=3, column=2, pady=10)

    root.mainloop()
