"""
File:
    src/gui/modules/AdvancedOptionsWindow.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Single instance GUI window for advanced options
"""

from tkinter import *
from src.utils.constants import COLOR_PALETTE
from src.gui.modules.ResetButton import ResetButton
from src.gui.modules.IOUSlider import IOUSlider
from src.gui.modules.ScoreSlider import ScoreSlider
from src.gui.modules.DeviceSelector import DeviceSelector
from src.gui.modules.CSVFileLoader import CSVFileLoader
from src.gui.modules.ModelLoader import ModelLoader
from src.gui.modules.ImageLoader import ImageLoader

class AdvancedOptionsWindow:
    """Single instance GUI window for advanced options"""
    
    def __init__(self, window: Tk, model: ModelLoader, img_dir: ImageLoader) -> None:
        """Class constructor

        Args:
            window (Tk): Root window of the advanced option window
            model (ModelLoader): Object containing the trained model file path
            img_dir (ImageLoader): Object containing the raw image directory path
        """

        # Declaring the advanced options window
        advanced_options_window = Toplevel()
        advanced_options_window.grab_set()
        advanced_options_window.title("Advanced options")
        advanced_options_window.geometry(f"+{window.winfo_screenwidth() + window.winfo_x()}+{window.winfo_y()}")
        advanced_options_window.config(background=COLOR_PALETTE["bg"])

        # Putting the options widgets on screen
        ResetButton(advanced_options_window,
                    model=model,
                    img_dir=img_dir,
                    iou=IOUSlider(advanced_options_window),
                    score=ScoreSlider(advanced_options_window),
                    device=DeviceSelector(advanced_options_window),
                    gt_csv=CSVFileLoader(advanced_options_window))
