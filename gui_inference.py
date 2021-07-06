from tkinter import *
import subprocess as sp

from src.gui.score_slider import Score_Slider
from src.gui.iou_slider import IoU_Slider
from src.gui.normalize_menu import Normalize_Menu
from src.gui.model_loader import Model_Loader
from src.gui.batch_slider import Batch_Slider
from src.gui.image_loader import Image_Loader

from src.utils.constants import INFERENCE_PATH

root = Tk()
root.title("Inference test")

model_ld = Model_Loader(root)
img_dir = Image_Loader(root)
norm = Normalize_Menu(root)
bs = Batch_Slider(root)
iou = IoU_Slider(root)
score = Score_Slider(root)


def start_inference(model_ld, img_dir, norm, bs, iou, score):
    
    cmd = [
        'python', 'inference_test.py',
        '--image_path', img_dir.get_img_dir(),
        '--inference_path', INFERENCE_PATH,
        '--model-file-name', model_ld.get_model(),
        '--normalize', norm.get_normalize_option(),
        '--batch', bs.get_batch_size(),
        '--iou_threshold', iou.get_iou_treshold(),
        '--score_threshold', score.get_score_treshold()
        ]
        

    # Execute current command
    p = sp.Popen(cmd)

    # Wait until the command finishes before continuing
    p.wait()
    

Button(root, text="Start", command=lambda: start_inference(model_ld, img_dir, norm, bs, iou, score)).grid(row=9, column=2, pady=25)


    
root.mainloop()

