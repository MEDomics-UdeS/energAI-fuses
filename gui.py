from tkinter import *
import subprocess as sp

from src.gui.modules.score_slider import Score_Slider
from src.gui.modules.iou_slider import IoU_Slider
from src.gui.modules.normalize_menu import Normalize_Menu
from src.gui.modules.model_loader import Model_Loader
from src.gui.modules.batch_slider import Batch_Slider
from src.gui.modules.image_loader import Image_Loader

from src.utils.constants import INFERENCE_PATH

root = Tk()
root.title("Inference test")

model_ld = Model_Loader(root)
img_dir = Image_Loader(root)
norm = Normalize_Menu(root)
bs = Batch_Slider(root)
iou = IoU_Slider(root)
score = Score_Slider(root)

Label(root, text="").grid(row=9, column=2, pady=25)
Label(root, text="Start inference test").grid(row=10, column=2, pady=5)
Button(root, text="Start", command=lambda: start_inference(model_ld, img_dir, norm, bs, iou, score)).grid(row=11, column=2)


def start_inference(model_ld, img_dir, norm, bs, iou, score):

    cmd = [
        'python', 'final_product.py',
        '--image_path', img_dir.get_img_dir(),
        '--inference_path', INFERENCE_PATH,
        '--model_file_name', model_ld.get_model(),
        '--normalize', norm.get_normalize_option(),
        '--batch', bs.get_batch_size(),
        '--iou_threshold', iou.get_iou_treshold(),
        '--score_threshold', score.get_score_treshold()
        ]
        

    # Execute current command
    p = sp.Popen(cmd)

    # Wait until the command finishes before continuing
    p.wait()

    
root.mainloop()
