from tkinter import *
from src.gui.normalize_menu import Normalize_Menu
from src.gui.model_loader import Model_Loader
from src.gui.batch_slider import Batch_Slider

root = Tk()
root.title("Inference test")

model_ld = Model_Loader(root)
norm_menu = Normalize_Menu(root)
bs = Batch_Slider(root)

    
root.mainloop()

