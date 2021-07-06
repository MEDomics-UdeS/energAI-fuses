from tkinter import *


class IoU_Slider:

    def __init__(self, root) -> None:
        self.__iou_treshold = 0.1
        
        Label(root, text="", padx=100).grid(row=5, column=1)
        Label(root, text="Select the IoU treshold").grid(row=5, column=2)
        Scale(root, from_=0.1, to=1.0, orient=HORIZONTAL, resolution=0.1,
              command=self.__slide).grid(row=6, column=2)

    def __slide(self, value):
        self.__iou_treshold = value

    def get_iou_treshold(self):
        return str(self.__iou_treshold)
