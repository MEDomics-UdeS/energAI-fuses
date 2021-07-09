from tkinter import *


class IoU_Slider:

    def __init__(self, root) -> None:
        self.__iou_treshold = 0.1
        
        Label(root, text="", pady=10).grid(row=2, column=0)
        Label(root, text="Select the IoU treshold").grid(row=3, column=0, padx=20, pady=10)
        Scale(root, from_=0.1, to=1.0, orient=HORIZONTAL, resolution=0.1,
              command=self.__slide).grid(row=4, column=0)

    def __slide(self, value):
        self.__iou_treshold = value

    def get_iou_treshold(self):
        return str(self.__iou_treshold)
