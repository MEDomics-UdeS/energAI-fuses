from tkinter import *

class Batch_Slider:

    def __init__(self, root) -> None:
        self.__batch_size = None
        
        Label(root, text="", padx=100).grid(row=3, column=1)
        Label(root, text="Select the batch size").grid(row=3, column=2)
        Scale(root, from_=1, to=20, orient=HORIZONTAL, command=self.__slide).grid(row=4, column=2)

    def __slide(self, value):
        self.__batch_size = value
    
    def get_batch_size(self):
        return str(self.__batch_size)

