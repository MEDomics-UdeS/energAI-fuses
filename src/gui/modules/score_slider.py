from tkinter import *


class Score_Slider:

    def __init__(self, root) -> None:
        
        Label(root, text="", pady=20).grid(row=2, column=2)
        Label(root, text="Select the score treshold").grid(row=3, column=2, padx=20, pady=10)
        slider = Scale(root, from_=0.1, to=1.0, orient=HORIZONTAL, resolution=0.1,
              command=self.__slide)
        slider.set(0.5)
        slider.grid(row=4, column=2, pady=10)

    def __slide(self, value):
        self.__score_treshold = value

    @property
    def value(self):
        return str(self.__score_treshold)
