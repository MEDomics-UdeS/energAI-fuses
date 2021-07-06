from tkinter import *


class Score_Slider:

    def __init__(self, root) -> None:
        Label(root, text="", padx=100).grid(row=7, column=1)
        Label(root, text="Select the score treshold").grid(row=7, column=2)
        Scale(root, from_=0.0, to=1.0, orient=HORIZONTAL, resolution=0.1,
              command=self.__slide).grid(row=8, column=2)

    def __slide(self, value):
        self.__score_treshold = value

    def get_score_treshold(self):
        return str(self.__score_treshold)
