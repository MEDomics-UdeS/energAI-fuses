from tkinter import *

class Normalize_Menu:

    def __init__(self, root) -> None:
        self.__normalize_option = StringVar()

        normalize = [('precalculated', 'precalculated'),
                    ('calculated', 'calculated'),
                    ('disabled', 'disabled')]

        Label(root,
              text="").grid(row=2, column=0, pady=10)
        
        Label(root, 
              text="Normalize the training dataset by mean & std").grid(row=3, column=0)

        for i, (option, value) in enumerate(normalize):
            Radiobutton(root, 
                        text=option,
                        variable=self.__normalize_option,
                        value=value).grid(row=i + 4, column=0)

    def get_normalize_option(self):
        return self.__normalize_option.get()