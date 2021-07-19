from tkinter import *
from tkinter.ttk import Radiobutton

class DeviceSelector:

    def __init__(self, window) -> None:

        self.__device_option = StringVar()

        devices = [('cuda', 'cuda'),
                    ('cpu', 'cpu')]


        Label(window, text="Select device for inference").grid(row=5, column=0, padx=20, pady=20)

        for i, (option, value) in enumerate(devices):
            Radiobutton(window,
                        text=option,
                        variable=self.__device_option,
                        value=value).grid(row=i + 6, column=0)

    @property
    def device(self):
        return self.__device_option.get()
