from tkinter import *
from PIL import Image, ImageTk
import os
from src.utils.constants import INFERENCE_PATH, IMAGE_EXT

class Image_Viewer:

    def __init__(self, window) -> None:

        # Creating all the images
        self.__img_list = []

        for file in os.listdir(INFERENCE_PATH):
            if file.endswith(f'.{IMAGE_EXT}'):
                self.__img_list.append(ImageTk.PhotoImage(
                    Image.open(f'{INFERENCE_PATH}{file}').resize((600, 480))))

        self.__status = Label(window, text=f'Image 1 of {len(self.__img_list)}', bd=1, relief="sunken", anchor="e")

        self.__my_label = Label(window, image=self.__img_list[0])
        self.__my_label.grid(row=0, column=0, columnspan=3)

        self.__previous_button = Button(window, text=" << Prev", command=lambda: self.__prev_img(window, 0), state="disabled")
        self.__next_button = Button(window, text="Next >>", command=lambda: self.__next_img(window, 0))
        self.__exit_button = Button(window, text="Exit", command=window.quit)

        self.__previous_button.grid(row=1, column=0)
        self.__next_button.grid(row=1, column=2)
        self.__exit_button.grid(row=1, column=1, pady=10)
        self.__status.grid(row=2, column=0, columnspan=3, sticky="w"+"e")
    

    def __prev_img(self, window, idx):
        global my_label
        global status
        global previous_button
        global next_button

        # Update the image
        self.__my_label.grid_forget()
        self.__my_label = Label(window, image=self.__img_list[idx])
        self.__my_label.grid(row=0, column=0, columnspan=3)

        # Update the buttons
        self.__previous_button.destroy()
        self.__previous_button = Button(window, text="<< Prev", command=lambda: self.__prev_img(window, idx - 1), state="disabled" if idx == 0 else "normal")
        self.__previous_button.grid(row=1, column=0)
        self.__next_button.destroy()
        self.__next_button = Button(window, text="Next >>", command=lambda: self.__next_img(window, idx))
        self.__next_button.grid(row=1, column=2)

        # Update status bar
        self.__status.grid_forget()
        self.__status = Label(
            window, text=f'Image {idx + 1} of {len(self.__img_list)}', bd=1, relief="sunken", anchor="e")
        self.__status.grid(row=2, column=0, columnspan=3, sticky="w"+"e")

    def __next_img(self, window, idx):
        global my_label
        global status
        global previous_button
        global next_button

        # Update the image
        self.__my_label.grid_forget()
        self.__my_label = Label(window, image=self.__img_list[idx + 1])
        self.__my_label.grid(row=0, column=0, columnspan=3)

        # Update the buttons
        self.__previous_button.destroy()
        self.__previous_button = Button(window, text="<< Prev", command=lambda: self.__prev_img(window, idx))
        self.__previous_button.grid(row=1, column=0)
        
        self.__next_button.destroy()
        self.__next_button = Button(window, text="Next >>", command=lambda: self.__next_img(window, idx + 1), state="disabled" if idx == len(self.__img_list) - 2 else "normal")
        self.__next_button.grid(row=1, column=2)

        # Update status bar
        self.__status.grid_forget()
        self.__status = Label(window, text=f'Image {idx + 2} of {len(self.__img_list)}', bd=1, relief="sunken", anchor="e")
        self.__status.grid(row=2, column=0, columnspan=3, sticky="w"+"e")
