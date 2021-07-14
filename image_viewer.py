from tkinter import *
from PIL import Image, ImageTk
import os
from src.utils.constants import INFERENCE_PATH, IMAGE_EXT

class Image_Viewer:

    def __init__(self, window) -> None:

        # Declaring the maximum dimensions of the image of 16:9 aspect ratio
        max_size = 1350, 760

        # Creating all the images
        self.__img_list = []

        for file in os.listdir(INFERENCE_PATH):
            if file.endswith(f'.{IMAGE_EXT}'):
                image = Image.open(f'{INFERENCE_PATH}{file}')

                # Resizes the image and keeps aspect ratio
                image.thumbnail(max_size, Image.ANTIALIAS)

                # Add the resized image to the list
                self.__img_list.append((file, ImageTk.PhotoImage(image)))

        # Sorts the images alphabetically by filename
        self.__img_list.sort(key=lambda x: x[0])

        # Putting a frame on screen to display images into
        self.__frame = LabelFrame(window, text=self.__img_list[0][0], padx=20, pady=20, width=1560, height=820)
        self.__frame.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
        self.__frame.pack_propagate(False)

        # Displaying the directory contents
        self.__status = Label(window, text=f'Image 1 of {len(self.__img_list)}', bd=1, relief="sunken", anchor="e")

        # Displaying the image in frame
        self.__label = Label(self.__frame, image=self.__img_list[0][1])
        self.__label.pack()

        self.__previous_button = Button(window, text=" << Prev", command=lambda: self.__prev_img(window, 0), state="disabled")
        self.__next_button = Button(window, text="Next >>", command=lambda: self.__next_img(window, 0))
        self.__exit_button = Button(window, text="Exit", command=window.quit)

        self.__previous_button.grid(row=1, column=0)
        self.__next_button.grid(row=1, column=2)
        self.__exit_button.grid(row=1, column=1, pady=10)
        self.__status.grid(row=2, column=0, columnspan=3, sticky="w"+"e")


    def __prev_img(self, window, idx):

        self.__frame.grid_forget()
        self.__frame = LabelFrame(window, text=self.__img_list[idx][0], padx=20, pady=20, width=1560, height=820)
        self.__frame.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
        self.__frame.pack_propagate(False)

        # Update the image
        self.__label.grid_forget()
        self.__label = Label(self.__frame, image=self.__img_list[idx][1])
        self.__label.pack()

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

        self.__frame.grid_forget()
        self.__frame = LabelFrame(window, text=self.__img_list[idx + 1][0], padx=20, pady=20, width=1560, height=820)
        self.__frame.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
        self.__frame.pack_propagate(False)

        # Update the image
        self.__label.grid_forget()
        self.__label = Label(self.__frame, image=self.__img_list[idx + 1][1])
        self.__label.pack()

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
