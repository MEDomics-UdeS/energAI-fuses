from tkinter import *
from PIL import Image, ImageTk
import os
from src.utils.constants import INFERENCE_PATH, IMAGE_EXT, COLOR_PALETTE, FONT_PATH

class ImageViewer:

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
        self.__frame = LabelFrame(window,
                                  background=COLOR_PALETTE["bg"],
                                  foreground=COLOR_PALETTE["fg"],
                                  highlightbackground=COLOR_PALETTE["active"],
                                  text=self.__img_list[0][0],
                                  font=(FONT_PATH, 14),
                                  padx=20,
                                  pady=20,
                                  width=1560,
                                  height=820)
        
        self.__frame.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
        
        # Keep the frame from resizing with the pictures
        self.__frame.pack_propagate(False)

        # Displaying the image in frame
        self.__image = Label(self.__frame, image=self.__img_list[0][1])
        self.__image.pack()

        # Creating the status bar
        self.__status = Label(window,
                              background=COLOR_PALETTE["bg"],
                              foreground=COLOR_PALETTE["fg"],
                              text=f'Image 1 of {len(self.__img_list)}',
                              font=(FONT_PATH, 10),
                              bd=1,
                              relief="sunken",
                              anchor="e")
        
        # Creating the navigation buttons
        self.__previous_button = Button(window,
                                        background=COLOR_PALETTE["widgets"],
                                        foreground=COLOR_PALETTE["fg"],
                                        activebackground=COLOR_PALETTE["active"],
                                        activeforeground=COLOR_PALETTE["fg"],
                                        highlightbackground=COLOR_PALETTE["active"],
                                        text=" << Prev",
                                        font=(FONT_PATH, 12),
                                        command=lambda: self.__prev_img(window, 0),
                                        state="disabled")
        
        self.__next_button = Button(window,
                                    background=COLOR_PALETTE["widgets"],
                                    foreground=COLOR_PALETTE["fg"],
                                    activebackground=COLOR_PALETTE["active"],
                                    activeforeground=COLOR_PALETTE["fg"],
                                    highlightbackground=COLOR_PALETTE["active"],
                                    text="Next >>",
                                    font=(FONT_PATH, 12),
                                    command=lambda: self.__next_img(window, 0))
        
        self.__exit_button = Button(window,
                                    background=COLOR_PALETTE["widgets"],
                                    foreground=COLOR_PALETTE["fg"],
                                    activebackground=COLOR_PALETTE["active"],
                                    activeforeground=COLOR_PALETTE["fg"],
                                    highlightbackground=COLOR_PALETTE["active"],
                                    text="Exit",
                                    font=(FONT_PATH, 12),
                                    command=window.destroy)

        # Putting the widgets on screen
        self.__previous_button.grid(row=1, column=0)
        self.__next_button.grid(row=1, column=2)
        self.__exit_button.grid(row=1, column=1, pady=10)
        self.__status.grid(row=2, column=0, columnspan=3, sticky="w"+"e")


    def __prev_img(self, window, idx):
        
        # Update the image filename
        self.__frame.config(text=self.__img_list[idx][0])
        
        # Update the image
        self.__image.config(image=self.__img_list[idx][1])

        # Update the buttons
        self.__previous_button.config(command=lambda: self.__prev_img(window, idx - 1),
                                      state="disabled" if idx == 0 else "normal")
        self.__next_button.config(command=lambda: self.__next_img(window, idx),
                                  state=NORMAL)

        # Update status bar
        self.__status.config(text=f'Image {idx + 1} of {len(self.__img_list)}')
        

    def __next_img(self, window, idx):

        # Update the image filename
        self.__frame.config(text=self.__img_list[idx + 1][0])

        # Update the image
        self.__image.config(image=self.__img_list[idx + 1][1])
        
        # Update the buttons
        self.__previous_button.config(command=lambda: self.__prev_img(window, idx),
                                      state=NORMAL)
        self.__next_button.config(command=lambda: self.__next_img(window, idx + 1),
                                  state="disabled" if idx == len(self.__img_list) - 2 else "normal")
        
        # Update status bar
        self.__status.config(text=f'Image {idx + 2} of {len(self.__img_list)}')
