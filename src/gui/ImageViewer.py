from tkinter import *
from PIL import Image
import os
from src.utils.constants import INFERENCE_PATH, IMAGE_EXT, COLOR_PALETTE, FONT_PATH
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from src.gui.modules.ReadOnlyTextBox import ReadOnlyTextBox


class ImageViewer:

    def __init__(self, window: Toplevel, textbox: ReadOnlyTextBox) -> None:

        self.__fig_list = []
        self.__canvas = None
        self.__toolbar = None

        # Creating all the images
        for file in os.listdir(INFERENCE_PATH):
            if file.endswith(f'.{IMAGE_EXT}'):
                image = Image.open(f'{INFERENCE_PATH}{file}')
                
                # Create the figure
                fig = Figure(dpi=100,
                             facecolor=COLOR_PALETTE["widgets"],
                             edgecolor=COLOR_PALETTE["fg"])
                fig.add_subplot(111).imshow(image)

                # Add the resized image to the list
                self.__fig_list.append((file, fig))

        # Sorts the images alphabetically by filename
        self.__fig_list.sort(key=lambda x: x[0])

        # Putting a frame on screen to display images into
        self.__frame = LabelFrame(window,
                                  background=COLOR_PALETTE["bg"],
                                  foreground=COLOR_PALETTE["fg"],
                                  highlightbackground=COLOR_PALETTE["active"],
                                  text=self.__fig_list[0][0],
                                  font=(FONT_PATH, 14),
                                  padx=20,
                                  pady=20,
                                  width=1560,
                                  height=820)
        
        self.__frame.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
        
        # Keep the frame from resizing with the pictures
        self.__frame.pack_propagate(False)

        # Displaying the image in frame
        self.__create_mpl_canvas(self.__fig_list[0][1], self.__frame)

        # Creating the status bar
        self.__status = Label(window,
                              background=COLOR_PALETTE["bg"],
                              foreground=COLOR_PALETTE["fg"],
                              text=f'Image 1 of {len(self.__fig_list)}',
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
                                    command=lambda: self.close_window(window, textbox))

        # Putting the widgets on screen
        self.__previous_button.grid(row=1, column=0)
        self.__next_button.grid(row=1, column=2)
        self.__exit_button.grid(row=1, column=1, pady=10)
        self.__status.grid(row=2, column=0, columnspan=3, sticky="w"+"e")
    
    
    def close_window(self, window: Toplevel, textbox: ReadOnlyTextBox) -> None:
        textbox.insert("Closing the Image Viewer app.\n\n")
        window.destroy()
    
    def __create_mpl_canvas(self, fig: Figure, frame: LabelFrame) -> None:
        
        # Deleting the widgets from the screen
        if self.__canvas is not None:
            self.__canvas.get_tk_widget().destroy()
        if self.__toolbar is not None:
            self.__toolbar.destroy()
        
        # Creating the new image canvas
        self.__canvas = FigureCanvasTkAgg(fig, master=frame)
        self.__canvas.draw()
        self.__canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        # Creating the new toolbar
        self.__toolbar = NavigationToolbar2Tk(self.__canvas, frame)
        self.__toolbar.update()
    

    def __prev_img(self, window: Toplevel, idx: int) -> None:
        
        # Update the image filename
        self.__frame.config(text=self.__fig_list[idx][0])
        
        # Update the image
        self.__create_mpl_canvas(self.__fig_list[idx][1], self.__frame)

        # Update the buttons
        self.__previous_button.config(command=lambda: self.__prev_img(window, idx - 1),
                                      state="disabled" if idx == 0 else "normal")
        self.__next_button.config(command=lambda: self.__next_img(window, idx),
                                  state=NORMAL)

        # Update status bar
        self.__status.config(text=f'Image {idx + 1} of {len(self.__fig_list)}')
        

    def __next_img(self, window: Toplevel, idx: int) -> None:

        # Update the image filename
        self.__frame.config(text=self.__fig_list[idx + 1][0])

        # Update the image
        self.__create_mpl_canvas(self.__fig_list[idx + 1][1], self.__frame)
        
        # Update the buttons
        self.__previous_button.config(command=lambda: self.__prev_img(window, idx),
                                      state=NORMAL)
        self.__next_button.config(command=lambda: self.__next_img(window, idx + 1),
                                  state="disabled" if idx == len(self.__fig_list) - 2 else "normal")
        
        # Update status bar
        self.__status.config(text=f'Image {idx + 2} of {len(self.__fig_list)}')
