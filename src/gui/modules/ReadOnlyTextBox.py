"""
File:
    src/gui/modules/ReadOnlyTextBox.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Custom read-only tkinter text box
"""

from tkinter import *

from src.utils.constants import COLOR_PALETTE


class ReadOnlyTextBox:
    """
    Custom read-only tkinter text box
    """
    def __init__(self,
                 window: Tk) -> None:
        """Class constructor

        Args:
            window (Tk): Root window
        """
        
        # Create the scrollbar
        self.__scroll = Scrollbar(window,
                                  activebackground=COLOR_PALETTE["purple"],
                                  background=COLOR_PALETTE["widgets"],
                                  highlightbackground=COLOR_PALETTE["bg"],
                                  troughcolor=COLOR_PALETTE["active"],
                                  orient=VERTICAL)
        self.__scroll.pack(side=RIGHT, fill=Y)

        # Create the textbox
        self.__textbox = Text(window,
                              background=COLOR_PALETTE["widgets"],
                              foreground=COLOR_PALETTE["green"],
                              highlightbackground=COLOR_PALETTE["bg"],
                              insertbackground=COLOR_PALETTE["fg"],
                              selectbackground=COLOR_PALETTE["purple"],
                              state=DISABLED,
                              width=115,
                              height=12,
                              wrap=WORD,
                              yscrollcommand=self.__scroll.set)
        self.__textbox.pack(side=LEFT, fill=BOTH)

        self.__scroll.config(command=self.__textbox.yview)

    def insert(self,
               text: str) -> None:
        """Inserts text in the text box

        Args:
            text (str): Text to be inserted
        """
        
        # Modify the state to allow insertion
        self.__textbox.config(state=NORMAL)
        
        # Insert the text and move the scrollbar to the bottom
        self.__textbox.insert(INSERT, text)
        self.__textbox.yview_moveto("1.0")
        
        # Modify the state again to prevent user insertion
        self.__textbox.config(state=DISABLED)
