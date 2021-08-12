from tkinter import *
from subprocess import Popen, PIPE
from threading import Thread
from queue import Queue, Empty
from typing import List
from src.gui.modules.ReadOnlyTextBox import ReadOnlyTextBox
from src.utils.constants import COLOR_PALETTE
from src.gui.ImageViewer import ImageViewer


def iter_except(function, exception):
    """Works like builtin 2-argument `iter()`, but stops on `exception`."""
    try:
        while True:
            yield function()
    except exception:
        return


class OutputRedirector:
    
    def __init__(self, window: Tk, target: ReadOnlyTextBox, cmd: List[str]) -> None:
        # Declare the parent window of the OutputRedirector
        self.__window = window
        
        # Declare the target text widget that gets the updates
        self.__target = target
        
        # Displaying the process to the user
        self.__target.insert(
            f'{"-" * 100}\nStarting Inference and resizing images\n{"-" * 100}\nThis process can take a few minutes depending on the size of the directory.\n')

        # Starts the inference process
        self.__process = Popen(cmd, stdout=PIPE)

        # Launch thread to read the subprocess output
        q = Queue()
        t = Thread(target=self.reader_thread, args=[q])
        t.daemon = True
        t.start()

        # Start the update loop
        self.update(q)
    
    def reader_thread(self, q: Queue) -> None:
        """Read subprocess output and put it into the queue."""
        try:
            with self.__process.stdout as pipe:
                for line in iter(pipe.readline, b''):
                    q.put(line)
        finally:
            q.put(None)

    def update(self, q: Queue) -> None:
        """Update GUI with items from the queue."""

        for line in iter_except(q.get_nowait, Empty):
            if line is None:
                
                self.__target.insert(f'\n{"-" * 100}\nInference completed, now opening the Image Viewer app...\n')
                
                # When the process is done open the image viewer app
                image_viewer_window = Toplevel()
                image_viewer_window.geometry("1600x926")
                image_viewer_window.config(background=COLOR_PALETTE["bg"])
                image_viewer_window.resizable(False, False)
                ImageViewer(window=image_viewer_window, textbox=self.__target)

                self.__process.kill()
            else:
                # Update the target widget
                self.__target.insert(line)
                break
            
        # Schedule the next update
        self.__window.after(25, self.update, q)
