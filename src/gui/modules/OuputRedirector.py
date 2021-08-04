from tkinter import *
import sys
from itertools import islice
from subprocess import Popen, PIPE
from textwrap import dedent
from threading import Thread
from queue import Queue, Empty


def iter_except(function, exception):
    """Works like builtin 2-argument `iter()`, but stops on `exception`."""
    try:
        while True:
            yield function()
    except exception:
        return


class OutputRedirector:
    
    def __init__(self, window, target, cmd):
        self.window = window
        self.target = target

        # start dummy subprocess to generate some output
        self.process = Popen(cmd, stdout=PIPE)

        # launch thread to read the subprocess output
        #   (put the subprocess output into the queue in a background thread,
        #    get output from the queue in the GUI thread.
        #    Output chain: process.readline -> queue -> label)
        # limit output buffering (may stall subprocess)
        q = Queue(maxsize=1024)
        t = Thread(target=self.reader_thread, args=[q])
        t.daemon = True  # close pipe if GUI process exits
        t.start()

        # show subprocess' stdout in GUI
        self.update(q)  # start update loop
        t.join()

    def reader_thread(self, q):
        """Read subprocess output and put it into the queue."""
        try:
            with self.process.stdout as pipe:
                for line in iter(pipe.readline, b''):
                    q.put(line)
        finally:
            q.put(None)

    def update(self, q):
        """Update GUI with items from the queue."""
        for line in iter_except(q.get_nowait, Empty):  # display all content
            if line is None:
                return
            else:
                self.target.insert(INSERT, line)  # update GUI
                self.window.update()
                break
        self.window.after(5, self.update, q)  # schedule next update

    def quit(self):
        self.process.kill()  # exit subprocess if GUI is closed (zombie!)
        self.window.destroy()
