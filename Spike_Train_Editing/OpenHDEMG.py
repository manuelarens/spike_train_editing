import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import seaborn as sns
import pandas as pd
import numpy as np

from openhdemg.library.mathtools import compute_sil
from openhdemg.library.plotemg import showgoodlayout
 

class IptsPlotter:
    def __init__(
        self,
        emgfile,
        addrefsig=False,
        timeinseconds=True,
        figsize=[20, 15],
        showimmediately=True,
        tight_layout=False,
    ):
        self.emgfile = emgfile
        self.addrefsig = addrefsig
        self.timeinseconds = timeinseconds
        self.figsize = figsize
        self.tight_layout = tight_layout
        self.fsamp = emgfile["FSAMP"]

        # Check for IPTS and MUPULSES
        self.ipts = self._validate_data(emgfile["IPTS"], pd.DataFrame, "IPTS")
        
        # Generate x-axis (seconds or samples)
        self.x_axis = (
            self.ipts.index / self.fsamp
            if timeinseconds
            else self.ipts.index
        )

        # Initialize the current MU index
        self.current_index = 0

        # Create plot
        self.fig, self.ax1 = plt.subplots(
            figsize=(figsize[0] / 2.54, figsize[1] / 2.54), num="IPTS"
        )

        # Plot the initial MU
        self.plot_current_mu()

        # Add zoom, scroll, and key press events
        self.fig.canvas.mpl_connect("scroll_event", self.zoom)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Add previous and next buttons
        self.add_buttons()

        # Add RectangleSelector
        self.rect_selector = RectangleSelector(
            self.ax1, self.onselect, useblit=True, button=[1]
        )

        if addrefsig:
            self.plot_reference_signal()


        # Show layout
        showgoodlayout(self.tight_layout, despined="2yaxes" if addrefsig else False)

        

        if showimmediately:
            plt.show()

    def _validate_data(self, data, expected_type, name):
        if isinstance(data, expected_type):
            return data
        raise TypeError(f"{name} is probably absent or not in the expected format")

    def _process_mu_pulses(self, mupulses):
        # Check if it is correct data type and convert to seconds
        if isinstance(mupulses, list):
            return [[pulse / self.fsamp for pulse in pulses] for pulses in mupulses]
        raise TypeError("MUPULSES is probably absent or not in a list")


    def plot_current_mu(self):
        self.ax1.clear()  # Clear previous plot
        self.ax1.plot(self.x_axis, self.ipts[self.current_index])
        self.ax1.set_ylabel(f"MU {self.current_index + 1}")
        self.plot_peaks()
        sil_value = compute_sil(self.ipts[self.current_index], self.emgfile["MUPULSES"][self.current_index])

        # Display SIL value
        self.ax1.text(
            0.5,
            1.05,
            f"SIL = {sil_value:.6f}",
            ha="center",
            va="center",
            transform=self.ax1.transAxes,
            fontsize=10,
            fontweight="bold",
        )
        self.ax1.set_xlabel("Time (Sec)" if self.timeinseconds else "Samples")
        # Add instructions
        self.add_instructions()

    def plot_peaks(self):
        mu_pulses = self._process_mu_pulses(self.emgfile["MUPULSES"])
        pulses = mu_pulses[self.current_index]
        for pulse in pulses:
            closest_idx = (np.abs(self.x_axis - pulse)).argmin()
            y_value = self.ipts.iloc[closest_idx, self.current_index]
            self.ax1.plot(pulse, y_value, "ro", markersize=2, label="Peak")

    def zoom(self, event):
        current_xlim = self.ax1.get_xlim()
        zoom_factor = 0.85 if event.button == "up" else 1.15
        midpoint = (current_xlim[0] + current_xlim[1]) / 2
        delta = (current_xlim[1] - current_xlim[0]) * zoom_factor / 2
        new_xlim = (midpoint - delta, midpoint + delta)

        if min(new_xlim) > 0 and max(new_xlim) < max(self.x_axis):
            self.ax1.set_xlim(new_xlim)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "left" or event.key == "a":
            self.scroll_left()
        elif event.key == "right" or event.key == "d":
            self.scroll_right()

    def scroll_left(self):
        current_xlim = self.ax1.get_xlim()
        delta = (current_xlim[1] - current_xlim[0]) * 0.1
        if current_xlim[0] - delta > 0:
            new_xlim = [current_xlim[0] - delta, current_xlim[1] - delta]
            self.ax1.set_xlim(new_xlim)
            self.fig.canvas.draw_idle()

    def scroll_right(self):
        current_xlim = self.ax1.get_xlim()
        delta = (current_xlim[1] - current_xlim[0]) * 0.1
        if current_xlim[1] + delta < max(self.x_axis):
            new_xlim = [current_xlim[0] + delta, current_xlim[1] + delta]
            self.ax1.set_xlim(new_xlim)
            self.fig.canvas.draw_idle()

    def add_buttons(self):
        # Storing the button objects as class attributes to prevent garbage collection
        color = "black"
        ax_prev = plt.axes([0.14, 0.85, 0.12, 0.04], facecolor=color)
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_prev.on_clicked(self.previous_mu)

        ax_next = plt.axes([0.87, 0.85, 0.12, 0.04], facecolor=color)
        self.btn_next = Button(ax_next, "Next")
        self.btn_next.on_clicked(self.next_mu)

        self.fig.canvas.draw_idle()  # Force canvas update after button creation


    def previous_mu(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.plot_current_mu()
            self.fig.canvas.draw_idle()
            

    def next_mu(self, event):
        if self.current_index < self.emgfile["NUMBER_OF_MUS"] - 1:
            self.current_index += 1
            self.plot_current_mu()
            self.fig.canvas.draw_idle()

    def onselect(self, eclick, erelease):
        x1, x2 = eclick.xdata, erelease.xdata
        y1, y2 = eclick.ydata, erelease.ydata
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        mask = (
            (self.x_axis > min(x1, x2))
            & (self.x_axis < max(x1, x2))
            & (self.ipts[self.current_index] > min(y1, y2))
            & (self.ipts[self.current_index] < max(y1, y2))
        )
        xmasked = self.x_axis[mask]
        ymasked = self.ipts[self.current_index][mask]

        if len(xmasked) > 0:
            #plot extra point
            xmax = xmasked[np.argmax(ymasked)]
            ymax = ymasked.max()
            self.ax1.plot(xmax, ymax, "ro", markersize=2, label="Peak")
            

            #add extra point to the emgfile
            x_idx = xmax*self.fsamp
            index = np.searchsorted(self.emgfile["MUPULSES"][self.current_index], x_idx)
            self.emgfile["MUPULSES"][self.current_index] = np.insert(self.emgfile["MUPULSES"][self.current_index], index, x_idx)

        self.fig.canvas.draw_idle()

    def add_instructions(self):
        instructions = (
            "Mouse Wheel: Zoom in/out\n"
            "Left/Right Arrow Keys: Scroll left/right\n"
            "Click and Drag: Select region for new peaks"
        )
        self.ax1.text(
            -0.15, 1.14, instructions, transform=self.ax1.transAxes, fontsize=10, verticalalignment="top"
        )

    def plot_reference_signal(self):
        ax2 = self.ax1.twinx()
        xref = (
            self.emgfile["REF_SIGNAL"].index / self.fsamp
            if self.timeinseconds
            else self.emgfile["REF_SIGNAL"].index
        )
        sns.lineplot(x=xref, y=self.emgfile["REF_SIGNAL"][0], color="0.4", ax=ax2)
        ax2.set_ylabel("MVC")

    def insert_into_sorted_arrays(mupulses, new_value):
        """
        Insert a new integer into each sorted numpy array within the list mupulses.
        
        Parameters
        ----------
        mupulses : list of numpy.ndarray
            List containing sorted numpy arrays where the new value will be inserted.
        new_value : int
            The new integer value to be inserted into the arrays.
        
        Returns
        -------
        list of numpy.ndarray
            List of numpy arrays with the new value inserted in each array.
        """
        updated_mupulses = []
        for arr in mupulses:
            # Find the index where the new value should be inserted
            index = np.searchsorted(arr, new_value)
            # Insert the new value
            new_arr = np.insert(arr, index, new_value)
            updated_mupulses.append(new_arr)
        
        return updated_mupulses