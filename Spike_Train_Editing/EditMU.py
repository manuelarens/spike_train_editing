"""
EditMU Class

This class provides a graphical user interface for editing and analyzing motor unit spike trains
in HD-EMG data. It allows for interactive modification, visualization, and recalculation of EMG
data using various tools and filters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import json
import os
import gzip
from sklearn.cluster import KMeans

from openhdemg.library.mathtools import compute_sil
from openhdemg.library.plotemg import showgoodlayout
from RecalcFilter import RecalcFilter
from processing_tools import get_binary_pulse_trains, whiteesig, extend_emg, pcaesig, detect_peaks, maxk

class EditMU:
    def __init__(
        self,
        emgfile,
        filepath,
        addrefsig=False,
        timeinseconds=True,
        figsize=(20, 15),
        showimmediately=True,
        tight_layout=False,
    ):
        """
        Initialize the EditMU class and set up the plotting environment.

        This constructor performs the following tasks:
        1. Sets up instance variables using the provided parameters, including the EMG data file,
        whether to add a reference signal, time unit preference, figure size, and layout settings.
        2. Validates and processes the EMG data from the provided file, specifically the IPTS and MUPULSES data.
        3. Initializes the x-axis based on whether the time is measured in seconds or samples.
        4. Sets the initial motor unit index to 0.
        5. Prepares lists and arrays for tracking plot states, including flags for the first plot,
        recalculated SIL values, and arrays for storing SIL values and colors.
        6. Creates a matplotlib figure and axis for plotting, with specified figure size.
        7. Sets up RectangleSelector objects for adding and removing spikes on the plot.
        8. Initializes boolean flags for managing spike addition and removal.
        9. Plots the initial motor unit data.
        10. Connects event handlers for zooming, scrolling, and key press events.
        11. Adds navigation buttons for moving between motor units.
        12. Optionally plots a reference signal if specified.
        13. Configures the plot layout and displays the plot if required.

        Parameters:
        - emgfile (dict): A dictionary containing EMG data including 'IPTS', 'MUPULSES', and 'FSAMP'.
        - addrefsig (bool): Flag to indicate whether to add a reference signal to the plot.
        - timeinseconds (bool): Flag to specify whether the x-axis should be in seconds or samples.
        - figsize (list): A list specifying the figure size in inches.
        - showimmediately (bool): Flag to indicate whether to display the plot immediately.
        - tight_layout (bool): Flag to indicate whether to use tight layout for the plot.

        The constructor ensures that the plotting environment is properly initialized and ready
        for displaying and interacting with motor unit data.
        """
        # Initialize attributes
        self.filepath = filepath
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

        self.edited_dict = {}

        self.grid_name = ['4-8-L']

        # Initialize additional attributes
        self.peak_artists = []
        self.sil_recalculated = [False] * len(emgfile['MUPULSES'])
        self.sil_old = np.zeros(len(emgfile['MUPULSES']))
        for i in range(len(self.emgfile["IPTS"].columns)):
            self.sil_old[i] = compute_sil(
            self.ipts[i],
            self.emgfile["MUPULSES"][i]
        )

        self.sil_new = np.zeros(len(emgfile['MUPULSES']))
        self.sil_color = 'black'

        # Create plot
        self.fig, self.ax1 = plt.subplots(
            figsize=(figsize[0] / 2.54, figsize[1] / 2.54),
            num="IPTS"
        )

        # Initialize the RectangleSelectors
        self.rect_selector_add = RectangleSelector(
            self.ax1, self.onselect_add, useblit=True, button=[1]
        )
        self.rect_selector_add.set_active(False)

        self.rect_selector_remove = RectangleSelector(
            self.ax1, self.onselect_remove, useblit=True, button=[1]
        )
        self.rect_selector_remove.set_active(False)

        # Initialize boolean flags
        self.remove_spikes_boolean = False
        self.add_spikes_boolean = False

        # Plot the initial MU
        self.plot_current_mu()

        # Add zoom, scroll, and key press events
        self.fig.canvas.mpl_connect("scroll_event", self.zoom)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Add previous and next buttons
        self.add_buttons()

        # Plot reference signal if needed
        if addrefsig:
            self.plot_reference_signal()

        # Show layout
        showgoodlayout(self.tight_layout, despined="2yaxes" if addrefsig else False)

        # Show the plot immediately if needed
        if showimmediately:
            plt.show()


    def _validate_data(self, data, expected_type, name):
        if isinstance(data, expected_type):
            return data
        raise TypeError(f"{name} is probably absent or not in the expected format")

    def _process_mu_pulses(self, mupulses):
        # Check if it is correct data type and convert to seconds for easy plotting against time axis
        if isinstance(mupulses, list):
            return [[pulse / self.fsamp for pulse in pulses] for pulses in mupulses]
        raise TypeError("MUPULSES is probably absent or not in a list")


    def plot_current_mu(self):
        """
        Plot the current motor unit (MU) and update its associated metrics.

        This method performs the following tasks:
        1. Clears the current plot to prepare for a new plot.
        2. Plots the data for the current MU using the x-axis and IPTS values.
        3. Sets the y-axis label to indicate the current MU number.
        4. Calls `plot_peaks` to plot peaks for the current MU.
        5. Computes the Signal-to-Interference Level (SIL) for the current MU.
        6. Determines the color of the SIL text based on the SIL difference from the previous plot.
        7. Updates the old SIL value with the new SIL value.
        8. Displays the SIL value on the plot with the appropriate color and formatting.
        9. Sets the x-axis label based on whether time is in seconds or samples.
        10. Adds any additional instructions or annotations to the plot.

        The method ensures that the plot is updated with the latest information about the
        motor unit, including visual markers for peaks and metrics for the SIL. It also
        updates the plot's appearance and labels according to the current state of the data.
        """

        # Clear previous plot
        self.ax1.clear()
        
        # Plot the current MU
        self.ax1.plot(self.x_axis, self.ipts[self.current_index])
        self.ax1.set_ylabel(f"MU {self.current_index + 1}")
        
        # Plot peaks
        self.plot_peaks()
        
        # Compute new SIL
        self.sil_new[self.current_index] = compute_sil(
            self.ipts[self.current_index],
            self.emgfile["MUPULSES"][self.current_index]
        )

        if not self.sil_recalculated[self.current_index]:
            # First plot or not recalculated: use default color and no difference text
            self.sil_color = 'black'
            sil_dif_text = ""
        else:
            # Calculate SIL difference
            sil_dif = self.sil_new[self.current_index] - self.sil_old[self.current_index]
            sil_dif_text = f' (Δ = {sil_dif:.5f})'

            # Determine SIL color based on difference
            if sil_dif > 0:
                self.sil_color = 'limegreen' if sil_dif < 0.02 else 'darkgreen'
            elif sil_dif < 0:
                self.sil_color = 'salmon' if sil_dif > -0.02 else 'red'
            else:
                self.sil_color = 'black'

        # Update old SIL value
        self.sil_old[self.current_index] = self.sil_new[self.current_index]

        # Display SIL value
        self.ax1.text(
            0.39,
            1.05,
            f"SIL = {self.sil_new[self.current_index]:.6f}{sil_dif_text}",
            ha="left",
            va="center",
            transform=self.ax1.transAxes,
            fontsize=10,
            fontweight="bold",
            color=self.sil_color
        )

        # Set x-axis label
        self.ax1.set_xlabel("Time (Sec)" if self.timeinseconds else "Samples")

        # Add instructions
        self.add_instructions()


    def plot_peaks(self):
        """
        Plot the peaks of the current motor unit (MU) on the graph.

        This method performs the following steps:
        1. Processes the motor unit pulses from the EMG file to get the pulses for the current MU.
        2. Clears any existing peak markers (artists) from the plot.
        3. Resets the list of peak artists.
        4. Plots each pulse as a red circle on the graph at the corresponding x-axis location.
        5. Updates the list of peak artists with the newly plotted peaks for interactive picking.

        The pulses are plotted as red circles with a small marker size, and picking is enabled for each peak
        to allow for interaction with the plotted peaks.
        """

        # Process motor unit pulses
        pulses = self._process_mu_pulses(self.emgfile["MUPULSES"])
        pulses = pulses[self.current_index]

        # Clear existing peak artists from the plot
        if hasattr(self, 'peak_artists') and self.peak_artists:
            for peak_artist in self.peak_artists:
                peak_artist.remove()  # Remove each peak from the plot

        # Reset the peak artists list
        self.peak_artists = []

        # Plot each pulse as a red circle
        for pulse in pulses:
            # Find the closest point on the x-axis to the pulse time
            closest_idx = (np.abs(self.x_axis - pulse)).argmin()
            # Get the y-value corresponding to the closest point
            y_value = self.ipts.iloc[closest_idx, self.current_index]
            # Plot the pulse as a red circle and enable picking
            peak_artist, = self.ax1.plot(
                pulse, y_value, "ro", markersize=2, picker=True
            )
            self.peak_artists.append(peak_artist)



    def zoom(self, event):
        """
        Handle zooming in and out of the plot based on mouse scroll events.

        This method performs the following actions:
        1. Retrieves the current x-axis limits of the plot.
        2. Determines the zoom factor based on the mouse scroll direction:
        - Zoom in if the scroll direction is "up".
        - Zoom out if the scroll direction is "down".
        3. Calculates the new x-axis limits based on the zoom factor and the midpoint of the current x-axis limits.
        4. Ensures the new x-axis limits do not go below 0 or exceed the maximum x-axis value.
        5. Updates the x-axis limits of the plot and redraws the canvas to reflect the zoom change.

        Parameters:
        - event (matplotlib.backend_bases.ScrollEvent): The mouse scroll event containing information about the scroll direction.
        """
        current_xlim = self.ax1.get_xlim()  # Get current x-axis limits
        zoom_factor = 0.85 if event.button == "up" else 1.15  # Determine zoom factor based on scroll direction

        # Calculate the midpoint and delta for the new x-axis limits
        midpoint = (current_xlim[0] + current_xlim[1]) / 2
        delta = (current_xlim[1] - current_xlim[0]) * zoom_factor / 2
        new_xlim = (midpoint - delta, midpoint + delta)

        # Ensure new x-axis limits are within valid range
        if min(new_xlim) > 0 and max(new_xlim) < max(self.x_axis):
            self.ax1.set_xlim(new_xlim)  # Update x-axis limits
            self.fig.canvas.draw_idle()  # Redraw the canvas to apply changes


    def on_key(self, event):
        """
        Handle key press events to navigate through motor units.

        This method performs the following actions based on the pressed key:
        1. Scrolls left through the motor units if the 'left' arrow key or 'a' key is pressed.
        2. Scrolls right through the motor units if the 'right' arrow key or 'd' key is pressed.

        Parameters:
        - event (matplotlib.backend_bases.KeyEvent): The key event containing information about the pressed key.
        """
        if event.key in ("left", "a"):
            self.scroll_left()  # Scroll left through motor units
        elif event.key in ("right", "d"):
            self.scroll_right()  # Scroll right through motor units


    def scroll_left(self):
        """
        Scroll the plot view to the left by a fixed percentage of the current x-axis range.

        This method performs the following actions:
        1. Retrieves the current x-axis limits of the plot.
        2. Calculates the delta for scrolling, which is 10% of the current x-axis range.
        3. Updates the x-axis limits to scroll left, ensuring the new limits do not go below 0.
        4. Redraws the canvas to reflect the changes.

        This method ensures that the plot view is shifted left while staying within valid bounds.

        """
        current_xlim = self.ax1.get_xlim()  # Get current x-axis limits
        delta = (current_xlim[1] - current_xlim[0]) * 0.1  # Calculate the scrolling delta

        # Check if new x-axis limits will be within valid range
        if current_xlim[0] - delta > 0:
            new_xlim = [current_xlim[0] - delta, current_xlim[1] - delta]  # Calculate new x-axis limits
            self.ax1.set_xlim(new_xlim)  # Update x-axis limits
            self.fig.canvas.draw_idle()  # Redraw the canvas to apply changes

    def scroll_right(self):
        """
        Scroll the plot view to the right by a fixed percentage of the current x-axis range.

        This method performs the following actions:
        1. Retrieves the current x-axis limits of the plot.
        2. Calculates the delta for scrolling, which is 10% of the current x-axis range.
        3. Updates the x-axis limits to scroll right, ensuring the new limits do not exceed the maximum x-axis value.
        4. Redraws the canvas to reflect the changes.

        This method ensures that the plot view is shifted right while staying within valid bounds.

        """
        current_xlim = self.ax1.get_xlim()  # Get current x-axis limits
        delta = (current_xlim[1] - current_xlim[0]) * 0.1  # Calculate the scrolling delta

        # Check if new x-axis limits will be within valid range
        if current_xlim[1] + delta < max(self.x_axis):
            new_xlim = [current_xlim[0] + delta, current_xlim[1] + delta]  # Calculate new x-axis limits
            self.ax1.set_xlim(new_xlim)  # Update x-axis limits
            self.fig.canvas.draw_idle()  # Redraw the canvas to apply changes


    def add_buttons(self):
        """
        Add interactive buttons to the plot for various functionalities.

        This method creates and positions buttons on the plot to perform the following actions:
        1. Navigate to the previous motor unit ("Previous" button).
        2. Navigate to the next motor unit ("Next" button).
        3. Add spikes to the plot ("Add spikes" button).
        4. Remove spikes from the plot ("Remove spikes" button).
        5. Recalculate the filter ("Recalc. filter" button).
        6. Delete the current motor unit ("Delete MU" button).

        The buttons are created with specific colors and are linked to their respective callback methods.
        The canvas is updated to reflect the addition of the buttons.
        """
        # Define button colors
        self.button_color = "ivory"
        self.button_active_color = "mistyrose"

        # Create and position "Previous" button
        ax_prev = plt.axes([0.01, 0.025, 0.12, 0.04])
        self.btn_prev = Button(ax_prev, "Previous", color=self.button_color)
        self.btn_prev.on_clicked(self.previous_mu)  # Link button to method

        # Create and position "Next" button
        ax_next = plt.axes([0.87, 0.025, 0.12, 0.04])
        self.btn_next = Button(ax_next, "Next", color=self.button_color)
        self.btn_next.on_clicked(self.next_mu)  # Link button to method

        # Create and position "Add spikes" button
        ax_add = plt.axes([0.85, 0.82, 0.14, 0.04])
        self.btn_add = Button(ax_add, "Add spikes", color=self.button_color)
        self.btn_add.on_clicked(self.add_spikes)  # Link button to method

        # Create and position "Remove spikes" button
        ax_remove = plt.axes([0.85, 0.76, 0.14, 0.04])
        self.btn_remove = Button(ax_remove, "Remove spikes", color=self.button_color)
        self.btn_remove.on_clicked(self.remove_spikes)  # Link button to method

        # Create and position "Recalc. filter" button
        ax_recalc = plt.axes([0.85, 0.70, 0.14, 0.04])
        self.btn_recalc = Button(ax_recalc, "Recalc. filter", color=self.button_color)
        self.btn_recalc.on_clicked(self.recalc_filter)  # Link button to method

        # Create and position "Delete MU" button with distinct color
        ax_delete = plt.axes([0.85, 0.95, 0.14, 0.04])
        self.btn_delete = Button(ax_delete, "Delete MU", color='tomato', hovercolor='salmon')
        self.btn_delete.on_clicked(self.delete_MU)  # Link button to method

        # Update the canvas to reflect the added buttons
        self.fig.canvas.draw_idle()


    def delete_MU(self, event):
        """
        Handle the deletion of a motor unit (MU) after user confirmation.

        This method performs the following actions:
        1. Displays a confirmation dialog to the user asking if they want to delete the current MU.
        2. If the user confirms:
        - Disconnects all buttons to prevent further interaction during the deletion process.
        - Removes the column corresponding to the current MU from the 'IPTS' DataFrame.
        - Deletes the motor unit pulses at the current index from the 'MUPULSES' list.
        - Updates relevant internal state variables:
            - Removes the entry at the current index from 'first_plot' and 'sil_recalculated' lists.
            - Deletes the entry at the current index from 'sil_old' and 'sil_new' numpy arrays.
        - Adjusts the current index to point to the last valid MU.
        - If no MU remains, closes the plot.
        - If there are remaining MUs, plots the current MU and redraws the canvas.
        3. Destroys the Tkinter root window used for the confirmation dialog.

        Args:
            event: The event object associated with the button click that triggered this method.
        """
        # Create a hidden Tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Show a confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            "Are you sure you want to delete this motor unit?",
            icon='warning',  # Adds a warning icon to the dialog
            parent=root
        )

        # Proceed with deletion if the user clicks "Yes"
        if confirm:
            self.disconnect_buttons()  # Disable button interactions during deletion

            # Drop the column corresponding to the current index
            self.emgfile['IPTS'].drop(columns=[self.current_index], inplace=True)
            self.emgfile['IPTS'].columns = range(self.emgfile['IPTS'].shape[1])

            # Remove the motor unit pulses at the current index
            del self.emgfile['MUPULSES'][self.current_index]

            # Update internal state variables
            self.sil_recalculated.pop(self.current_index)
            self.sil_old = np.delete(self.sil_old, self.current_index)
            self.sil_new = np.delete(self.sil_new, self.current_index)

            # Update the current index to the last valid motor unit
            if self.current_index >= len(self.emgfile['MUPULSES']):
                self.current_index = len(self.emgfile['MUPULSES']) - 1
                if self.current_index == -1:
                    print("No MUs remaining, closing script")
                    plt.close(self.fig)

            # Plot the current MU if any remain
            if self.current_index >= 0:
                self.plot_current_mu()
                self.fig.canvas.draw_idle()

        # Destroy the root window
        root.destroy()

        
    def add_spikes(self, event):
        """
        Toggle the addition of spikes on the plot.

        This method activates or deactivates the RectangleSelector for adding spikes based on the current state. 
        If the add spikes mode is turned off, it will activate the RectangleSelector for adding spikes and 
        deactivate the RectangleSelector for removing spikes (if active). It also updates the button colors 
        to reflect the current state.

        Args:
            event: The event object associated with the button click that triggered this method.
        """
        if not self.add_spikes_boolean:
            # Deactivate the RectangleSelector for removing spikes if it is active
            if self.remove_spikes_boolean:
                self.rect_selector_remove.set_active(False)
                self.remove_spikes_boolean = False
                self.btn_remove.color = self.button_color

            # Activate the RectangleSelector for adding spikes
            self.rect_selector_add.set_active(True)
            self.add_spikes_boolean = True
            self.btn_add.color = self.button_active_color
            self.fig.canvas.draw_idle()  # Refresh the canvas to reflect changes
        else:
            # If add spikes mode is already active, disconnect buttons to exit mode
            self.disconnect_buttons()

    def remove_spikes(self, event):
        """
        Toggle the removal of spikes from the plot.

        This method activates or deactivates the RectangleSelector for removing spikes based on the current state. 
        If the remove spikes mode is turned off, it will activate the RectangleSelector for removing spikes and 
        deactivate the RectangleSelector for adding spikes (if active). It also updates the button colors 
        to reflect the current state.

        Args:
            event: The event object associated with the button click that triggered this method.
        """
        if not self.remove_spikes_boolean:
            # Deactivate the RectangleSelector for adding spikes if it is active
            if self.add_spikes_boolean:
                self.rect_selector_add.set_active(False)
                self.add_spikes_boolean = False
                self.btn_add.color = self.button_color

            # Activate the RectangleSelector for removing spikes
            self.rect_selector_remove.set_active(True)
            self.remove_spikes_boolean = True
            self.btn_remove.color = self.button_active_color
            self.fig.canvas.draw_idle()  # Refresh the canvas to reflect changes
        else:
            # If remove spikes mode is already active, disconnect buttons to exit mode
            self.disconnect_buttons()

    
    def disconnect_buttons(self):
        """
        Disconnect and reset the spike addition and removal functionalities.

        This method deactivates the RectangleSelectors used for adding and removing spikes, and resets 
        their corresponding boolean states and button colors to their default values.
        """
        # Deactivate the RectangleSelector for adding spikes
        self.rect_selector_add.set_active(False)
        self.add_spikes_boolean = False
        self.btn_add.color = self.button_color

        # Deactivate the RectangleSelector for removing spikes
        self.rect_selector_remove.set_active(False)
        self.remove_spikes_boolean = False
        self.btn_remove.color = self.button_color

    def previous_mu(self, event):
        """
        Switch to the previous motor unit and update the plot.

        This method decreases the index of the current motor unit, updates the plot to display the previous 
        motor unit's data, and redraws the canvas. It also disconnects any active spike addition or removal 
        functionalities.

        Args:
            event: The event object associated with the button click that triggered this method.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.plot_current_mu()
            self.fig.canvas.draw_idle()  # Refresh the canvas to reflect changes
            self.disconnect_buttons()  # Reset spike addition/removal functionalities

    def next_mu(self, event):
        """
        Switch to the next motor unit and update the plot.

        This method increases the index of the current motor unit, updates the plot to display the next 
        motor unit's data, and redraws the canvas. It also disconnects any active spike addition or removal 
        functionalities.

        Args:
            event: The event object associated with the button click that triggered this method.
        """
        if self.current_index < len(self.emgfile["IPTS"].columns) - 1:
            self.current_index += 1
            self.plot_current_mu()
            self.fig.canvas.draw_idle()  # Refresh the canvas to reflect changes
            self.disconnect_buttons()  # Reset spike addition/removal functionalities

            
    def onselect_add(self, eclick, erelease):
        """
        Handle the rectangular selection for adding spikes.

        This method processes a rectangular region selected by the user to identify and plot
        the peak within the selected area. It also updates the motor unit pulses to include
        the new spike. The rectangular selection is defined by the click and release events.

        Args:
            eclick: The mouse click event at the start of the selection.
            erelease: The mouse release event at the end of the selection.
        """
        # Extract coordinates from the click and release events
        x1, x2 = eclick.xdata, erelease.xdata
        y1, y2 = eclick.ydata, erelease.ydata

        # Ensure coordinates are ordered from min to max
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Create a mask to filter data within the selected rectangular region
        mask = (
            (self.x_axis > min(x1, x2))
            & (self.x_axis < max(x1, x2))
            & (self.ipts[self.current_index] > min(y1, y2))
            & (self.ipts[self.current_index] < max(y1, y2))
        )
        xmasked = self.x_axis[mask]
        ymasked = self.ipts[self.current_index][mask]

        if len(xmasked) > 0:
            # Identify the peak with the maximum y-value within the selected region
            xmax = xmasked[np.argmax(ymasked)]
            ymax = ymasked.max()

            # Plot the peak and add it to the list of peak artists
            peak_artist, = self.ax1.plot(xmax, ymax, "ro", markersize=2, label="Peak")
            self.peak_artists.append(peak_artist)  # Store the artist object

            # Add the new peak to the motor unit pulses
            x_idx = xmax * self.fsamp
            index = np.searchsorted(self.emgfile["MUPULSES"][self.current_index], x_idx)
            self.emgfile["MUPULSES"][self.current_index] = np.insert(
                self.emgfile["MUPULSES"][self.current_index], index, x_idx
            )

        # Refresh the canvas to reflect changes
        self.fig.canvas.draw_idle()


    def onselect_remove(self, eclick, erelease):
        """
        Handle the rectangular selection for removing spikes.

        This method processes a rectangular region selected by the user to identify and remove
        spikes within the selected area. It updates the motor unit pulses to reflect the removal
        of these spikes. The rectangular selection is defined by the click and release events.

        Args:
            eclick: The mouse click event at the start of the selection.
            erelease: The mouse release event at the end of the selection.
        """
        # Extract coordinates from the click and release events
        x1, x2 = eclick.xdata, erelease.xdata
        y1, y2 = eclick.ydata, erelease.ydata

        # Ensure coordinates are ordered from min to max
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Create a mask to filter data within the selected rectangular region
        mask = (
            (self.x_axis > min(x1, x2))
            & (self.x_axis < max(x1, x2))
            & (self.emgfile["IPTS"][self.current_index] > min(y1, y2))
            & (self.emgfile["IPTS"][self.current_index] < max(y1, y2))
        )
        xmasked = self.x_axis[mask]
        ymasked = self.emgfile["IPTS"][self.current_index][mask]

        # Initialize a deletion counter and set a maximum number of deletions
        delete_count = 0
        max_deletions = 10

        if len(xmasked) > 0:
            # Iterate over the selected peaks and remove them
            for i, (xmax, ymax) in enumerate(zip(xmasked, ymasked)):
                # Find the closest pulse (in samples) to remove
                x_idx = int(xmax * self.fsamp)  # Convert xmax to sample index
                pulses = self.emgfile["MUPULSES"][self.current_index]
                index = np.searchsorted(pulses, x_idx)
                if index < len(pulses) and pulses[index] == x_idx:
                    self.emgfile["MUPULSES"][self.current_index] = np.delete(pulses, index)

                    # Increment the deletion counter
                    delete_count += 1
                    if delete_count >= max_deletions:
                        # Stop if the maximum number of deletions is reached
                        print('Limit of 10 deletions at a time reached')

                        # Update the figure with the remaining peaks
                        self.plot_peaks()
                        self.fig.canvas.draw_idle()
                        return

            # Update the figure with the remaining peaks
            self.plot_peaks()
            self.fig.canvas.draw_idle()
    
    def recalc_filter(self, event):
        """
        Recalculate the pulse train and update the motor unit plot.

        This method is triggered by a button event to recalculate the pulse train based on the
        current motor unit's pulses. It also updates the plot to reflect the recalculated
        pulse train and recalculated peaks. The method handles the following steps:
        1. Disconnects buttons to prevent further interactions during recalculation.
        2. Checks if there are pulses available for recalculation.
        3. Recalculates the pulse train and peaks.
        4. Updates the plot with the new pulse train and peak information.
        5. Resets the button color after the recalculation is complete.

        Args:
            event: The event that triggered the recalculation, typically a button click.
        """
        print('Recalculating pulse train...')
        self.disconnect_buttons()
        self.btn_recalc.color = self.button_active_color

        # Check if there are pulses to recalculate
        if len(self.emgfile["MUPULSES"][self.current_index]) == 0:
            print('No pulses selected, please select pulses or delete MU')
            return

        # Recalculate the pulse train
        spikes = self.recalc_pulse_train()

        # Recalculate peaks based on the new pulse train
        self.recalc_peaks(spikes)

        # Plot the updated motor unit
        print("Plotting new MU")
        self.sil_recalculated[self.current_index] = True
        self.plot_current_mu()
        self.fig.canvas.draw_idle()
        self.sil_recalculated[self.current_index] = False

        print("Ready for next edit")
        print('')

        # Reset button color after recalculation
        self.btn_recalc.color = self.button_color



    def recalc_pulse_train(self):
        """
        Recalculate the pulse train based on the current EMG signal and update the IPTS.

        This method performs the following tasks:
        1. Initializes an EMG object and prepares it by converting and formatting the EMG data.
        2. Extends the EMG signal and calculates the pulse train using various signal processing techniques.
        3. Applies whitening and PCA to the extended EMG signal.
        4. Detects peaks in the recalculated pulse train.
        5. Updates the IPTS with the recalculated pulse train.

        Returns:
            np.ndarray: Array of detected spikes based on the recalculated pulse train.
        """
        # Initialize EMG object and prepare the data
        emg_obj = RecalcFilter()

        emg_obj.convert_dict(self.emgfile, grid_names=self.grid_name)  # Converts MATLAB output to EMG object format
        emg_obj.grid_formatter()  # Adds spatial context to the EMG object

        emg = emg_obj.signal_dict["data"]
        extension_factor = int(np.round(emg_obj.ext_factor / len(emg)))
        emg_obj.signal_dict['extend_obvs_old'] = np.zeros([
            1, np.shape(emg)[0] * extension_factor, np.shape(emg)[1] + extension_factor - 1 - emg_obj.differential_mode
        ])

        # Prepare EMG signal and calculate pulse train according to MUedit
        spikes = self.emgfile["MUPULSES"][self.current_index]
        eSIG = extend_emg(emg_obj.signal_dict['extend_obvs_old'][0], emg, extension_factor)

        ReSIG = np.matmul(eSIG, eSIG.T) / len(eSIG)
        iReSIGt = np.linalg.pinv(ReSIG)
        E, D = pcaesig(eSIG)
        wSIG, _, dewhiteningMatrix = whiteesig(eSIG, E, D)
        wSIG_selected = wSIG[:, spikes]
        MUFilters = np.sum(wSIG_selected, axis=1)

        Pt = ((dewhiteningMatrix @ MUFilters).T @ iReSIGt) @ eSIG
        Pt = Pt[:len(emg[0])]  # Adjust the length to match the original EMG signal

        # Post-process the pulse train
        Pt[:round(0.1 * emg_obj.sample_rate)] = 0
        Pt[-round(0.1 * emg_obj.sample_rate):] = 0
        Pt = Pt * np.abs(Pt)

        # Detect peaks from new pulse train
        min_peak_distance = round(emg_obj.sample_rate * 0.005)
        spikes = detect_peaks(Pt, mpd=min_peak_distance)

        # Scale according to 10 biggest peaks
        Pt /= np.mean(maxk(Pt, 10))

        # Save new pulse train
        self.emgfile["IPTS"][self.current_index] = np.pad(Pt, (0, 4))
        self.ipts[self.current_index] = self.emgfile["IPTS"][self.current_index] 

        return spikes



    def recalc_peaks(self, spikes):
        """
        Recalculate peaks by applying k-means clustering and removing outliers.

        This method performs the following tasks:
        1. Retrieves the pulse train and IPTS data for the current motor unit.
        2. Applies k-means clustering to classify the spikes into two clusters.
        3. Identifies the cluster with the highest centroid to determine the relevant spikes.
        4. (Optionally) Removes outliers from the detected spikes based on a threshold.
        5. Updates the MUPULSES with the filtered spikes.

        Args:
            spikes (np.ndarray): Array of spike indices to be processed.
        """
        # Access the current motor unit's pulse train and IPTS data
        Pt = self.ipts[self.current_index]
        
        # Select pulse values corresponding to the provided spike indices
        pulse_values = np.array(Pt[spikes])

        # Apply k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1).fit(pulse_values.reshape(-1, 1))
        
        # Determine the cluster with the highest centroid
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes2 = spikes[np.where(kmeans.labels_ == spikes_ind)]
        
        # Optionally remove outliers
        """
        mean_val = np.mean(Pt[spikes2])
        std_val = np.std(Pt[spikes2])
        self.threshold = mean_val + 3 * std_val
        spikes2 = spikes2[Pt[spikes2] <= self.threshold]
        """
        
        # Update the MUPULSES with the filtered spikes
        self.emgfile["MUPULSES"][self.current_index] = spikes2

        
    def add_instructions(self):
        """
        Add instructions to the plot as a text box.

        This method adds a block of text to the plot with instructions for user interactions,
        such as zooming, scrolling, and selecting regions for new peaks.
        """
        instructions = (
            "Mouse Wheel: Zoom in/out\n"
            "A/D, Left/Right Arrow Keys: Scroll left/right\n"
            "Click and Drag: Select region for new peaks"
        )
        self.ax1.text(
            -0.15, 1.14, instructions, transform=self.ax1.transAxes,
            fontsize=10, verticalalignment="top"
        )

    def plot_reference_signal(self):
        """
        Plot the reference signal on a secondary y-axis.

        This method creates a secondary y-axis on the current plot to display the reference signal.
        The reference signal is plotted against the same x-axis as the primary data, with the y-axis
        label set to 'MVC'.

        The x-axis values are determined based on the `fsamp` and `timeinseconds` attributes.

        """
        ax2 = self.ax1.twinx()
        xref = (
            self.emgfile["REF_SIGNAL"].index / self.fsamp
            if self.timeinseconds
            else self.emgfile["REF_SIGNAL"].index
        )
        sns.lineplot(x=xref, y=self.emgfile["REF_SIGNAL"][0], color="0.4", ax=ax2)
        ax2.set_ylabel("MVC")


    def save_EMG_decomposition(self):
        """
        Saves the decomposed EMG data into a JSON file, compressing it using gzip.
        The function adapts the 'save_to_json()' method from OpenHDEMG.

        The saved file will contain details about the raw EMG signal, reference signal, 
        accuracy, pulse trains (MUPULSES), binary motor unit firings, and other metadata.

        The resulting JSON file is saved in the same folder as the input EMG file, 
        and the filename is appended with '_edited'.
        
        Returns:
            str: The file path of the saved JSON file.
        """

        # Define file paths and filenames
        savefolder = os.path.dirname(self.filepath)
        base = os.path.basename(self.filepath)  # Get only the file+extension
        filename = os.path.splitext(base)[0]

        # Get the binary pulse trains
        binary_mus_firing = get_binary_pulse_trains(self.emgfile["MUPULSES"], self.emgfile['RAW_SIGNAL'].shape[0])

        # Construct the new file path for the edited EMG decomposition
        self.file_path_json = os.path.join(savefolder, filename + '_edited.json')

        # Directly assign values for JSON dumping
        source = json.dumps("CUSTOMCSV")
        filename = json.dumps("training40")
        raw_signal = pd.DataFrame(self.emgfile['RAW_SIGNAL']).to_json(orient='split')
        ref_signal = pd.DataFrame(self.emgfile['REF_SIGNAL']).to_json(orient='split')
        accuracy = pd.DataFrame(self.sil_old).to_json(orient='split')
        ipts = pd.DataFrame(self.emgfile['IPTS']).to_json(orient='split')
        mupulses = json.dumps([np.array(item).tolist() for item in self.emgfile['MUPULSES']])
        fsamp = json.dumps(float(self.fsamp))
        ied = json.dumps(float(8.75))  # for grid type 4-8-L
        binary_mus_firing = pd.DataFrame(binary_mus_firing).T.to_json(orient='split')
        emg_length = json.dumps(self.emgfile['RAW_SIGNAL'].shape[0])
        number_of_mus = json.dumps(len(self.emgfile['IPTS'].columns))
        extras = pd.DataFrame([]).to_json(orient='split')

        # Create the final JSON structure
        emgfile = {
            "SOURCE": source,
            "FILENAME": filename,
            "RAW_SIGNAL": raw_signal,
            "REF_SIGNAL": ref_signal,
            "ACCURACY": accuracy,
            "IPTS": ipts,
            "MUPULSES": mupulses,
            "FSAMP": fsamp,
            "IED": ied,
            "EMG_LENGTH": emg_length,
            "NUMBER_OF_MUS": number_of_mus,
            "BINARY_MUS_FIRING": binary_mus_firing,
            "EXTRAS": extras,
        }

        # Compress and write the JSON file
        with gzip.open(self.file_path_json, "wt", encoding="utf-8", compresslevel=4) as f:
            json.dump(emgfile, f)

        return self.file_path_json