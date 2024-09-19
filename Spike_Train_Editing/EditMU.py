import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import seaborn as sns
import pandas as pd
import numpy as np
import time
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import messagebox


from openhdemg.library.mathtools import compute_sil
from openhdemg.library.plotemg import showgoodlayout
from recalc_filter import recalc_filter
from processing_tools import *

 

class EditMU:
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

        self.peak_artists = []

        # Create plot
        self.fig, self.ax1 = plt.subplots(
            figsize=(figsize[0] / 2.54, figsize[1] / 2.54), num="IPTS"
        )

        # Initialize the rectangleselectors
        self.rect_selector_add = RectangleSelector(self.ax1, self.onselect_add, useblit=True, button=[1])
        self.rect_selector_add.set_active(False)

        self.rect_selector_remove = RectangleSelector(self.ax1, self.onselect_remove, useblit=True, button=[1])
        self.rect_selector_remove.set_active(False)

        self.remove_spikes_boolean = False
        self.add_spikes_boolean = False

        # Plot the initial MU
        self.plot_current_mu()

        # Add zoom, scroll, and key press events
        self.fig.canvas.mpl_connect("scroll_event", self.zoom)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Add previous and next buttons
        self.add_buttons()

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
        pulses = self._process_mu_pulses(self.emgfile["MUPULSES"])
        pulses = pulses[self.current_index]

        # Clear existing peak artists
        for peak_artist in self.peak_artists:
            peak_artist.remove()

        # Plot each pulse as a red circle
        for pulse in pulses:
            closest_idx = (np.abs(self.x_axis - pulse)).argmin()  # Find closest point on the x-axis
            y_value = self.ipts.iloc[closest_idx, self.current_index]  # Get the y-value
            peak_artist, = self.ax1.plot(pulse, y_value, "ro", markersize=2, picker=True)  # Enable picking
            self.peak_artists.append(peak_artist)


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
        self.button_color = "ivory"
        self.button_active_color = "mistyrose"

        # Previous MU
        ax_prev = plt.axes([0.01, 0.025, 0.12, 0.04])
        self.btn_prev = Button(ax_prev, "Previous", color=self.button_color)
        self.btn_prev.on_clicked(self.previous_mu)

        # Next MU
        ax_next = plt.axes([0.87, 0.025, 0.12, 0.04])
        self.btn_next = Button(ax_next, "Next", color=self.button_color)
        self.btn_next.on_clicked(self.next_mu)

        # Add spikes
        ax_add = plt.axes([0.85, 0.82, 0.14, 0.04])
        self.btn_add = Button(ax_add, "Add spikes", color=self.button_color)
        self.btn_add.on_clicked(self.add_spikes)
        
        # Remove spikes
        ax_remove = plt.axes([0.85, 0.76, 0.14, 0.04])
        self.btn_remove = Button(ax_remove, "Remove spikes", color=self.button_color)
        self.btn_remove.on_clicked(self.remove_spikes)

        # Recalculate filter
        ax_recalc = plt.axes([0.85, 0.70, 0.14, 0.04])
        self.btn_recalc = Button(ax_recalc, "Recalc. filter", color=self.button_color)
        self.btn_recalc.on_clicked(self.recalc_filter)

        ax_delete = plt.axes([0.85, 0.95, 0.14, 0.04])
        self.btn_delete = Button(ax_delete, "Delete MU", color='tomato', hovercolor='salmon')
        self.btn_delete.on_clicked(self.delete_MU)

        self.fig.canvas.draw_idle()  # Force canvas update after button creation

    def delete_MU(self, event):
        # Create a Tkinter root window (it will be hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Show a confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            "Are you sure you want to delete this motor unit?",
            icon='warning',  # Adds a warning icon to the dialog
            parent=root
        )

        # Proceed with deletion only if the user clicks "Yes"
        if confirm:
            self.disconnect_buttons()

            # Drop the column corresponding to the current index
            self.emgfile['IPTS'].drop(columns=[self.current_index], inplace=True)
            self.emgfile['IPTS'].columns = range(self.emgfile['IPTS'].shape[1])

            # Remove the motor unit pulses at the current index
            del self.emgfile['MUPULSES'][self.current_index]

            # Update the current index to the last valid motor unit
            if self.current_index >= len(self.emgfile['MUPULSES']):
                self.current_index = len(self.emgfile['MUPULSES']) - 1
                if self.current_index == -1:
                    print("No MU's remaining, closing script")
                    plt.close(self.fig)

            # Plot the current MU if there are any left
            if self.current_index >= 0:
                self.plot_current_mu()
                self.fig.canvas.draw_idle()

        # Destroy the root window
        root.destroy()
        
    def add_spikes(self, event):
        if not self.add_spikes_boolean:
            # turn off other one
            if self.remove_spikes_boolean:
                self.rect_selector_remove.set_active(False)
                self.remove_spikes_boolean = False
                self.btn_remove.color = self.button_color

            # Add RectangleSelector
            self.rect_selector_add.set_active(True)
            self.add_spikes_boolean = True
            self.btn_add.color = self.button_active_color
            self.fig.canvas.draw_idle()
        else:
            self.disconnect_buttons()
            
    def remove_spikes(self, event):
        if not self.remove_spikes_boolean:
            # turn off other one
            if self.add_spikes_boolean:
                self.rect_selector_add.set_active(False)
                self.add_spikes_boolean = False
                self.btn_add.color = self.button_color

            self.rect_selector_remove.set_active(True)
            self.remove_spikes_boolean = True
            self.btn_remove.color = self.button_active_color
            self.fig.canvas.draw_idle()
        else:
            self.disconnect_buttons()
    
    
    def disconnect_buttons(self):
        # Disconnect add and remove spike buttons
        self.rect_selector_add.set_active(False)
        self.add_spikes_boolean = False
        self.btn_add.color = self.button_color

        self.rect_selector_remove.set_active(False)
        self.remove_spikes_boolean = False
        self.btn_remove.color = self.button_color

    def previous_mu(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.plot_current_mu()
            self.fig.canvas.draw_idle()
            self.disconnect_buttons()
            

    def next_mu(self, event):
        if self.current_index < len(self.emgfile["IPTS"].columns) - 1:
            self.current_index += 1
            self.plot_current_mu()
            self.fig.canvas.draw_idle()
            self.disconnect_buttons()
            
    def onselect_add(self, eclick, erelease):
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

            # Plot the peak and store it in self.peak_artists
            peak_artist, = self.ax1.plot(xmax, ymax, "ro", markersize=2, label="Peak")
            self.peak_artists.append(peak_artist)  # Store the artist object
            

            #add extra point to the emgfile
            x_idx = xmax*self.fsamp
            index = np.searchsorted(self.emgfile["MUPULSES"][self.current_index], x_idx)
            self.emgfile["MUPULSES"][self.current_index] = np.insert(self.emgfile["MUPULSES"][self.current_index], index, x_idx)

        self.fig.canvas.draw_idle()

    def onselect_remove(self, eclick, erelease):
        # Get coordinates of the rectangle selection
        x1, x2 = eclick.xdata, erelease.xdata
        y1, y2 = eclick.ydata, erelease.ydata
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Find points within the selection area
        mask = (
            (self.x_axis > min(x1, x2)) &
            (self.x_axis < max(x1, x2)) &
            (self.emgfile["IPTS"][self.current_index] > min(y1, y2)) &
            (self.emgfile["IPTS"][self.current_index] < max(y1, y2))
        )
        xmasked = self.x_axis[mask]
        ymasked = self.emgfile["IPTS"][self.current_index][mask]

        # Initialize a deletion counter
        delete_count = 0
        max_deletions = 10  # Set maximum deletions to 10

        if len(xmasked) > 0:
            # Loop through all selected peaks and remove them
            for i, (xmax, ymax) in enumerate(zip(xmasked, ymasked)):
                # Find the closest pulse (in samples) to remove
                x_idx = int(xmax * self.fsamp)  # Convert xmax to sample index
                pulses = self.emgfile["MUPULSES"][self.current_index]
                index = np.searchsorted(pulses, x_idx)
                if index < len(pulses) and pulses[index] == x_idx:  # Check if the index matches
                    self.emgfile["MUPULSES"][self.current_index] = np.delete(pulses, index)

                    # Increment the deletion counter
                    delete_count += 1
                    if delete_count >= max_deletions:
                        # Stop if we have reached the maximum number of deletions
                        print('Limit of 10 deletions at a time reached')

                        # Call the plot_peaks function to update the figure
                        self.plot_peaks()
                        self.fig.canvas.draw_idle()
                        return
                    
            # Call the plot_peaks function to update the figure
            self.plot_peaks()
            self.fig.canvas.draw_idle()



    
    def recalc_filter(self, event):
        
        print('Recalculating pulse train...')
        self.disconnect_buttons()
        self.btn_recalc.color = self.button_active_color

        # Recalculate the pulse train
        spikes = self.recalc_pulse_train()

        # Placeholder for recalculating peaks
        self.recalc_peaks(spikes)

        ###### plotting
        print("Plotting new MU")
        self.plot_current_mu()
        self.fig.canvas.draw_idle()
        print("Ready for next edit")
        print('')

        self.btn_recalc.color = self.button_color


    def recalc_pulse_train(self):
        """Recalculates the pulse train based on the current EMG signal and updates IPTS."""
        emg_obj = recalc_filter()

        emg_obj.convert_dict(self.emgfile, grid_names=['4-8-L'])  # adds signal_dict to the emg_obj, using Matlab output of ISpin
        emg_obj.grid_formatter()  # adds spatial context

        emg = emg_obj.signal_dict["data"]
        extension_factor = int(np.round(emg_obj.ext_factor / len(emg)))
        emg_obj.signal_dict['extend_obvs_old'] = np.zeros([1, np.shape(emg)[0] * extension_factor, np.shape(emg)[1] + extension_factor - 1 - emg_obj.differential_mode])

        spikes = self.emgfile["MUPULSES"][self.current_index]

        eSIG = extend_emg(emg_obj.signal_dict['extend_obvs_old'][0], emg, extension_factor)

        ReSIG = np.matmul(eSIG, eSIG.T) / len(eSIG)
        iReSIGt = np.linalg.pinv(ReSIG)
        E, D = pcaesig(eSIG)
        wSIG, _, dewhiteningMatrix = whiteesig(eSIG, E, D)
        wSIG_selected = wSIG[:, spikes]
        MUFilters = np.sum(wSIG_selected, axis=1)

        Pt = ((dewhiteningMatrix @ MUFilters).T @ iReSIGt) @ eSIG
        Pt = Pt[:len(emg[0])]  # Keep the size the same as the original EMG signal

        Pt[:round(0.1 * emg_obj.sample_rate)] = 0
        Pt[-round(0.1 * emg_obj.sample_rate):] = 0
        Pt = Pt * np.abs(Pt)

        min_peak_distance = round(emg_obj.sample_rate * 0.005)
        spikes = detect_peaks(Pt, mpd = min_peak_distance)

        Pt /=  np.mean(maxk(Pt, 10))

        self.emgfile["IPTS"][self.current_index] = np.pad(Pt, (0, 4))
        return spikes


    def recalc_peaks(self, spikes):
            
        """Recalculate peaks by applying k-means clustering and removing outliers."""
        
        # Access the current MU's pulse train and IPTS data
        Pt = self.emgfile["IPTS"][self.current_index]
        
        # Select the pulse values for clustering
        pulse_values = np.array(Pt[spikes])
        
        # Apply k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_values.reshape(-1,1)) # 2D array with 1 column

        spikes_ind = np.argmax(kmeans.cluster_centers_) # Determine highest centroid
        # fix below 
        spikes2 = spikes[np.where(kmeans.labels_ == spikes_ind)]
                
        # Remove outliers: spikes where the pulse values are greater than mean + 3*std
        """
        mean_val = np.mean(Pt[spikes2])
        std_val = np.std(Pt[spikes2])
        self.threshold = mean_val + 3 * std_val
        spikes2 = spikes2[Pt[spikes2] <= self.threshold]
        #"""
        
        # Update MUPULSES with the filtered spikes
        self.emgfile["MUPULSES"][self.current_index] = spikes2
        
        



    def add_instructions(self):
        instructions = (
            "Mouse Wheel: Zoom in/out\n"
            "A/D, Left/Right Arrow Keys: Scroll left/right\n"
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
