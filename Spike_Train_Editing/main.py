"""
This script handles EMG data processing including:
1. Selecting and reading EMG data files.
2. Running offline EMG decomposition to extract motor units.
3. Displaying raw EMG signals and motor units.
4. Editing motor unit peaks via an interactive interface.
5. Saving the edited motor unit data back to a file.

The script utilizes a custom `EditMU` class for peak editing.
"""

import sys
from os.path import join, dirname, realpath

# Define paths for modules and measurements directories
EXAMPLE_DIR = dirname(realpath(__file__))  # Directory of this script file
MODULES_DIR = join(EXAMPLE_DIR, '..')  # Parent directory with all modules
MEASUREMENTS_DIR = join(EXAMPLE_DIR, '../measurements')  # Directory with all measurements
sys.path.append(MODULES_DIR)  # Add modules directory to system path for importing custom modules

# Now import the modules that depend on the path
import numpy as np
from tkinter import filedialog
import openhdemg.library as emg
from EMG_Decomposition import EMG_Decomposition
from TMSiFileFormats.file_readers import Poly5Reader
from EditMU import EditMU
from processing_tools import emg_from_json

GRID_TYPE = '4-8-L'


def main():
    """
    Main function to handle the EMG data processing workflow.
    It starts with file selection, data display, offline decomposition, and finally
    editing and saving motor unit data.
    """

    
    # Open file dialog to select the EMG data file
    #"""
    filepath = filedialog.askopenfilename(
        title='Select data file',
        filetypes=(('Data files (.poly5, .xdf)', '*.poly5 *.xdf'), ('All files', '*.*')),
        initialdir=MEASUREMENTS_DIR
    )
    #"""

    #"""
    # If no file is selected, exit the script
    if filepath == '':
        print("No file selected. Exiting the script.")
        sys.exit()

    # Display the raw EMG file using MNE
    display_raw_emg(filepath)
    #"""

    # Run offline EMG decomposition
    filepath_decomp = run_offline_decomposition(filepath)

    #filepath_decomp = r'C:\\Manuel\\Uni\\Master\\Stage\\Code\\tmsi-python-interface-main\\tmsi-python-interface-main\\measurements\\training_measurement-20240611_085328_decomp.json' #own decomp
    #filepath_decomp = r'C:\\Manuel\\Uni\\Master\\Stage\\Code\\tmsi-python-interface-main\\tmsi-python-interface-main\\measurements\\training_20240611_085441_decomp.json' #tmsi decomp
    #filepath_decomp = r'C:\\Manuel\\Uni\\Master\\Stage\\Code\\tmsi-python-interface-main\\tmsi-python-interface-main\\measurements\\Pre_25_b.json' #openhdemg decomp

    # Load decomposed motor units and enable editing
    edit_decomposed_mu(filepath_decomp)


def display_raw_emg(filepath):
    """
    Function to read and display raw EMG data from the selected file.
    Only connected channels are displayed to avoid unnecessary noise.
    """
    # Read the Poly5 data file
    data = Poly5Reader(filepath)
    mne_object = data.read_data_MNE(add_ch_locs=True)  # Load data into MNE object

    # Get channel names and raw data samples
    ch_names = mne_object.info['ch_names']
    samples_mne = mne_object._data

    # Filter out unconnected channels (those with all-zero signals)
    show_chs = []
    for idx, ch in enumerate(mne_object._data):
        if ch.any():
            show_chs = np.hstack((show_chs, mne_object.info['ch_names'][idx]))

    # Pick only the channels to display
    data_object = mne_object.pick(show_chs)

    # Plot the selected EMG channels
    data_object.plot(
        scalings=dict(eeg=250e-6),
        start=0, duration=5, n_channels=5,
        title=filepath, block=True
    )


def run_offline_decomposition(filepath):
    """
    Function to run offline EMG decomposition on the selected file.
    Decomposed motor units are saved in a JSON file for further processing.
    """
    print('START OFFLINE DECOMPOSITION')

    # Create EMG decomposition object and run the decomposition process
    offline_decomp = EMG_Decomposition(filepath=filepath)
    offline_decomp.run(grid_name=GRID_TYPE)

    # Get the path to the saved decomposed JSON file
    filepath_decomp = offline_decomp.emg_obj.file_path_json
    print("OFFLINE DECOMPOSITION DONE")

    return filepath_decomp


def edit_decomposed_mu(filepath_decomp):
    """
    Function to load the decomposed motor units from a JSON file,
    allow for manual editing, and save the edited motor units.
    """
    # Load the EMG decomposition data from the JSON file
    emgfile = emg_from_json(filepath_decomp)
    emgfile = emg.sort_mus(emgfile)  # Sort motor units by discharge time

    # Plot the motor unit pulses
    emg.plot_mupulses(emgfile)

    # Open the editor to manually adjust peaks
    mu_editor = EditMU(emgfile, filepath_decomp)
    
    # Save the edited motor units back to a file
    filepath_decomp_edited = mu_editor.save_EMG_decomposition()

    # Load and plot the edited motor units
    emgfile_edited = emg_from_json(filepath_decomp_edited)
    emgfile_edited = emg.sort_mus(emgfile_edited)
    emg.plot_mupulses(emgfile_edited)


if __name__ == "__main__":
    main()
