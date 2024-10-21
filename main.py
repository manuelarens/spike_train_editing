"""
This script handles EMG data processing including:
1. Selecting and reading EMG data files.
2. Running offline EMG decomposition to extract motor units.
3. Displaying raw EMG signals and motor units.
4. Editing motor unit peaks via an interactive interface.
5. Saving the edited motor unit data back to a file.

The script utilizes a custom `EditMU` class for Motor Unit editing.
"""

from tkinter import filedialog
import sys
from os.path import join, dirname, realpath
import numpy as np
import openhdemg.library as emg

# Define paths for modules and measurements directories
EXAMPLE_DIR = dirname(realpath(__file__))  # Directory of this script file
MODULES_DIR = join(EXAMPLE_DIR, '..')  # Parent directory with all modules
MEASUREMENTS_DIR = '.\\measurements'  # Directory with all measurements
sys.path.append(MODULES_DIR)  # Add modules directory to system path for importing custom modules

# Now import the modules that depend on the path
from processing_tools import emg_from_json
from EMG_Decomposition import EMGDecomposition
from EditMU import EditMU
from reader_files.poly5reader import Poly5Reader

# Define the mode: 'decompose' or 'edit'
MODE = 'edit'  # Choose either 'decompose' for .poly5 files or 'edit' for pre-decomposed .json files

GRID_NAMES = ['4-8-L']  # If ngrids > 1, fill in ['name_grid1', 'name_grid2', etc...]

def main():
    """
    Main function to handle the EMG data processing workflow.
    Depending on the mode, either decomposes a .poly5 file or edits a .json file.
    """

    if MODE == 'decompose':
        # Open file dialog to select the .poly5 data file for decomposition
        filepath = filedialog.askopenfilename(
            title='Select data file',
            filetypes=(('Data files (.poly5)', '*.poly5'), ('All files', '*.*')),
            initialdir=MEASUREMENTS_DIR
        )

        # If no file is selected, exit the script
        if not filepath:
            print("No file selected. Exiting the script.")
            sys.exit()

        # Users are prompted to reject noisy/unconnected channels before decomposition.
        # Only the selected channels will be used for motor unit decomposition.
        # Click on channel name to toggle on/off
        rejected_chan = display_raw_emg(filepath)

        # Run offline decomposition
        filepath_decomp = run_offline_decomposition(filepath, rejected_chan)

    elif MODE == 'edit':
        # Define the path to the pre-decomposed .json file
        filepath_decomp = join(
            MEASUREMENTS_DIR,
            'training_measurement-20240611_085328_decomp.json'
        )

        #filepath_decomp = join(MEASUREMENTS_DIR, 'training_20240611_085441_decomp.json' #tmsi decomp
        #filepath_decomp = join(MEASUREMENTS_DIR, 'Pre_25_b.json' #openhdemg decomp

    else:
        raise ValueError("Invalid MODE. Choose 'decompose' or 'edit'.")

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

    # Filter out unconnected channels (those with all-zero signals)
    print('Click on the names of channels you want to reject, channels turn transparent when de-selected')
    show_chs = []
    for idx, ch in enumerate(mne_object.get_data()):
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
    return data_object.info['bads']

def run_offline_decomposition(filepath, rejected_chan):
    """
    Function to run offline EMG decomposition on the selected file.
    Decomposed motor units are saved in a JSON file for further processing.
    """
    print('START OFFLINE DECOMPOSITION')

    # Create EMG decomposition object and run the decomposition process
    offline_decomp = EMGDecomposition(filepath=filepath, rejected_chan = rejected_chan)
    offline_decomp.run(grid_names=GRID_NAMES)

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
    #emg.plot_mupulses(emgfile)

    # Open the editor to manually adjust peaks
    mu_editor = EditMU(emgfile, filepath_decomp)

    # Save the edited motor units back to a file
    print('Saving edited decomposition...')
    filepath_decomp_edited = mu_editor.save_EMG_decomposition()

    # Load and plot the edited motor units
    print('Save complete! Showing new spike trains')
    emgfile_edited = emg_from_json(filepath_decomp_edited)
    emgfile_edited = emg.sort_mus(emgfile_edited)
    emg.plot_mupulses(emgfile_edited)

if __name__ == "__main__":
    main()
