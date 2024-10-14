# EMG Decomposition and Motor Unit Editing

This project handles EMG (Electromyography) data processing, including EMG signal decomposition to extract motor units (MUs) and provides an interface to manually edit the detected motor unit peaks.

## Features

- **EMG Data File Selection**: Users can select and load EMG data from Poly5, XDF, or other supported file formats. In **edit mode**, users provide the path to a pre-decomposed JSON file.
- **Offline EMG Decomposition**: Decomposes EMG data into motor units and saves them as a JSON file. The decomposition process includes the ability to reject noisy or unconnected channels before decomposition.
- **Motor Unit Visualization**: Displays raw EMG signals and decomposed motor unit firing patterns.
- **Manual Editing of Motor Units**: Provides an interactive interface to adjust motor unit peaks by adding, removing, or adjusting spikes.
- **Saving Edited Motor Units**: Edited motor unit data can be saved back to a file. The saved file will be placed in the same directory as the selected input file.

## Files and Directories

- **main.py**: The main script that runs the full EMG decomposition and editing workflow. Run this script and select a file to be decomposed or edit the decomposition. The mode can be set inside the script by changing the `MODE` variable (`'decompose'` or `'edit'`).
- **EditMU.py**: Contains the `EditMU` class, which provides the interface for manually editing motor unit peaks (add/remove peaks, adjust spike times).
- **EMG_Decomposition.py**: Contains the `EMGDecomposition` class, which handles the offline EMG decomposition process given a .poly5/.xdf file. This class also provides options to reject noisy channels.
- **EMG_classes.py**: Defines the EMG-related classes and utilities used throughout the project.
- **RecalcFilter.py**: Contains the `RecalcFilter` class, used to recalculate the filter for the current motor unit based on user modifications to the spikes. This class is essential for adjusting motor unit detection based on filtered signal updates.
- **processing_tools.py**: Contains helper functions such as file loading, signal processing, and saving routines.

## Run program

To run the script, execute the `main.py` file. Inside the script, you can select between two modes by changing the `MODE` variable:

- **Edit mode**: Assumes you have the path to a pre-decomposed `.json` file, and allows you to edit this file.
- **Decomposition mode**: Loads a selected `.poly5` file, previews the EMG channels where you can deselect certain channels, decomposes the data into motor units, and proceeds into edit mode for manual refinement of the motor unit spikes.

## Prerequisites

Ensure you have the following Python packages installed:

- `numpy`
- `scipy`
- `mne`
- `pandas`
- `matplotlib`
- `tkinter` (for file dialog)
- `gzip` (for handling compressed files)
- `openhdemg`
- `seaborn`
- `numba`
- `scikit-learn`
- `tqdm`

You can install these dependencies using pip:

```bash
pip install numpy scipy mne pandas matplotlib tkinter openhdemg seaborn numba scikit-learn tqdm
```
