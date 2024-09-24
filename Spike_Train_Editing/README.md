# EMG Decomposition and Motor Unit Editing

This project handles EMG (Electromyography) data processing, including EMG signal decomposition to extract motor units (MUs) and provides an interface to manually edit the detected motor unit peaks.

## Features

- **EMG Data File Selection**: Users can select and load EMG data from Poly5, XDF, or other supported file formats.
- **Offline EMG Decomposition**: Decomposes EMG data into motor units and saves them as a JSON file.
- **Motor Unit Visualization**: Displays raw EMG signals and decomposed motor unit firing patterns.
- **Manual Editing of Motor Units**: Provides an interactive interface to adjust motor unit peaks.
- **Saving Edited Motor Units**: Edited motor unit data can be saved back to a file for further analysis. The files are saved back to the directory of the file that is selected.

## Files and Directories

- **main.py**: The main script that runs the full EMG decomposition and editing workflow. Run this script and select file to be decomposed and edit the decomposition.
- **EditMU.py**: Contains the `EditMU` class, which provides the interface for manually editing motor unit peaks.
- **EMG_Decomposition.py**: Contains the `EMGDecomposition` class, which handles the offline EMG decomposition process given a .poly5/.xdf file.
- **EMG_classes.py**: Defines the EMG-related classes and utilities used throughout the project.
- **RecalcFilter.py**: Contains the `RecalcFilter` class, used to recalculate the filter for the current motor unit
- **processing_tools.py** Contains helper functions

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
