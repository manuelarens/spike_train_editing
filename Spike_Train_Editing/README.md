# EMG Decomposition and Motor Unit Editing

This project handles EMG (Electromyography) data processing, including EMG signal decomposition to extract motor units (MUs) and provides an interface to manually edit the detected motor unit peaks.

## Features

- **EMG Data File Selection**: Users can select and load EMG data from Poly5, XDF, or other supported file formats.
- **Offline EMG Decomposition**: Decomposes EMG data into motor units and saves them as a JSON file.
- **Motor Unit Visualization**: Displays raw EMG signals and decomposed motor unit firing patterns.
- **Manual Editing of Motor Units**: Provides an interactive interface to adjust motor unit peaks.
- **Saving Edited Motor Units**: Edited motor unit data can be saved back to a file for further analysis.

## Files and Directories

- **main.py**: The main script that runs the full EMG decomposition and editing workflow.
- **EditMU.py**: Contains the `EditMU` class, which provides the interface for manually editing motor unit peaks.
- **EMG_Decomposition.py**: Handles the offline EMG decomposition process, using a defined grid type for motor unit extraction.
- **EMG_classes.py**: Defines the EMG-related classes and utilities used throughout the project.

## Prerequisites

Ensure you have the following Python packages installed:

- `numpy`
- `scipy`
- `mne`
- `pandas`
- `matplotlib`
- `tkinter` (for file dialog)
- `gzip` (for handling compressed files)

You can install these dependencies using pip:

```bash
pip install numpy scipy mne pandas matplotlib tkinter
