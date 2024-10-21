# EMG Decomposition and Motor Unit Editing

This project handles HD-EMG post-processing, including EMG signal decomposition to extract motor units (MUs) and provides an interface to manually edit the detected motor unit peaks. Consult the User Manual pdf in the repository for more details. This program has been created and tested in Python 3.10.5.

## Files and Directories

- **main.py**: The main script that runs the full EMG decomposition and editing workflow. Run this script and select a file to be decomposed or edit the decomposition. The mode can be set inside the script by changing the `MODE` variable (`'decompose'` or `'edit'`).
- **EditMU.py**: Contains the `EditMU` class, which provides the interface for manually editing motor unit peaks (add/remove peaks, adjust spike times).
- **EMG_Decomposition.py**: Contains the `EMGDecomposition` class, which handles the offline EMG decomposition process given a .poly5 file. This class also provides options to reject noisy channels.
- **EMG_classes.py**: Contains all functions called by `EMGDecomposition.py`, managing EMG signal processing, decomposition, and related utilities.
- **processing_tools.py**: Contains helper functions such as file loading, signal processing, and saving routines.
- **./measurements**: Directory where all measurements are stored and saved after processing. A sample measurement and (edited) decomposition is added.
- **./reader_files**: Files that handle reading of certain file types.
- **jsontomat.m**: A MATLAB script that converts OpenHDEMG-style `.json` files into `.mat` files, making the data compatible with MUEdit for further analysis. The script decompresses, decodes, and organizes EMG data, including motor unit pulse trains and associated metadata, and saves the output in `.mat` format. This allows direct comparison between datasets processed using this program and those used in MUEdit.

## Downloading or Cloning the Repository

To get started with the project, you can either download or clone the repository from GitHub (to easily copy code from below, view this .md file at the [GitHub repository](https://github.com/manuelarens/spike_train_editing)):

### Option 1: Download the Repository

1. Visit the [GitHub repository](https://github.com/manuelarens/spike_train_editing).
2. Click on the green **Code** button.
3. Select **Download ZIP**.
4. Extract the ZIP file to your desired directory.

### Option 2: Clone the Repository

To clone the repository using Git, follow these steps:`

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:

   ```bash
   git clone https://github.com/manuelarens/spike_train_editing.git
   ```

4. Once the repository is cloned, navigate into the project directory:

```bash
cd spike_train_editing
```

## Setting Up the Environment

To avoid package conflicts and ensure a consistent environment, it is recommended to create and use a virtual environment. Follow these steps:

1. **Create a Virtual Environment**:
   Open your terminal and navigate to the project directory. Run the following command to create a virtual environment named `.venv`:

   ``` bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment** (or accept IDE prompt to enter environment):
   - On **Windows**:

     ``` bash
     .venv\Scripts\activate
     ```

   - On **macOS/Linux**:

     ```bash
     source .venv/bin/activate
     ```

3. **Install Required Packages**:
   After activating the virtual environment, install all dependencies using the `requirements.txt` file provided in the project directory (this may take a minute):

   ```bash
   pip install -r requirements.txt
    ```

## Run Program

To run the script, execute the `main.py` file.

```bash
python main.py
```

Inside the script, you can select between two modes by changing the `MODE` variable:

- **Edit mode**: Assumes you have the path to a pre-decomposed `.json` file, and allows you to edit this file.
- **Decomposition mode**: Loads a selected `.poly5` file, previews the EMG channels where you can deselect certain channels, decomposes the data into motor units, and proceeds into edit mode for manual refinement of the motor unit spikes.
