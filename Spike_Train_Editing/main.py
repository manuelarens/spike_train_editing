import sys
from os.path import join, dirname, realpath
Example_dir = dirname(realpath(__file__)) # directory of this file
modules_dir = join(Example_dir, '..') # directory with all modules
measurements_dir = join(Example_dir, '../measurements') # directory with all measurements
sys.path.append(modules_dir)
import numpy as np

import openhdemg.library as emg

from tkinter import filedialog 

from offline_EMG import EMG_Decomposition
from TMSiFileFormats.file_readers import Poly5Reader, Xdf_Reader, Edf_Reader
from EditMU import EditMU

grid_type = '4-8-L'


filepath = filedialog.askopenfilename(title = 'Select data file', filetypes = (('data-file (.poly5)', '*.poly5'),('All files', '*.*')),initialdir=measurements_dir)
#"""
if filepath == '':
    print("No file selected. Exiting the script.")
    sys.exit()  # Exit the script

### DISPLAY FILE
data = Poly5Reader(filepath)
mne_object = data.read_data_MNE(add_ch_locs = True)

info_mne = mne_object.info
ch_names = mne_object.info['ch_names']
samples_mne = mne_object._data

# Do not show unconnected channels
show_chs = []
for idx, ch in enumerate(mne_object._data):
    if ch.any():
        show_chs = show_chs = np.hstack((show_chs,mne_object.info['ch_names'][idx]))
data_object = mne_object.pick(show_chs)
data_object.plot(scalings=dict(eeg = 250e-6), start= 0, duration = 5, n_channels = 5, title = filepath, block = True) 
#"""


"""
### DECOMPOSE FILE

print('START OFFLINE DECOMPOSITION')
offline_decomp = EMG_Decomposition(filepath=filepath, rejected_chan=EMG_reject)
offline_decomp.run(grid_name=grid_type)
filepath_decomp = offline_decomp.emg_obj.file_path_json
print("OFFLINE DECOMPOSITION DONE")
#"""

### DISPLAY DECOMPOSED MOTOR UNITS AND EDIT PEAKS
#emgfile = filepath_decomp
emgfile = emg.emg_from_json(r'C:\Manuel\Uni\Master\Stage\Code\tmsi-python-interface-main\tmsi-python-interface-main\measurements\Pre_25_b.json')
#print(emgfile["SOURCE"])

emgfile = emg.sort_mus(emgfile)
plotter = EditMU(emgfile)

