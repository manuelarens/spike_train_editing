import sys
from os.path import join, dirname, realpath
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

Example_dir = dirname(realpath(__file__)) # directory of this file
modules_dir = join(Example_dir, '..') # directory with all modules
measurements_dir = join(Example_dir, '../measurements') # directory with the measurements
sys.path.append(modules_dir)

from .poly5_force_file_reader import Poly5Reader
from TMSiFileFormats.file_readers import Xdf_Reader, Edf_Reader

class MVC_Extraction:
    def __init__(self, force_channel = 'AUX 1-2'):
        self.MVC_value = 0
        self.offset_value = 0 
        self.samples = []
        self.ch_names = []
        self.sample_rate = 0
        self.force_channel = force_channel

    def run_initialize(self):
        self.load() #Load a file that can be selected yourself
        self.force_channel_ind = self.ch_names.index(self.force_channel) # gets the index of the force channel
        self.MVC_value, self.offset_value = self.extract_MAX_and_offset() # calculates the MVC and offset value of the MVC. 
        return self.MVC_value, self.offset_value
    
    def run(self):
        self.load() #Load a file that can be selected yourself
        self.force_channel_ind = self.ch_names.index(self.force_channel) # gets the index of the force channel
        self.MVC_value, self.offset_value = self.extract_MVC_and_offset() # calculates the MVC and offset value of the MVC. 
        return self.MVC_value, self.offset_value

    def load(self):
        filename = filedialog.askopenfilename(title = 'Select data file', filetypes = (('data-files', '*.poly5 .xdf .edf'),('All files', '*.*')))
        self.mne_object =[]
        try:
            if filename.lower().endswith('poly5'):
                data = Poly5Reader(filename)

                # Extract the samples and channel names from the Poly5Reader object
                self.samples = data.samples
                self.ch_names = data.ch_names
                self.sample_rate = data.sample_rate
                # Conversion to MNE raw array
                self.mne_object = data.read_data_MNE() 
            elif filename.lower().endswith('xdf'):
                # Extract the samples and channel names from the XDF object
                reader = Xdf_Reader(filename)
                data = reader.data[0]
                
                self.samples = data.get_data()
                self.ch_names = data.ch_names
                self.sample_rate = data.info['sfreq']
                self.num_channels = len(self.ch_names)
            elif not filename:
                tk.messagebox.showerror(title='No file selected', message = 'No data file selected.')

            else:
                tk.messagebox.showerror(title='Could not open file', message = 'File format not supported. Could not open file.')

        except:
            tk.messagebox.showerror(title='Could not open file', message = 'Something went wrong. Could not open file.')

    def extract_MVC_and_offset(self):
        """
        Extract the mvc and offset. Applies moving average over 250 ms window.
        Normalizes this moving average by substracting offset and scaling it between 0 and 100. 
        The lowest value of this moving average is the offset.
        Calculates the consequtive differences and saves them if they are higher than threshold.
        Searches for periods were these consequtive differences are more than self.sample_rate (1 second) apart. 
        Flags these periods as plateaus and adds them in a array. 
        Takes the maximum values of these plateaus as the MVC value.
        """
        #Calculate moving average over 250 ms (highest value is MVC and lowest value is offset) 
        #Some recordings exhibit zeros at the end of the file so these will not be included in the moving mean. 
        w = int(0.25*self.sample_rate)
        self.mov_mean = np.convolve(abs(self.samples[self.force_channel_ind,:-1000]), np.ones(w), "valid") / w
        self.MVC_value = np.max(self.mov_mean)
        self.offset_value = np.min(self.mov_mean)
        self.mov_mean_norm = (self.mov_mean - self.offset_value ) / (self.MVC_value - self.offset_value) * 100
        differences = np.diff(self.mov_mean_norm)
        thr = 25/self.sample_rate
        # Find indices where absolute differences exceed the threshold
        significant_diff_indices = np.where(abs(differences) > thr)[0]

        # Find consecutive sequences of significant differences
        plateaus = np.array([])
        for i in range(1, len(significant_diff_indices)):
            significant_index_value = self.mov_mean_norm[significant_diff_indices[i]] 
            if significant_diff_indices[i] - significant_diff_indices[i-1] > self.sample_rate and significant_index_value > 10:
                plateau_segment = self.mov_mean[significant_diff_indices[i-1]:significant_diff_indices[i]]
                plateaus = np.concatenate((plateaus, plateau_segment))                
        
        try:
            self.MVC_value = np.max(plateaus)

        except ValueError as e:
            print(e)


        return self.MVC_value, self.offset_value
    
    def extract_MAX_and_offset(self):
        #Roughly calculate max and offset of the array, meant for offset and max value determination of load cell
        #moving average over 250 ms (highest value is MAX and lowest value is offset) 
        #Some recordings exhibit zeros at the end of the file so these will not be included in the moving mean. 
        w = int(0.250*self.sample_rate)
        self.mov_mean = np.convolve(abs(self.samples[self.force_channel_ind,:-1000]), np.ones(w), "valid") / w
        self.MVC_value = np.max(self.mov_mean)
        self.offset_value = np.min(self.mov_mean)
        return self.MVC_value, self.offset_value