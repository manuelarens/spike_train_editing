import os
import numpy as np
import matplotlib.pyplot as plt
from processing_tools import *
import tkinter as tk
from tkinter import simpledialog

from scipy.signal import detrend
from scipy.io import loadmat
import pandas as pd 
import json 
import tkinter as tk
import gzip 
from TMSiFileFormats.file_readers import Xdf_Reader
from Test_Scripts_Nathan.poly5_force_file_reader import Poly5Reader
# root = tk.Tk()
np.random.seed(1337)


class EMG():
    
    def __init__(self):
        # processing settings
        self.its = 100 # number of iterations of the fixed point algorithm 
        self.ref_exist = 1 # Boolean for whether an existing reference is used for signal batching (otherwise, manual selection)
        self.windows = 1  # number of segmented windows over each contraction
        self.check_emg = 0 # Boolean for the review process of EMG channels, where 0 = Automatic selection 1 = Visual checking
        self.drawing_mode = 0 # 0 = Output in the command window ; 1 = Output in a figure
        self.differential_mode = 0 # 0 = no; 1 = yes (filter out the smallest MU, can improve decomposition at the highest intensities
        self.peel_off = 0 # Boolean to determine whether we will update the residual EMG by removing the motor units with the highest SIL value
        self.peeloffwin = 0.025 # Duration of windows for checking for MUAPs in the EMG signal
        self.cov_filter = 0 # 0 = no; 1 = yes (Boolean to determine to filter MU filters based on CoV)
        self.cov_thr = 0.5 # Threshold for CoV of ISI values
        self.sil_thr = 0.9 # Threshold for SIL values (matching Negro et.al 2016)
        self.CoVDR = 0.3 #Threshold for CoV of Discharge Rate that we want to reach for cleaning discharge times when refineMU is on
        self.silthrpeeloff = 0.9 # Threshold for MU removed from the signal (if the sparse  deflation is on)
        self.ext_factor = 1000 # extension of observations for numerical stability 
        self.edges2remove = 0.5 # trimming the batched data, to remove the effects of spectral leakage
        self.plat_thr = 0.01 # giving some padding about the segmentation of the plateau region, if used
        # post processing
        self.alignMUAP = 0 # Boolean to determine whether we will realign the discharge times with the peak of MUAPs (channel with the MUAP with the highest p2p amplitudes, from double diff EMG signal)
        self.refineMU = 0 # Boolean to determine whether we refine MUs, involve 1) removing outliers (1st time), 2) revaluating the MU pulse trains
        self.dup_thr = 0.3 # Threshold that defines the minimal percentage of common discharge times between duplicated motor units
        self.cov_dr = 0.3 # Threshold that define the CoV of discharge that we aim to reach, if we refine the MUs (i.e. refineMU = 1)

class offline_EMG(EMG):

    # child class of EMG, so will inherit it's initialisaiton
    def __init__(self, to_filter, rejected_chan = []):
        super().__init__() # Using constructor of parent class (EMG)
        self.to_filter = to_filter # Boolean to determine whether or not you notch and butter filter the signal, more relevant for matching real-time protocols
        self.rejected_chan = rejected_chan

    def select_file(self, filepath):
         # select a file and store path to the file 
        # filepath = r'C:/Users/natha/OneDrive - Universiteit Twente/Universiteit/Master/Internship/Python Interface/tmsi-python-interface/measurements/Training_measurement+ForceProfile-20240410_140900.poly5'
        self.filepath_poly5_xdf = filepath
        filepath = filepath.replace('/', '\\') # switch slashes
        self.filepath = filepath 
        self.savefolder = os.path.dirname(filepath) # store in which folder to save the decomposition result 
        base=os.path.basename(filepath) #get only the file+extension  
        self.filename = os.path.splitext(base)[0] # get the name of the file without the extension 
        
    def convert(self): # input file: Matlabfile (build on output of ISpin)
        signal_mat = loadmat(self.filepath)['signal'] #from .mat, import only the struct 'signal'
        """get sample frequency, number of channels, grids and muscle names, the EMG signal, 
        the target and the path of the force profile """
        # specify fields of the dataset 
        fsamp = int(signal_mat['fsamp']) 
        nchans = int(signal_mat['nChan'])
        ngrids = int(signal_mat['ngrid'])
        grid_name = str(signal_mat['gridname'][0][0][0][0][0]) #for now: 1 grid! TODO: adjust
        grid_names = [grid_name]
        muscle_name=str(signal_mat['muscle'][0][0][0][0][0]) #for now: 1 muscle! TODO: adjust 
        muscle_names = [muscle_name] 

        # read in the EMG trial data
        emg_data = signal_mat['data'][0][0]

        # create a dictionary containing all relevant signal parameters and data
        signal = dict(data = emg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids,grids = grid_names[:ngrids],muscles = muscle_names[:ngrids]) # discard the other muscle and grid entries, not relevant

        # if the signals were recorded with a feedback generated by ISpin, get the target and the path performed by the participant
        if self.ref_exist:
            target = signal_mat['target'][0][0][0]
            path = signal_mat['path'][0][0][0]
            signal['path'] = path
            signal['target'] = target
        
        self.signal_dict = signal
        self.decomp_dict = {} # initialising this dictionary here for later use
        self.dict = {} # initialising this dictionary here for later use
        return
    
    def convert_poly5_xdf(self, grid_names = ['TMSi8-8-L'], muscle_names = ['BB']):
        try:
            if self.filepath_poly5_xdf.lower().endswith('poly5'):
                data = Poly5Reader(self.filepath_poly5_xdf)

                # Extract the samples and channel names from the Poly5Reader object
                self.samples = data.samples
                self.ch_names = data.ch_names
                self.sample_rate = data.sample_rate
                # Conversion to MNE raw array
            elif self.filepath_poly5_xdf.lower().endswith('xdf'):
                reader = Xdf_Reader(self.filepath_poly5_xdf)
                data = reader.data[0]
                
                self.samples = data.get_data()
                self.ch_names = data.ch_names
                self.sample_rate = data.info['sfreq']
                self.num_channels = len(self.ch_names)
                
            elif not self.filepath_poly5_xdf:
                tk.messagebox.showerror(title='No file selected', message = 'No data file selected.')

            else:
                tk.messagebox.showerror(title='Could not open file', message = 'File format not supported. Could not open file.')

        except:
            tk.messagebox.showerror(title='Could not open file', message = 'Something went wrong. Could not open file.')

        fsamp = int(self.sample_rate)
        channels = self.ch_names
        print(channels[1:-3])
        nchans = len(channels)
        ngrids = len(grid_names)
        # read in the EMG trial data
        emg_data = self.samples[1:-3,:]
        # create a dictionary containing all relevant signal parameters and data
        signal = dict(data = emg_data, fsamp = fsamp, nchans = nchans, ngrids = ngrids,grids = grid_names[:ngrids],muscles = muscle_names[:ngrids]) # discard the other muscle and grid entries, not relevant
       
        # if the signals were recorded with a feedback generated by ISpin, get the target and the path performed by the participant
        if self.ref_exist:
            target_ind = channels.index('Force Profile')
            target = self.samples[target_ind]
            path_ind = channels.index('AUX 1-2')
            path = self.samples[path_ind]
            signal['path'] = path
            signal['target'] = target
        self.signal_dict = signal
        self.decomp_dict = {} # initialising this dictionary here for later use
        self.dict = {} # initialising this dictionary here for later use
        return
        
    def grid_formatter(self):

        """ Match up the signals with the grid shape and numbering """

        grid_names = self.signal_dict['grids']
        self.signal_dict['filtered_data'] = np.zeros([np.shape(self.signal_dict['data'])[0],np.shape(self.signal_dict['data'])[1]]) # Filtered data should have same shape
        # Initialize maps as lists, as more grids may be used for the input 
        ## TODO: test using >1 grids! 
        c_map = [] # initializing amount of columns  
        r_map = [] # initializing amount of rows

        for i in range(self.signal_dict['ngrids']):
            if grid_names[i] == '4-8-L':
                ElChannelMap = [[7, 13, 18, 24], 
                                [6, 14, 17, 25], 
                                [5, 15, 16, 26], 
                                [4, 12, 19, 27], 
                                [3, 11, 20, 28], 
                                [2, 10, 21, 29], 
                                [1, 9, 22, 30], 
                                [0, 8, 23, 31]]
                self.rejected_chan = convert_channel_names_to_indices(self.rejected_chan, ElChannelMap)

                rejected_channels = np.zeros([self.signal_dict['ngrids'],32])
                rejected_channels[i, self.rejected_chan] = 1
                IED = 8.75
                self.emg_type = 0 # surface HD EMG 
                
                
            elif grid_names[i] == '8-8-L':
                ElChannelMap = [[16, 21, 26, 31, 32, 37, 42, 47],
                                [15, 20, 25, 30, 33, 38, 43, 48],
                                [14, 19, 24, 29, 34, 39, 44, 49],
                                [13, 18, 23, 28, 35, 40, 45, 50],
                                [12, 17, 22, 27, 36, 41, 46, 51],
                                [8, 9, 10, 11, 52, 53, 54, 55],
                                [4, 5, 6, 7, 56, 57, 58, 59],
                                [0, 1, 2, 3, 60, 61, 62, 63]]
                self.rejected_chan = convert_channel_names_to_indices(self.rejected_chan, ElChannelMap)

                rejected_channels = np.zeros([self.signal_dict['ngrids'],64])
                rejected_channels[i, self.rejected_chan] = 1
                IED = 8.75
                self.emg_type = 0 # surface HD EMG 
                                 
            elif grid_names[i] == 'GR04MM1305':
                ElChannelMap = [[0, 24, 25, 50, 51], 
                        [0, 23, 26, 49, 52], 
                        [1, 22, 27, 48, 53], 
                        [2, 21, 28, 47, 54], 
                        [3, 20, 29, 46, 55], 
                        [4, 19, 30, 45, 56], 
                        [5, 18, 31, 44, 57], 
                        [6, 17, 32, 43, 58],  
                        [7, 16, 33, 42, 59], 
                        [8, 15, 34, 41, 60],  
                        [9, 14, 35, 40, 61], 
                        [10, 13, 36, 39, 62], 
                        [11, 12, 37, 38, 63]]
                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 4
                self.emg_type = 0

            elif grid_names[i] == 'ELSCH064NM2':
                ElChannelMap = [[0, 0, 1, 2, 3],
                        [15, 7, 6, 5, 4],
                        [14, 13, 12, 11, 10],
                        [18, 17, 16, 8, 9],
                        [19, 20, 21, 22, 23],
                        [27, 28, 29, 30, 31],
                        [24, 25, 26, 32, 33],
                        [34, 35, 36, 37, 38],
                        [44, 45, 46, 47, 39],
                        [43, 42, 41, 40, 38],
                        [53, 52, 51, 50, 49],
                        [54, 55, 63, 62, 61],
                        [56, 57, 58, 59, 60]]

                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 8
                self.emg_type = 0

            elif grid_names[i] == 'GR08MM1305':
                ElChannelMap = [[0, 24, 25, 50, 51], 
                    [0, 23, 26, 49, 52], 
                    [1, 22, 27, 48, 53], 
                    [2, 21, 28, 47, 54], 
                    [3, 20, 29, 46, 55], 
                    [4, 19, 30, 45, 56], 
                    [5, 18, 31, 44, 57], 
                    [6, 17, 32, 43, 58],  
                    [7, 16, 33, 42, 59], 
                    [8, 15, 34, 41, 60],  
                    [9, 14, 35, 40, 61], 
                    [10, 13, 36, 39, 62], 
                    [11, 12, 37, 38, 63]]
                
                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 8
                self.emg_type = 0

            elif grid_names[i] == 'GR10MM0808':
                ElChannelMap = [[7, 15, 23, 31, 39, 47, 55, 63],
                        [6, 14, 22, 30, 38, 46, 54, 62],
                        [5, 13, 21, 29, 37, 45, 53, 61],
                        [4, 12, 20, 28, 36, 44, 52, 60],
                        [3, 11, 19, 27, 35, 43, 51, 59],
                        [2, 10, 18, 26, 34, 42, 50, 58],
                        [1, 9, 17, 25, 33, 41, 49, 57],
                        [0, 8, 16, 24, 32, 40, 48, 56]]

                rejected_channels = np.zeros([self.signal_dict['ngrids'],65])
                IED = 10
                self.emg_type = 0

            elif grid_names[i] == 'intraarrays':
                ElChannelMap = [[0, 10, 20, 30],
                        [1, 11, 21, 31],
                        [2, 12, 22, 32],
                        [3, 13, 23, 33],
                        [4, 14, 24, 34],
                        [5, 15, 25, 35],
                        [6, 16, 26, 36],
                        [7, 17, 27, 37],
                        [8, 18, 28, 38],
                        [9, 19, 29, 39]]

                rejected_channels = np.zeros([self.signal_dict['ngrids'],40])
                IED = 1
                self.emg_type = 1
            else:
                raise Exception("\nGrid name not recognised\n")
            
            ElChannelMap = np.array(ElChannelMap) # convert list to np array
            chans_per_grid = (np.shape(ElChannelMap)[0] * np.shape(ElChannelMap)[1])
            coordinates = np.zeros([chans_per_grid,2]) # find where the channels where in the grid 
            
            # find indices of elements, converting them back into tuples and iterate for all channels 
            flattened_map = np.array(ElChannelMap).flatten() # flatten 2D to 1D array
            coordinates = [np.unravel_index(np.where(flattened_map == value)[0][0], np.shape(ElChannelMap)) for value in range(chans_per_grid)]

            c_map.append(np.shape(ElChannelMap)[1]) # amount of columns 
            r_map.append(np.shape(ElChannelMap)[0]) # amount of rows 
            
            grid = i + 1 # move on to next grid 
            """
            filtering, not necessary (Matlab: only for visualisation of the signals)
            self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:] = notch_filter(self.signal_dict['data'][chans_per_grid*(grid-1):grid*chans_per_grid,:],self.signal_dict['fsamp'])
            self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:] = bandpass_filter(self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:],self.signal_dict['fsamp'],emg_type = self.emg_type)        
            print('Signal filtering complete')
            """
        self.c_maps = c_map
        self.r_maps = r_map
        self.rejected_channels = rejected_channels
        self.ied = IED
        self.coordinates = coordinates
       
    def manual_rejection(self):
        # NOTE: DID NOT CHECK THIS FUNCTION, TO DO: TEST/ADJUST FUNCTION
        """ Manual rejection for channels with noise/artificats by inspecting plots of the grid channels """

        for i in range(self.signal_dict['ngrids']):

            grid = i + 1
            chans_per_grid = (self.r_maps[i] * self.c_maps[i]) - 1
            sig2inspect = self.signal_dict['filtered_data'][chans_per_grid*(grid-1):grid*chans_per_grid,:]

            for c in range(self.c_maps[i]):
                for r in range(self.r_maps[i]):

                    num_chans2reject = []
                    if (c+r) > 0: # TO-DO: remove the assumption of the left corner channel being invalid
                        plt.plot(sig2inspect[(c*self.r_maps[i])+r-1,:]/max(sig2inspect[(c*self.r_maps[i])+r-1,:])+r+1)
                plt.show()
                
                inputchannels = simpledialog.askstring(title="Channel Rejection",
                                  prompt="Please enter channel numbers to be rejected (1-13), input with spaces between numbers:")
                print("The selected channels for rejection are:", inputchannels)
                
                if inputchannels:
                    str_chans2reject = inputchannels.split(" ")
                    for j in range(len(str_chans2reject)):

                        num_chans2reject.append(int(str_chans2reject[j])+c*self.r_maps[i]-1)

                    self.rejected_channels[i,num_chans2reject] =  1
        
        # TO DO: remove the assumption of the top LHC channel needing to be rejected
        self.rejected_channels = self.rejected_channels[:,1:] # get rid of the irrelevant top LHC channel
      
    def batch_w_target(self): 
        """use the EMG signal where the target value is higher than the threshold """
        plateau = np.where(self.signal_dict['target'] >= max(self.signal_dict['target']) * self.plat_thr)[0] # find indices of the plateau threshold
        discontinuity = np.where(np.diff(plateau) > 1)[0] # check if the plateau is reached in succeeding indices 
        if self.windows > 1 and not discontinuity: 
        
            plat_len = plateau[-1] - plateau[0] # [-1]: last indix where plateau is reached
            wind_len = np.floor(plat_len/self.windows) # give the length of the different windows 
            batch = np.zeros(self.windows*2) #
            for i in range(self.windows):

                batch[i*2] = plateau[0] + i*wind_len + 1
                batch [(i+1)*2-1] = plateau[0] + (i+1)*wind_len #different than matlab

            self.plateau_coords = batch
            
        elif self.windows >= 1 and discontinuity: #if discontinuity is not empty -> matlab r. 19
            #difference with matlab: self.windows > 1 
            prebatch = np.zeros([len(discontinuity)+1,2]) 
            
            prebatch[0,:] = [plateau[0],plateau[discontinuity[0]]] #prebatch: start of plateau and point where it changes 
            n = len(discontinuity)
            for i, d in enumerate(discontinuity): # index i, element d 
                if i < n - 1:
                    prebatch[i+1,:] = [plateau[d+1],plateau[discontinuity[i+1]]] # between 2 elements of plateau
                else:
                    prebatch[i+1,:] = [plateau[d+1],plateau[-1]] # for de last prebatch 

            plat_len = prebatch[:,-1] - prebatch[:,0] # difference between last and first column of each row 
            wind_len = np.floor(plat_len/self.windows)
            batch = np.zeros([len(discontinuity)+1,self.windows*2])
            print(range(self.windows))
            for i in range(self.windows):
                
                batch[:,i*2] = prebatch[:,0] + i*wind_len +1
                batch[:,(i+1)*2-1] = prebatch[:,0] + (i+1)*wind_len

            batch = np.sort(batch.reshape([1, np.shape(batch)[0]*np.shape(batch)[1]]))
            self.plateau_coords = batch
            
        else:
            # the last option is having only one window and no discontinuity in the plateau; in that case, you leave as is
            batch = [plateau[0],plateau[-1]] 
            print('plateau coordinates', plateau[0], plateau[-1])
            self.plateau_coords = batch
        
        # with the markers for windows and plateau discontinuities, batch the emg data ready for decomposition
        tracker = 0
        n_intervals = (int(len(self.plateau_coords)/2))
        batched_data = [None] * (self.signal_dict['ngrids'] * n_intervals)

        for i in range(int(self.signal_dict['ngrids'])):
            
            grid = i + 1
            chans_per_grid = (self.r_maps[i] * self.c_maps[i]) #aangepast
            
            for interval in range(n_intervals):
                #the data slice is the slice of 1 grid, only where the threshold of the target is reached 
                data_slice = self.signal_dict['data'][chans_per_grid*(grid-1):grid*chans_per_grid, int(self.plateau_coords[interval*2]):int(self.plateau_coords[(interval+1)*2-1])+1]
                rejected_channels_slice = self.rejected_channels[i,:] == 1
                # Remove rejected channels
                print(data_slice.shape)
                print(rejected_channels_slice.shape)
                batched_data[tracker] = np.delete(data_slice, rejected_channels_slice, 0)
                tracker += 1

        self.signal_dict['batched_data'] = batched_data
        self.chans_per_grid = chans_per_grid    

    def batch_wo_target(self):
        """segment the EMG signals without a set force profile, but base it on the EMG amplitude"""
        # NOTE: DID NOT CHECK THIS FUNCTION 
        # TODO: check this function
        shape = np.shape(self.signal_dict['data'])
        fake_ref = np.zeros(shape[1])
        
        # only looking at the first half of all EMG channel data
        half_length = int(np.floor(shape[0] / 2))
        tmp = np.zeros((half_length, shape[1]))
        for i in range(np.shape(tmp)[0]):
            tmp[i,:] =  moving_mean1d(abs(self.signal_dict['data'][i,:]),self.signal_dict['fsamp'])
        
        fake_ref = np.mean(tmp,axis=0)
        self.signal_dict['path'] = fake_ref
        self.signal_dict['target'] = fake_ref

        plt.figure(figsize=(10,8))
        plt.plot(tmp.T, color = [0.5,0.5,0.5], linewidth = 0.5)
        plt.plot(fake_ref,'k',linewidth=1)
        plt.ylim([0, max(fake_ref)*1.5])
        plt.title('EMG amplitude for first half of channel data')
        plt.grid()
        window_clicks = plt.ginput(2*self.windows, show_clicks = True) # TO-DO: is the output of plt.ginput 1d?
        plt.show()
                
        self.plateau_coords = np.zeros([1,self.windows *2])
        chans_per_grid = (self.r_maps[0] * self.c_maps[0])
        self.chans_per_grid = chans_per_grid
        tracker = 0
        n_intervals = (int(len(self.plateau_coords)/2))
        batched_data = [None] * (self.signal_dict['ngrids'] * n_intervals) 
        
        for interval in range(self.windows):
                   
            self.plateau_coords[0,interval*2] = np.floor(window_clicks[interval*2][0])
            self.plateau_coords[0,(interval+1)*2-1] = np.floor(window_clicks[(interval+1)*2-1][0])
        
        for i in range(int(self.signal_dict['ngrids'])):

            grid = i + 1
            for interval in range(n_intervals):
                
                data_slice = self.signal_dict['data'][chans_per_grid*(grid-1):grid*chans_per_grid, int(self.plateau_coords[interval*2]):int(self.plateau_coords[(interval+1)*2-1])+1]
                rejected_channels_slice = self.rejected_channels[i,:] == 1
                batched_data[tracker] = np.delete(data_slice, rejected_channels_slice, 0)
                tracker += 1
        
        self.signal_dict['batched_data'] = batched_data
        print(batched_data)

  
################################ CONVOLUTIVE SPHERING ########################################
    def convul_sphering(self,g,interval,tracker):

        """ 1) Filter the batched EMG data 2) Extend to improve speed of convergence/reduce numerical instability 3) Remove any DC component  4) Whiten """
        chans_per_grid = self.chans_per_grid
        grid = g+1
        if self.to_filter: # adding since will need to avoid this step if doing real-time decomposition + biofeedback, rond ergens af?
            #self.signal_dict['batched_data'][tracker] = notch_filter(self.signal_dict['batched_data'][tracker],self.signal_dict['fsamp'])
            self.signal_dict['batched_data'][tracker] = bandpass_filter(self.signal_dict['batched_data'][tracker],self.signal_dict['fsamp'],emg_type = self.emg_type)  
            pass

        # differentiation - typical EMG generation model treats low amplitude spikes/MUs as noise, which is common across channels so can be cancelled with a first order difference. Useful for high intensities - where cross talk has biggest impact.
        if self.differential_mode:
            # TODO: not checked this mode!
            to_shift = self.signal_dict['batched_data'][tracker]
            self.signal_dict['batched_data'][tracker]= []
            self.signal_dict['batched_data'][tracker]= np.diff(self.signal_dict['batched_data'][tracker],n=1,axis=-1)

        # signal extension - increasing the number of channels to 1000
        # Holobar 2007 -  Multichannel Blind Source Separation using Convolutive Kernel Compensation (describes matrix extension)
        extension_factor = int(np.round(self.ext_factor/len(self.signal_dict['batched_data'][tracker])))
        self.signal_dict['extend_obvs_old'][interval] = extend_emg(self.signal_dict['extend_obvs_old'][interval], self.signal_dict['batched_data'][tracker], extension_factor)
        self.signal_dict['sq_extend_obvs'][interval] = (self.signal_dict['extend_obvs_old'][interval] @ self.signal_dict['extend_obvs_old'][interval].T) / np.shape(self.signal_dict['extend_obvs_old'][interval])[1]
        self.signal_dict['inv_extend_obvs'][interval] = np.linalg.pinv(self.signal_dict['sq_extend_obvs'][interval]) # different method of pinv in MATLAB --> SVD vs QR
        
        # de-mean the extended emg observation matrix
        self.signal_dict['extend_obvs_old'][interval] = detrend(self.signal_dict['extend_obvs_old'][interval], axis=- 1, type='constant', bp=0)
        
        # whiten the signal + impose whitened extended observation matrix has a covariance matrix equal to the identity for time lag zero
        self.decomp_dict['whitened_obvs_old'][interval],self.decomp_dict['whiten_mat'][interval], self.decomp_dict['dewhiten_mat'][interval] = whiten_emg(self.signal_dict['extend_obvs_old'][interval])
        
        # remove the edges
        self.signal_dict['extend_obvs'][interval] = self.signal_dict['extend_obvs_old'][interval][:,int(np.round(self.signal_dict['fsamp']*self.edges2remove)-1):-int(np.round(self.signal_dict['fsamp']*self.edges2remove))]
        self.decomp_dict['whitened_obvs'][interval] = self.decomp_dict['whitened_obvs_old'][interval][:,int(np.round(self.signal_dict['fsamp']*self.edges2remove)-1):-int(np.round(self.signal_dict['fsamp']*self.edges2remove))]
        
        if g == 0: # don't need to repeat for every grid, since the path and target info (informing the batches), is the same for all grids
            """find the new plateau coordinates, when the edges are removed"""
            self.plateau_coords[interval*2] = self.plateau_coords[interval*2]  + int(np.round(self.signal_dict['fsamp']*self.edges2remove)) - 1
            self.plateau_coords[(interval+1)*2 - 1] = self.plateau_coords[(interval+1)*2-1]  - int(np.round(self.signal_dict['fsamp']*self.edges2remove))

        print('Signal extension and whitening complete')
######################### FAST ICA AND CONVOLUTIVE KERNEL COMPENSATION  ############################################

    def fast_ICA_and_CKC(self,g,interval,tracker,cf_type = 'skew',ortho_type = 'ord_deflation'):
        """Fast ICA and source improvement step"""
        
        init_its = np.zeros([self.its],dtype=int) # tracker of initialisations of separation vectors across iterations
        fpa_its = 500 # maximum number of iterations for the fixed point algorithm (optimalization of the weight vector)
        
        # identify the time instant at which the maximum of the squared summation of all whitened extended observation vectors
        # occurs. Then, projection vector is initialised to the whitened observation vector, at this located time instant.
        Z = np.array(self.decomp_dict['whitened_obvs'][interval]).copy()
        sort_sq_sum_Z = np.argsort(np.square(np.sum(Z, axis = 0))) # sort the activity indices (in time)
        time_axis = np.linspace(0,np.shape(Z)[1],np.shape(Z)[1])/self.signal_dict['fsamp']  # create a time axis for spiking activity

        # choosing contrast function here, avoid repetitively choosing within the iteration loop
        if cf_type == 'skew':
            cf = skew
            dot_cf = dot_skew
        elif cf_type == 'kurt':
            cf = kurt
            dot_cf = dot_kurt
        elif cf_type == 'exp':
            cf = exp
            dot_cf = dot_exp
        elif cf_type == 'logcosh':
            cf = logcosh
            dot_cf = dot_logcosh
      
        # each iteration, a new weight vector is estimated, create placeholder for this and the corresponding CoVISIs
        temp_MU_filters = np.zeros([np.shape(self.decomp_dict['whitened_obvs'][interval])[0],self.its])
        temp_CoVs = np.zeros([self.its])

        for i in range(self.its):
                #################### FIXED POINT ALGORITHM #################################
                init_its[i] = sort_sq_sum_Z[-(i+1)] # since the indexing starts at -1 the other way (for ascending order list)
                self.decomp_dict['w_sep_vect'] = Z[:,int(init_its[i])].copy() # retrieve the corresponding signal value to initialise the separation vector
                
                # orthogonalise separation vector before fixed point algorithm
                if ortho_type == 'ord_deflation':
                    self.decomp_dict['w_sep_vect'] -= np.dot(self.decomp_dict['B_sep_mat'] @ self.decomp_dict['B_sep_mat'].T, self.decomp_dict['w_sep_vect'])
                elif ortho_type == 'gram_schmidt':
                    #TO DO: did not check this orthogonalisation step 
                    self.decomp_dict['w_sep_vect'] = ortho_gram_schmidt(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'])
             
                # normalise separation vector before fixed point algorithm 
                self.decomp_dict['w_sep_vect'] /= np.linalg.norm(self.decomp_dict['w_sep_vect'])
            
                # use the fixed point algorithm to identify consecutive separation vectors
                self.decomp_dict['w_sep_vect'] = fixed_point_alg(self.decomp_dict['w_sep_vect'],self.decomp_dict['B_sep_mat'],Z, cf, dot_cf,fpa_its,ortho_type)
                
                # get the first iteration of spikes using k means ++
                fICA_source, spikes = get_spikes(self.decomp_dict['w_sep_vect'],Z, self.signal_dict['fsamp'])
            
                ################# MINIMISATION OF COV OF DISCHARGES ############################
                if len(spikes) > 10:

                    # determine the interspike interval
                    ISI = np.diff(spikes/self.signal_dict['fsamp'])
                    # determine the coefficient of variation
                    temp_CoVs[i] = np.std(ISI, ddof=1)/np.mean(ISI) #use the unbiased std to calculate the CoVISI
                    
                    # update the sepearation vector by summing all the spikes
                    w_n_p1 = np.sum(Z[:,spikes],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
                    # minimisation of covariance of interspike intervals
                    temp_MU_filters[:,i], temp_CoVs[i] = min_cov_isi(w_n_p1, Z, self.signal_dict['fsamp'],temp_CoVs[i])
                
                    # store the MU filters as columns in the B matrix 
                    self.decomp_dict['B_sep_mat'][:,i] = (self.decomp_dict['w_sep_vect']).real # no need to shallow copy here

                    # calculate SIL
                    fICA_source, spikes, self.decomp_dict['SILs'][interval,i] = get_silohuette(temp_MU_filters[:,i],Z,self.signal_dict['fsamp'])
                    
                    # peel off the found spike train from the EMG signal 
                    # TODO: check this 
                    if self.peel_off == 1 and self.decomp_dict['SILs'][interval,i] > self.sil_thr:
                        Z = peel_off(Z, spikes, self.signal_dict['fsamp'])

                    if self.drawing_mode == 1:
                        plt.clf()
                        plt.ion()
                        plt.show()
                        plt.subplot(2, 1, 1)
                        plt.plot(self.signal_dict['target'], 'k--', linewidth=2)
                        plt.plot([self.plateau_coords[interval*2], self.plateau_coords[interval*2]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.plot([self.plateau_coords[(interval+1)*2 - 1], self.plateau_coords[(interval+1)*2 - 1]], [0, max(self.signal_dict['target'])], color='r', linewidth=2)
                        plt.title('Grid #{} - Iteration #{} - Sil = {}'.format(g, i+1, self.decomp_dict['SILs'][interval,i]))
                        plt.subplot(2, 1, 2)
                        plt.plot(time_axis, fICA_source,linewidth = 0.5)
                        plt.plot(time_axis[spikes],fICA_source[spikes],'o')
                        plt.grid()
                        plt.draw()
                        plt.pause(1e-6)
                    else:
                        print('Grid #{} - Iteration #{} - Sil = {}'.format(g, i, self.decomp_dict['SILs'][interval,i]))

                else:
                    print('Grid #{} - Iteration #{} - less than 10 spikes '.format(g, i))
                    # without enough spikes, we skip minimising the covariation of discharges to improve the separation vector
                    self.decomp_dict['B_sep_mat'][:,i] = self.decomp_dict['w_sep_vect'].real  # no need to shallow copy here

                print('Grid #{} - Iteration #{} - Sil = {}'.format(g, i, self.decomp_dict['SILs'][interval,i]))
                
        # keep the MU filters that had associated SIL values equal or greater than the imposed SIL threshold
        temp_MU_filters = temp_MU_filters[:,self.decomp_dict['SILs'][interval,:] >= self.sil_thr] 
        
        # if the CoV must be higher than a certain value, select the corresponding MU filter 
        if self.cov_filter:
            temp_CoVs = temp_CoVs[self.decomp_dict['SILs'][interval,:] >= self.sil_thr] 
            temp_MU_filters = temp_MU_filters[:,temp_CoVs <= self.cov_thr]
        
        self.decomp_dict['MU_filters'][interval] = temp_MU_filters
        self.decomp_dict['CoVs'][interval] = temp_CoVs

        temp_MU_filters = None
        temp_CoVs = None

######################################## POST PROCESSING #####################################################

    def post_process_EMG(self,g,tracker):
        '''
        # batch processing over each window
        # remove duplicates 
        # remove outliers generating irrelevant discharge rates (1st time)
        # reevaluate all the unique motor units over the contractions 
        # remove outliers generating irrelevant discharge rates (2nd time)
        '''
        
        # batch processing over each window
        extension_factor = int(np.round(self.ext_factor/len(self.signal_dict['batched_data'][tracker])))
        pulse_trains, discharge_times = batch_process_filters(self.decomp_dict['MU_filters'], self.decomp_dict['whitened_obvs'], self.plateau_coords, extension_factor, self.differential_mode,np.size(self.signal_dict['data'][1]),self.signal_dict['fsamp'])
        
        # realign the discharge times with the centre of the MUAP
        if self.alignMUAP:
            pass 
        else: 
             discharge_times_aligned = discharge_times.copy()

        # ONLY FOR 1 GRID! TODO: MAKE FOR MORE GRIDS!
        if np.shape(pulse_trains)[0] > 0: # when there are pulse trains found 
            # maxlag = round(fsamp)/40
            # jitter = 0.0025
            pulse_trains, discharge_times_new, MU_filters_new = remove_duplicates(self.decomp_dict['MU_filters'], pulse_trains, discharge_times, discharge_times_aligned, round(self.signal_dict['fsamp']/40), 0.00025, self.signal_dict['fsamp'], self.dup_thr)
            self.decomp_dict['MU_filters'][0] = MU_filters_new

            # if we want further automatic refinement of MUs, prior to manual edition
            if self.refineMU: 
                # TODO: not checked this function yet

                # Remove outliers generating irrelevant discharge rates before manual edition (1st time)
                discharge_times_new = remove_outliers(pulse_trains, discharge_times_new, self.signal_dict['fsamp'])
                
                # Re-evaluate all of the UNIQUE MUs over the contraction
                # TODO: adjust to have adding of length(signal.EMGmask{i}), :) --> generalises to cases where the upper left electrode is not excluded
                self.decomp_dict['pulse_trains'][g], discharge_times_new = refine_mus(self.signal_dict['data'][self.chans_per_grid*(g):self.chans_per_grid*(g) + len(self.rejected_channels[g]),:], self.rejected_channels[g], pulse_trains, discharge_times_new, self.signal_dict['fsamp'])
                
                # Remove outliers generating irrelevant discharge rates before manual edition (2nd time)
                discharge_times_new = remove_outliers(pulse_trains, discharge_times_new, self.CoVDR, self.signal_dict['fsamp'])
            else:
                self.decomp_dict['pulse_trains'][g] = pulse_trains #make placeholder for it? 

            # generate binary spike trains (only containing 0/1) using the discharge times and the length of the data 
            binary_spike_trains = get_binary_pulse_trains(discharge_times_new, np.size(self.signal_dict['data'][1]))
            
            self.decomp_dict['discharge_times'][g] = discharge_times_new
            self.decomp_dict['SILs'] = [None] * np.shape(self.decomp_dict['pulse_trains'][g])[0] #placeholder 
            self.dict['BINARY_MUS_FIRING'] = binary_spike_trains
            self.discharge_times = discharge_times_new
            self.mu_filters = MU_filters_new
            
            Z = np.array(self.decomp_dict['whitened_obvs'][0]).copy()
            
            # store the final SILs for analysis in OpenHDEMG
            for i in range((np.shape(self.decomp_dict['MU_filters'][0]))[1]):
                # calculate the SIL from the MUfilter and the whitened data 
                _, _, self.decomp_dict['SILs'][i] = get_silohuette(self.decomp_dict['MU_filters'][0][:, i],Z,self.signal_dict['fsamp']) # did not take into account multiple intervals
        print('Post-processing complete')

        
######################################## SAVING DATA #####################################################

    def save_EMG_decomposition(self, g,tracker): # TODO: Check whether g or tracker? 
        # adapted from OpenHDEMG, function save_to_json() 
        self.file_path_json = os.path.join(self.savefolder, self.filename +'_decomp.json')
        self.dict["SOURCE"] = "CUSTOMCSV"
        self.dict["FILENAME"] = "training40" 
        raw_emg = sort_raw_emg(self.signal_dict['data'],  self.signal_dict['grids'][0], self.signal_dict['fsamp'], self.emg_type)
        self.dict["RAW_SIGNAL"] = pd.DataFrame(raw_emg).T
        self.dict["REF_SIGNAL"] = pd.DataFrame(self.signal_dict['path'])
        self.dict["ACCURACY"] = pd.DataFrame(self.decomp_dict['SILs'])
        self.dict["IPTS"] = pd.DataFrame(self.decomp_dict['pulse_trains'][0]).T
        self.dict["MUPULSES"] = [np.array(item) for item in self.discharge_times]
        self.dict["FSAMP"] = float(self.signal_dict['fsamp'])
        self.dict["IED"] = float(self.ied)
        self.dict["BINARY_MUS_FIRING"] = pd.DataFrame(self.dict['BINARY_MUS_FIRING']).T
        self.dict["EMG_LENGTH"] = self.signal_dict['data'].shape[1]
        self.dict["NUMBER_OF_MUS"] = np.shape(self.decomp_dict['pulse_trains'][0])[0]
        self.dict["EXTRAS"] = pd.DataFrame(columns=[0])
        
        
        
        # based on the function 
        source = json.dumps(self.dict["SOURCE"])
        filename = json.dumps(self.dict["FILENAME"])
        fsamp = json.dumps(self.dict["FSAMP"])
        ied = json.dumps(self.dict["IED"] )
        emg_length = json.dumps(self.dict["EMG_LENGTH"])
        number_of_mus = json.dumps(self.dict["NUMBER_OF_MUS"])

        # Access and convert the dict to a json object.
        # orient='split' is fundamental for performance.
        raw_signal = self.dict["RAW_SIGNAL"].to_json(orient='split')
        ref_signal = self.dict["REF_SIGNAL"].to_json(orient='split')
        accuracy = self.dict["ACCURACY"].to_json(orient='split')
        ipts = self.dict["IPTS"].to_json(orient='split')
        binary_mus_firing = self.dict["BINARY_MUS_FIRING"].to_json(orient='split')
        extras = self.dict["EXTRAS"].to_json(orient='split')

        # Every array has to be converted in a list; then, the list of lists
        # can be converted to json.
        mupulses = []
        for ind, array in enumerate(self.dict["MUPULSES"]):
            mupulses.insert(ind, array.tolist())
        mupulses = json.dumps(mupulses)

        # Convert a dict of json objects to json. The result of the conversion
        # will be saved as the final json file.
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

        # Compress and write the json file
        with gzip.open(
            self.file_path_json,
            "wt",
            encoding="utf-8",
            compresslevel=4
        ) as f:
            json.dump(emgfile, f)

