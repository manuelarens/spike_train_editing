from EMG_classes import offline_EMG, EMG
import glob, os
import numpy as np
import pandas as pd 

class EMG_Decomposition():
    def __init__(self, filepath, rejected_chan = []):
        # Set location to absolute path of directory containing current script file 
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))) # get current working directory, gets directory name of current script file
        self.rejected_chan = rejected_chan
        print('net voor start offline emg')
        self.emg_obj = offline_EMG(0, rejected_chan=self.rejected_chan) # 0/1 filter signal
        self.file = filepath

    def run(self, grid_name = '4-8-L'):
        ################## FILE ORGANISATION ################################

        self.emg_obj.select_file(self.file) #select the training file (.mat), composed by ISpin 
        self.emg_obj.convert_poly5_xdf(grid_names=[grid_name], muscle_names=['TA'])# adds signal_dict to the self.emg_obj, using Matlab output of ISpin 
        self.emg_obj.grid_formatter() # adds spatial context

        if self.emg_obj.check_emg: # if you want to check the signal quality, perform channel rejection
            # TODO: check, NOT CHECKED YET 
            self.emg_obj.manual_rejection()

        #################### BATCHING #######################################
        if self.emg_obj.ref_exist: # if you want to use the target path to segment the EMG signal, to isolate the force plateau
            self.emg_obj.batch_w_target()
        else:
            # TODO: check, NOT CHECKED YET 
            self.emg_obj.batch_wo_target() # if you don't have one, batch without the target path

        ################### CREATE PLACEHOLDERS #############################
        self.emg_obj.signal_dict['diff_data'] = [] #placeholder for the differential data 
        tracker = 0 # tracker corresponds the the grid number 
        nwins = int(len(self.emg_obj.plateau_coords)/2) # amount of force profiles to decompose
        for g in range(int(self.emg_obj.signal_dict['ngrids'])): 
                extension_factor = int(np.round(self.emg_obj.ext_factor/np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0])) # calculate extension factor using #EMGchannels 
                # these two arrays are holding extended emg data PRIOR to the removal of edges
                self.emg_obj.signal_dict['extend_obvs_old'] = np.zeros([nwins, np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor), np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[1] + extension_factor -1 - self.emg_obj.differential_mode ])
                self.emg_obj.decomp_dict['whitened_obvs_old'] = self.emg_obj.signal_dict['extend_obvs_old'].copy()
                # these two arrays are the square and inverse of extneded emg data PRIOR to the removal of edges        
                self.emg_obj.signal_dict['sq_extend_obvs'] = np.zeros([nwins,np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor),np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0]*(extension_factor)])
                self.emg_obj.signal_dict['inv_extend_obvs'] = self.emg_obj.signal_dict['sq_extend_obvs'].copy()
                # dewhitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
                self.emg_obj.decomp_dict['dewhiten_mat'] = self.emg_obj.signal_dict['sq_extend_obvs'].copy()
                # whitening matrix PRIOR to the removal of edges (no effect either way on matrix dimensions)
                self.emg_obj.decomp_dict['whiten_mat'] = self.emg_obj.signal_dict['sq_extend_obvs'].copy()
                # these two warrays are holding extended emg data AFTER the removal of edges
                self.emg_obj.signal_dict['extend_obvs'] = self.emg_obj.signal_dict['extend_obvs_old'][:,:,int(np.round(self.emg_obj.signal_dict['fsamp']*self.emg_obj.edges2remove)-1):-int(np.round(self.emg_obj.signal_dict['fsamp']*self.emg_obj.edges2remove))].copy()
                self.emg_obj.decomp_dict['whitened_obvs'] = self.emg_obj.signal_dict['extend_obvs'].copy()
                
                for interval in range (nwins): 
                    
                    # initialise zero arrays for separation matrix B and separation vectors w
                    self.emg_obj.decomp_dict['B_sep_mat'] = np.zeros([np.shape(self.emg_obj.decomp_dict['whitened_obvs'][interval])[0],self.emg_obj.its])
                    self.emg_obj.decomp_dict['w_sep_vect'] = np.zeros([np.shape(self.emg_obj.decomp_dict['whitened_obvs'][interval])[0],1])
                    # MU filters needs a pre-allocation with more flexibility, to delete parts later
                    self.emg_obj.decomp_dict['MU_filters'] = [None]*(nwins)
                    self.emg_obj.decomp_dict['CoVs'] = [None]*(nwins)
                    
                    self.emg_obj.decomp_dict['SILs'] = np.zeros([nwins,self.emg_obj.its]) 
                    
                #################### Convolutional Sphering ########################################
                    self.emg_obj.convul_sphering(g,interval,tracker) #signal extension & whitening
                    
                #################### FAST ICA ########################################
                    self.emg_obj.fast_ICA_and_CKC(g,interval,tracker) # find the weight vector using the FPA and source improvement
                    
        ##################### POSTPROCESSING #################################
                # still per grid 
                self.emg_obj.decomp_dict['pulse_trains'] = [None]*(nwins)
                self.emg_obj.decomp_dict['discharge_times'] = [None]*(nwins)
                self.emg_obj.decomp_dict['IPTS'] = [None]*(nwins)
                
                self.emg_obj.post_process_EMG(g, tracker) # g and tracker are unncessary! remove  
                tracker = tracker + 1 #to the next grid!! -> makes more sense to do that first and than post_process EMG 

        # save results     
        print('Saving data')

        # self.emg_obj.save_EMG_decomposition(g,tracker)
        self.emg_obj.save_EMG_decomposition(g,tracker, interval) #g and tracker are unused 
        print('Data saved')
