from EMG_classes import offline_EMG
import os
import numpy as np


class EMGDecomposition:
    """
    A class to handle the EMG decomposition process.

    This class performs various steps of EMG data decomposition, including 
    convolutional sphering, Independent Component Analysis (ICA), and post-processing 
    of the EMG signals. It manages file selection, data formatting, and the overall 
    workflow of EMG analysis.

    Attributes:
        filepath (str): The path to the input file.
        rejected_chan (list): List of rejected channels.
        emg_obj (offline_EMG): Instance of the offline_EMG class to handle EMG data processing.
        file (str): The path to the current file being processed.
    """

    def __init__(self, filepath, rejected_chan=None):
        """
        Initialize the EMGDecomposition class.

        Args:
            filepath (str): Path to the EMG file.
            rejected_chan (list, optional): Channels to be rejected from analysis. Defaults to an empty list.
        """
        if rejected_chan is None:
            rejected_chan = []

        # Set location to absolute path of the directory containing current script file
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )  # Get current working directory, gets directory name of current script file

        self.rejected_chan = rejected_chan
        print('Initializing offline EMG...')
        self.emg_obj = offline_EMG(0, rejected_chan=self.rejected_chan)  # 0/1 filter signal
        self.file = filepath

    def run(self, grid_name='4-8-L'):
        """
        Run the decomposition process for the EMG file.

        This function selects the EMG file, formats the grid data, batches the EMG data with or without the target, 
        and performs several steps of EMG decomposition including convolutional sphering, ICA, and post-processing.

        Args:
            grid_name (str, optional): The name of the grid to be used in the decomposition. Defaults to '4-8-L'.
        """
        # File organization and selection
        self.emg_obj.select_file(self.file)  # Select the training file (.mat), composed by ISpin
        self.emg_obj.convert_poly5_xdf(grid_names=[grid_name], muscle_names=['TA'])  # Adds signal_dict to the emg_obj, using Matlab output of ISpin
        print('Data loaded')
        self.emg_obj.grid_formatter()  # Adds spatial context

        # Batching the data
        if self.emg_obj.ref_exist:  # If target path is available, batch the EMG signal to isolate force plateau
            self.emg_obj.batch_w_target()
        else:
            # Not yet checked
            self.emg_obj.batch_wo_target()  # If no target path, batch without it

        # Create placeholders for decomposition
        self.emg_obj.signal_dict['diff_data'] = []  # Placeholder for the differential data
        tracker = 0  # Tracker corresponds to the grid number
        nwins = int(len(self.emg_obj.plateau_coords) / 2)  # Number of force profiles to decompose
        print(f'nwins = {nwins}')

        # Loop through grids
        for g in range(int(self.emg_obj.signal_dict['ngrids'])):
            extension_factor = int(np.round(
                self.emg_obj.ext_factor / np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0]
            ))  # Calculate extension factor using #EMG channels

            # Arrays holding extended EMG data PRIOR to removal of edges
            self.emg_obj.signal_dict['extend_obvs_old'] = np.zeros([
                nwins,
                np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0] * extension_factor,
                np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[1] + extension_factor - 1 - self.emg_obj.differential_mode
            ])
            self.emg_obj.decomp_dict['whitened_obvs_old'] = self.emg_obj.signal_dict['extend_obvs_old'].copy()

            # Arrays for square and inverse of extended EMG data
            self.emg_obj.signal_dict['sq_extend_obvs'] = np.zeros([
                nwins,
                np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0] * extension_factor,
                np.shape(self.emg_obj.signal_dict['batched_data'][tracker])[0] * extension_factor
            ])
            self.emg_obj.signal_dict['inv_extend_obvs'] = self.emg_obj.signal_dict['sq_extend_obvs'].copy()

            # Whitening and dewhitening matrices
            self.emg_obj.decomp_dict['dewhiten_mat'] = self.emg_obj.signal_dict['sq_extend_obvs'].copy()
            self.emg_obj.decomp_dict['whiten_mat'] = self.emg_obj.signal_dict['sq_extend_obvs'].copy()

            # Arrays for extended EMG data AFTER removal of edges
            start_idx = int(np.round(self.emg_obj.signal_dict['fsamp'] * self.emg_obj.edges2remove) - 1)
            end_idx = -int(np.round(self.emg_obj.signal_dict['fsamp'] * self.emg_obj.edges2remove))
            
            self.emg_obj.signal_dict['extend_obvs'] = (
                self.emg_obj.signal_dict['extend_obvs_old'][:, :, start_idx:end_idx]
            ).copy()
            self.emg_obj.decomp_dict['whitened_obvs'] = self.emg_obj.signal_dict['extend_obvs'].copy()

            for interval in range(nwins):
                # Initialize separation matrix B and vector w
                self.emg_obj.decomp_dict['B_sep_mat'] = np.zeros([
                    np.shape(self.emg_obj.decomp_dict['whitened_obvs'][interval])[0], self.emg_obj.its
                ])
                self.emg_obj.decomp_dict['w_sep_vect'] = np.zeros([
                    np.shape(self.emg_obj.decomp_dict['whitened_obvs'][interval])[0], 1
                ])

                # Initialize MU filters and CoVs
                self.emg_obj.decomp_dict['MU_filters'] = [None] * nwins
                self.emg_obj.decomp_dict['CoVs'] = [None] * nwins
                self.emg_obj.decomp_dict['SILs'] = np.zeros([nwins, self.emg_obj.its])

                # Convolutional Sphering
                print('Starting convolutional sphering...')
                self.emg_obj.convul_sphering(g, interval, tracker)#signal extension & whitening

                # Fast ICA
                print('Starting ICA...')
                self.emg_obj.fast_ICA_and_CKC(g, interval, tracker)  # Find weight vector using FPA and source improvement

            # Post-processing for each grid
            self.emg_obj.decomp_dict['pulse_trains'] = [None] * nwins
            self.emg_obj.decomp_dict['discharge_times'] = [None] * nwins
            self.emg_obj.decomp_dict['IPTS'] = [None] * nwins

            print('Post-processing...')
            self.emg_obj.post_process_EMG(g, tracker)  # g and tracker are unnecessary here
            tracker += 1  # Move to the next grid

        # Save results
        print('Saving data...')
        self.emg_obj.save_EMG_decomposition(g, tracker)  # g and tracker are unused in this function
        print('Data saved.')
