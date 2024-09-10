'''
(c) 2023 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file ${force_feedback_plotter_helper.py}
 * @brief Plotter helper for creating force feedback plotter for HD-EMG purposes.
 *  The plotter plots the desired force profile and actual normalized force input of lead cell
 *  based on mvc offset and value 
 */
''' 

import numpy as np 
from operator import itemgetter
from os.path import join, dirname, realpath
from datetime import datetime
import pandas as pd
import time

from TMSiFrontend.plotters.signal_plotter import SignalPlotter
from TMSiSDK.sample_data_server.sample_data import SampleData

from TMSiBackend.data_consumer.consumer import Consumer
from TMSiBackend.data_consumer.consumer_thread import ConsumerThread
from TMSiGui.gui import Gui
from Test_Scripts_Nathan.MVC_extraction import MVC_Extraction
from Test_Scripts_Nathan.poly5_force_file_writer import Poly5Writer
from Test_Scripts_Nathan.xdf_force_file_writer import XdfWriter
import matplotlib.pyplot as plt

from Online_EMG.online_emg_plotter import OnlineEMGPlotter
from TMSiPlotterHelpers.filtered_signal_plotter_helper import FilteredSignalPlotterHelper
from Online_EMG.EMG_classes import online_EMG

class OnlineEMGPlotterHelper(FilteredSignalPlotterHelper):
    def __init__(self, device, meas_type, filepath,  filename = 'Online_EMG+ForceProfile', 
                 file_type = 'poly5', force_channel = 'AUX 1-2', grid_type=None, hpf=0, lpf=0, order=1, rejected_channels = []):
        
        super().__init__(device=device, grid_type=grid_type, hpf=hpf, lpf=lpf, order=order)
        self.filepath = filepath
        self.main_plotter = SignalPlotter() 
        self.diff_pointer = 0
        self.total_pointer = 0 
        self.MVC_value = 1423437.222875
        self.MVC_offset = 1245847.233375
        self.meas_type = meas_type
        self.force_channel = force_channel
        self.plotter2 = OnlineEMGPlotter(meas_type = meas_type)
        self.save_pointer = [0]
        self.filename = filename
        self.file_type = file_type
        self.online_EMG = online_EMG(self.filepath, rejected_channels=rejected_channels)

    def callback(self, callback_object):
        """
        Gets the data from the Monitor instance and plots it in 2 different windows. 
        Main window contains the Online EMG plotter and secondary window has the normal signal plotter
        """
        response = callback_object["buffer"]
        pointer_data_to_plot = response.pointer_buffer
        data_to_plot = response.dataset
        # Wait untill data is coming in
        if data_to_plot is None:
            return
        size_dataset = np.shape(data_to_plot)[1]
        n_channels = np.shape(data_to_plot)[0]
        # # add whitening zone 
        # if pointer_data_to_plot != size_dataset:
        #     num_time_samples = self.sampling_frequency * self.main_plotter.window_size
        #     whitening_zone = int(self.whitening_zone * num_time_samples)
        #     space_to_fill = size_dataset - pointer_data_to_plot
        #     if space_to_fill < whitening_zone:
        #         data_to_plot[:, pointer_data_to_plot:] = \
        #             np.full((n_channels, space_to_fill), np.nan)
        #     else:
        #         data_to_plot[:, pointer_data_to_plot:pointer_data_to_plot+whitening_zone] = \
        #             np.full((n_channels, whitening_zone), np.nan)
        # # Send data to plotter and update chart
        # self.main_plotter.update_chart(data_to_plot = data_to_plot[self.channel_conversion_list], time_span=self.time_span)

        ## Implement a counter to update time axis in plotter 2
        self.save_pointer.append(int(pointer_data_to_plot))
        if self.main_plotter_refresh_counter >= 1: 
            #After the first iteration, a diff_pointer is calculated to see how many samples have been added to the buffer.
            if self.save_pointer[-1] > self.save_pointer[-2]: 
                self.diff_pointer = self.save_pointer[-1]-self.save_pointer[-2]
                self.total_pointer += self.diff_pointer
            else: #if the pointer_to_data_plot is lower than the previous saved_pointer
                self.diff_pointer = (self.save_pointer[-1] + 40000 - self.save_pointer[-2]) % 40000
                self.total_pointer += self.diff_pointer
        else:
            #  Always called at the first iteration, so here empty arrays are made. total pointer and diff pointer are 
            # the same at the first iteration
            self.total_pointer += pointer_data_to_plot
            self.diff_pointer = pointer_data_to_plot
            self.data_MU  = np.empty((np.shape(self.online_EMG.mu_filters)[1],  response.size_buffer))
            self.data_MU[:] = np.nan
            self.discharges = np.empty((np.shape(self.online_EMG.mu_filters)[1], len(self.force_profile)))
            self.disrate = np.empty((np.shape(self.online_EMG.mu_filters)[1], len(self.force_profile)))
        # Update plotter2 depending on refresh rate
        if self.main_plotter_refresh_counter % self.plotter2_refresh_rate == 0:
            if size_dataset < response.size_buffer / 2 :
                data_to_return = np.empty((n_channels, response.size_buffer))
                data_to_return[:] = np.nan
                data_to_return[:,:pointer_data_to_plot] = data_to_plot[:,:pointer_data_to_plot]
                data_EMG_Force = data_to_return
                # HERE THE EMG CHANNELS (R#C# CHANNELS) ARE PASSED INTO THE LIVE DECOMPOSITION ALGORITHM. IF ADDITIONAL CHANNELS AUX OR BIP CHANNELS OUTSIDE OF THE LOAD CELL ARE CONNECTED
                # THIS WILL CRASH. 
                DisTimesTemp = self.online_EMG.callback(data_EMG_Force[1:-3, self.save_pointer[-2]:self.save_pointer[-1]])
                self.discharges[:, self.save_pointer[-2]:self.save_pointer[-1]] = DisTimesTemp
                self.calculate_disrate(self.discharges)
                self.data_MU[:,self.save_pointer[-2]:self.save_pointer[-1]] = self.disrate[:, self.save_pointer[-2]:self.save_pointer[-1]]
                # append the right part of the force profile to the data 
                data_to_plotter2 = np.concatenate((data_to_return[self.force_channel_saving,:].reshape(1, -1), self.force_profile[np.arange(0,response.size_buffer)].reshape(1, -1)), axis=0)
                # self.data_MU[:, :pointer_data_to_plot] =  DisTimesTemp
                data_to_plotter2 = np.concatenate((data_to_plotter2, self.data_MU), axis=0)
            else:
                # Flip data such that the newest data is on the end
                data_to_return = np.ones_like(data_to_plot)
                data_to_return[:,size_dataset - pointer_data_to_plot:] = data_to_plot[:,:pointer_data_to_plot]
                data_to_return[:,:size_dataset - pointer_data_to_plot] = data_to_plot[:,pointer_data_to_plot:]
                half_index = response.size_buffer - int(len(self.running_time_span) / 2)
                # Select the right channels 
                data_to_return = np.concatenate([data_to_return[:, -half_index:], np.full((n_channels, half_index), np.nan)], axis=1)
                # HERE THE EMG CHANNELS (R#C# CHANNELS) ARE PASSED INTO THE LIVE DECOMPOSITION ALGORITHM. IF ADDITIONAL CHANNELS AUX OR BIP CHANNELS OUTSIDE OF THE LOAD CELL ARE CONNECTED
                # THIS WILL CRASH. 
                data_EMG_Force = data_to_return[1:-3,:]
                DisTimesTemp = self.online_EMG.callback(data_EMG_Force[:64, int(response.size_buffer/2)-self.diff_pointer:int(response.size_buffer/2)])
                self.discharges[:, self.total_pointer-self.diff_pointer:self.total_pointer] = DisTimesTemp
                self.calculate_disrate(self.discharges)
                self.data_MU[:,:int(response.size_buffer/2)] = self.disrate[:, self.total_pointer-5*self.sampling_frequency:self.total_pointer]
                # self.data_MU_temp = self.data_MU[:,:int(response.size_buffer/2)]
                # self.data_MU_temp = np.concatenate((self.data_MU_temp, DisTimesTemp), axis=1)
                # self.data_MU[:,:int(response.size_buffer/2)] = self.data_MU_temp[:,-int(response.size_buffer/2):]
                self.middlesecond = self.total_pointer/self.sampling_frequency
                # append the right part of the force profile to the data 
                profile_select = np.arange(self.total_pointer-half_index,  self.total_pointer+half_index)
                data_to_plotter2 = np.concatenate((data_to_return[self.force_channel_saving,:].reshape(1, -1), self.force_profile[profile_select].reshape(1, -1)), axis=0)
                data_to_plotter2 = np.concatenate((data_to_plotter2, self.data_MU), axis=0)

        
        if self.meas_type == 3: 
            Normalized_data_to_plot = (abs(data_to_plotter2[self.force_channel_ind]) - self.MVC_offset ) / (self.MVC_value - self.MVC_offset) * 100
            data_to_return[self.force_channel_saving,:] = Normalized_data_to_plot
            data_to_plotter2[self.force_channel_ind,:] = Normalized_data_to_plot
            # Ensure data is saved correctly
            if size_dataset < response.size_buffer / 2: 
                appended_force_profile = np.concatenate((data_to_return, self.force_profile[np.arange(0,response.size_buffer)].reshape(1, -1)), axis=0)
                data_to_save = appended_force_profile[:, self.save_pointer[-2]:self.save_pointer[-1]]
            else: 
                appended_force_profile = np.concatenate((data_to_return, self.force_profile[profile_select].reshape(1, -1)), axis=0)
                data_to_save = appended_force_profile[:, (int(response.size_buffer/2))-(self.diff_pointer):int(response.size_buffer/2)]
            self.put_sample_data(data=data_to_save)
        
        self.plotter2.buffer_size = response.size_buffer
        self.plotter2.update_chart(data_to_plot = data_to_plotter2[:,::5],  time_span=self.running_time_span[::5])
        self.plotter2.update_time_ticks(self.middlesecond-5, self.middlesecond+5)
        self.plotter2.time_elapsed = self.total_pointer / self.sampling_frequency
        self.plotter2.update_timer()
        self.main_plotter_refresh_counter += 1

    def initialize(self):
        """
        Initializes channel settings, where the force_profile channel is added. 
        Also extracts the names, unit names, and channel types of the channels. 
        Initializes an extra file writer to write away the data and uses CHANGED file writers.
        """
        super().initialize()
        # Set time span of secondary window to 10 s
        self.plotter2.chart.set_time_min_max(0, 10)
        self.running_time_span = np.arange(0,10,1.0/self.sampling_frequency)
        self.middlesecond = 5
        # Set refresh rate to update every 10 times (=1Hz) to have readible signals
        self.plotter2_refresh_rate = 0.5
        self.main_plotter_refresh_counter = 0

        #set channel settings
        self.online_EMG.preprocess_with_biofeedback()
        self.force_profile_name = 'Force Profile'
        self.channels_default =self.device.get_device_active_channels() 
        self._read_grid_info()    
        self._get_channel_conversion_list()
        self.channels_for_saving=[self.channels_default[idx] for idx in self.channel_conversion_list] 
         # Force channel is added to the list of channels
        self.channels_for_saving.append(self.force_profile_name) 
        
        # Extract channel components for the file saving
        self.channel_names_saving, self.channel_unit_names_saving, self.channel_types_saving = self.plotter2.get_channel_info(self.channels_for_saving)
        self.force_channel_saving= self.channel_names_saving.index(self.force_channel)
        self.channel_unit_names_saving[self.force_channel_saving] = 'a.u.'

        # Setup channels for plotting
        self.channels=[self.channels_default[idx] for idx in self.channel_conversion_list] # all the channels from the device
        self.channels_for_plotting = [self.channels[-3]] # Extract the AUX channel
        self.channels_for_plotting.append(self.force_profile_name)  # Add the force profile

        for i in range(np.shape(self.online_EMG.mu_filters)[1]): # add the amount of motor units
            self.channels_for_plotting.append(f'Motor Unit {i+1}')
            
        # Extract channel components for the plotting
        self.channel_names, self.channel_unit_names, self.channel_types = self.plotter2.initialize_channels_components(self.channels_for_plotting)
        self.force_channel_ind = self.channel_names.index(self.force_channel)
        self.channel_unit_names[self.force_channel_ind] = 'a.u.' 
        

        
    def get_force_profile(self, slope=5, mvc_level=30, mvc_duration=30, rest=120, repeat=3):
        # Creates the force profile based on the inputs of the user (Maybe put this in new class with different options for the profile)
        if slope == 0:
            slope_up = []
            slope_down = []
        else: 
            slope_up = np.arange(0,mvc_level, slope*1/self.sampling_frequency)
            slope_down =  np.arange(mvc_level,0, -slope*1/self.sampling_frequency)
        start = np.zeros(10*self.sampling_frequency)
        end = np.zeros(30*self.sampling_frequency)
        rest = np.zeros(rest*self.sampling_frequency)
        plateau = np.ones(mvc_duration*self.sampling_frequency)*mvc_level
        block =  np.concatenate([slope_up, plateau, slope_down])
        profile = np.concatenate([start, block])
        if repeat > 1:
            for i in range(0,repeat-1):
                profile = np.append(profile, rest)
                profile = np.append(profile, block)
        profile = np.concatenate([profile, end])
        time_axis = np.arange(0,len(profile)/self.sampling_frequency, 1/self.sampling_frequency)
        return profile, time_axis
    
    def start_live(self):
        """
        Initializes the force profile for a live decomposition based on user input. 
        Runs the MVC_Extraction.py which opens the file director and 
        extracts the offset and mvc value from a self-chosen file. 
        Initializes the Consumer, ConsumerThread, and Monitor instances and starts the measurement.

        """
        # Specific function to start a training measurement. It will normalize the force channel input and set the right force profile. 
        self.initialize_file_writer() 
        self.MVC_instance = MVC_Extraction(force_channel=self.force_channel)
        self.MVC_value, self.MVC_offset = self.MVC_instance.run()
        # Initialize queue
        self.consumer = Consumer()
        # Initialize thread
        self.consumer_thread = self.consumer_thread_class(
            consumer_reading_queue=self.consumer.reading_queue,
            sample_rate=self.device.get_device_sampling_frequency()
        )
        self.plotter2.scale = self.plotter2.live_y_resolution_value.value()
        self.initialize_force_profile('Measurement')
        self.consumer.open(server = self.device,
                           reading_queue_id = self.device.get_id(),
                           consumer_thread=self.consumer_thread)
        # Apply monitor function and send data to callback
        self.monitor = self.monitor_class(monitor_function = self.monitor_function, callback=self.callback, on_error=self.on_error)
        self.monitor.start()
        self.device.start_measurement(self.measurement_type)

                                                                       
    def start(self):
        # Function manage the 'Start MVC', 'Profile Preview' and 'Start Measurment' buttons
        self.plotter2.live_btn_preview.clicked.connect(lambda: (self.initialize_force_profile('Measurement'), self.preview_force_profile()))
        self.plotter2.live_btn_start_measurement.clicked.connect(lambda: self.start_live())
        
            
    def initialize_force_profile(self, input):
        """
        Initializes the force profile
        """
        self.force_profile, self.time_axis = self.get_force_profile(self.plotter2.live_slope_value.value(),self.plotter2.live_level_value.value()
                                                    ,self.plotter2.live_level_duration_value.value(), self.plotter2.live_rest_value.value(),
                                                    self.plotter2.live_repeat_value.value())
    def stop(self):
        #Stops all processes
        self.ForceFileWriter.close()
        self.monitor.stop()
        self.consumer.close()
        self.device.stop_measurement()
    
    def preview_force_profile(self):
        #Function to preview the force profile 
        plt.plot(self.time_axis, self.force_profile)
        plt.xlabel('Time (s)')
        plt.ylabel('Force (% MVC)')
        plt.title('Force Profile Preview')
        plt.xlim([0, self.time_axis[-1]])

        # Displaying the plot
        plt.show()

    def initialize_file_writer(self):
        """
        Initializes a ForceFileWriter, which is an instance of the right file writer. 
        The poly5 and xdf file writers are changed to be able to handle the added force profile. 
        """
        # Initializes the file writer instance to write away the data + force profile
        if self.file_type.lower()=='poly5':
            self.ForceFileWriter = Poly5Writer(filename=self.filename)
            self.ForceFileWriter.open(device=self.device, channel_names= self.channel_names_saving, channel_unit_names=self.channel_unit_names_saving)
        elif self.file_type.lower()=='xdf':
            self.ForceFileWriter = XdfWriter(filename=self.filename, add_ch_locs=False)
            self.ForceFileWriter.open(device=self.device, channel_names= self.channel_names_saving, channel_units=self.channel_unit_names_saving, channel_types=self.channel_types_saving)
        else: 
            print('File type not recognized, set to poly5')
            self.ForceFileWriter = Poly5Writer(filename=self.filename)
            self.ForceFileWriter.open(device=self.device, channel_names= self.channel_names_saving, channel_unit_names=self.channel_unit_names_saving)
        # Initializes a SampleData instance pass to the queue of the filewriter
        self.data_instance = SampleData(len(self.channels_for_saving), 0, [])

    def put_sample_data(self, data):
        # Changes all the SampleData attributes and puts the instance in the Queue of the filewriter. 
        self.data_instance.num_sample_sets = np.shape(data)[1]
        self.data_instance.num_samples_per_sample_set =  np.shape(data)[0]
        data = np.reshape(np.transpose(data), (np.size(data),))
        self.data_instance.samples = data
        self.ForceFileWriter.q_sample_sets.put(self.data_instance)

    def calculate_disrate(self, discharge):
        """ 
        Calculates the discharge rate for every sample as the total amount of discharges in previous second.
        It selects the newest data. If the total lengt
        """
        data = discharge[:,(self.total_pointer-self.diff_pointer):self.total_pointer] #Select the newest data
        kernel = np.ones(self.sampling_frequency) # create the kernel for the sum
        discharge_rate = np.empty(np.shape(data)) # initialize matrix for discharge rate

        # If the total length of discharges is less than the length of the new data minus the kernel length, the data must be padded with zeros at the begin.
        if np.shape(discharge[:,:self.total_pointer])[1] < np.shape(data)[1]+self.sampling_frequency: 
            padded_data = np.zeros((np.shape(data)[0],  np.shape(data)[1]+self.sampling_frequency-1))
            padded_data[:,-self.diff_pointer:] = data
        else: #the padded data is the kernel length + new data to get the running sum of the last second
            padded_data = discharge[:, (self.total_pointer-self.diff_pointer-self.sampling_frequency+1):self.total_pointer]

        #Sums all the discharges of the last second for every new data point that was added in the buffer.
        for i in range(np.shape(padded_data)[0]):
            discharge_rate[i,:] = np.convolve(padded_data[i,:], kernel, mode='valid')
        self.disrate[:,self.total_pointer-self.diff_pointer:self.total_pointer] = discharge_rate



