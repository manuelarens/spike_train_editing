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
 * @file ${Setup_Experiment.py}
 * @brief This file takes user inputs for the type of measurement and 
 * the type of file writer to use. It initializes the right filepaths, 
 * Gui and file writers for a both a MVC or Training measurement. 
 */   
'''

import numpy as np
from os.path import join, dirname, realpath, normpath, exists
import json
import sys

from Test_Scripts_Nathan.force_feedback_plotter_helper import ForceFeedbackPlotterHelper
from Test_Scripts_Nathan.Gui_Experiment import Gui_Experiment
from TMSiGui.gui import Gui as Gui

from TMSiFileFormats.file_writer import FileWriter, FileFormat
from Online_EMG.online_emg_plotter_helper import OnlineEMGPlotterHelper

class Setup_Experiment:
    def __init__(self, device, app, force_channel, grid_type = None):
        self.grid_type = grid_type      
        self.type = str(type)
        self.device = device
        self.app = app
        self.force_channel = force_channel
    
    def run(self, filepath_decomp = r'C:/Users/natha/OneDrive - Universiteit Twente/Universiteit/Master/Internship/Python Interface 5.2.0/tmsi-python-interface/measurements/Training_measurement+ForceProfile-20240410_153554_decomp.json'
            , EMG_reject = []):
        """ 
        Asks what kind of measurement you want to do: an calibration, MVC or training 
        Asks what kind of file type you want to use for the measurement. 
        Initializes the forcefeedbackplotterhelper, GUI_experiment and right file writer instances. 
        Launches the app.
        """
        # Initializes the right filewriter, filename, plotterhelper and gui for an experiment. 
        measurement_type=int(input("What kind of measurement do you want to do? Calibration (type: 0), MVC (type: 1), Training (type: 2), Live Decomposition (type: 3) \n"))
        self.file_path()
        if measurement_type==0:
            file_type = input("What kind of file format do you want (Options: poly5 or xdf)\n")
            # Initialise a file-writer class (Poly5-format) and state its file path
            if file_type.lower()=='poly5':
                file_writer = FileWriter(FileFormat.poly5, join(self.measurements_dir,"Calibration_measurement.poly5"))
            elif file_type.lower()=='xdf':
                file_writer = FileWriter(FileFormat.xdf, join(self.measurements_dir,"Calibration_measurement.xdf"))
            else: 
                print('File type not recognized, set to poly5')
                file_writer = FileWriter(FileFormat.poly5, join(self.measurements_dir,"Calibration_measurement.poly5"))
            file_writer.open(self.device)    
            # Define the handle to the device
            plotter_helper = ForceFeedbackPlotterHelper(device=self.device, meas_type= measurement_type, force_channel=self.force_channel, 
                                                        filename = join(self.measurements_dir,"Calibration_measurement"), hpf=20, lpf=500, order=2,
                                                        file_type=file_type)
            # Define the GUI object and show it 
            gui = Gui_Experiment(plotter_helper = plotter_helper)
            # Enter the event loop
            self.app.exec_()
            file_writer.close()
        elif measurement_type==1:
            # Initialise a file-writer class (Poly5-format) and state its file path
            file_type = input("What kind of file format do you want (Options: poly5 or xdf)\n")
            # Define the handle to the device
            plotter_helper = ForceFeedbackPlotterHelper(device=self.device, meas_type = measurement_type , force_channel=self.force_channel, 
                                                        filename = join(self.measurements_dir,"MVC_measurement+ForceProfile"), hpf=20, lpf=500, order=2,
                                                        file_type=file_type)
            # Define the GUI object and show it 
            gui = Gui_Experiment(plotter_helper = plotter_helper)
            # Enter the event loop
            self.app.exec_()

        elif measurement_type==2:
             # Initialise a file-writer class (Poly5-format) and state its file path
            file_type = input("What kind of file format do you want (Options: poly5 or xdf)\n")
            self.plotter_helper = ForceFeedbackPlotterHelper(device=self.device, meas_type= measurement_type, force_channel=self.force_channel, 
                                                        filename = join(self.measurements_dir,"Training_measurement+ForceProfile"), hpf=20, lpf=500, order=2,
                                                        file_type=file_type)
            # Define the GUI object and show it 
            gui = Gui_Experiment(plotter_helper = self.plotter_helper)
            # Enter the event loop
            self.app.exec_()
            #After closing event loop, the file writer gets closed
        elif measurement_type==3:
            file_type = input("What kind of file format do you want (Options: poly5 or xdf)\n")
            measurement_type = 3
            plotter_helper = OnlineEMGPlotterHelper(device=self.device, meas_type= measurement_type, filepath=filepath_decomp, 
                                                    force_channel=self.force_channel, filename=join(self.measurements_dir,"Online_EMG+ForceProfile"),  
                                                    hpf=20, lpf=500, order=2, rejected_channels=EMG_reject)
                                                
            # Define the GUI object and show it 
            gui = Gui_Experiment(plotter_helper = plotter_helper)
            self.app.exec_()
            print('post processing....')
            plotter_helper.online_EMG.process(grid_names=self.grid_type, filename=plotter_helper.ForceFileWriter.filename)
            print('processing done')
            self.device.close()
        else:
            print('This is not a valid input')

    def file_path(self):
        Example_dir = dirname(realpath(__file__)) # directory of this file
        modules_dir = join(Example_dir, '..') # directory with all modules
        self.measurements_dir = join(Example_dir, '../measurements') # directory with all measurements
        sys.path.append(modules_dir)
