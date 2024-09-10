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
 * @file ${Gui_Experiment.py}
 * @brief This file is used as GUI for measurements with force feedback.
    
'''

from PySide2.QtWidgets import *
from TMSiFrontend import TMSiStyle

class Gui_Experiment:
    def __init__(self,plotter_helper):
        """
        Creates the windows and sets the force channel and force profile to be displayed.
        Also initiates the execution of a measurement when the main window is closed.
        """
        self.plotter_helper = plotter_helper

        # Main window (plotter2, so forcefeedback tool is the main window)
        self.window = QMainWindow()
        self.window.setWindowTitle("Force Feedback Window")
        self.main_plotter = plotter_helper.main_plotter
        self.plotter2 = plotter_helper.plotter2
        self.window.setCentralWidget(self.plotter2)
        self.window.resize(self.plotter2.size())
        self.window.closeEvent = self.closeEvent
        self.window.setStyleSheet(TMSiStyle)
        self.window.show()
        
        # Secondary window
        self.window2 = QMainWindow()
        self.plotter2 = plotter_helper.plotter2
        self.window2.setWindowTitle("Primary window")
        self.window2.setCentralWidget(self.main_plotter)
        self.window2.resize(self.main_plotter.size())
        self.window2.setStyleSheet(TMSiStyle)
        self.window2.show()
        # Initialze and start acquisition
        self.plotter_helper.initialize()
        self.main_plotter.enable_all_channels(enabled = False) # disables all channels of EMG plotter
        self.plotter2.enable_all_channels(enabled = False) #disables all channels of feedback tool 
        self.plotter2.enable_channels() #Enables the force channel and force profile. 
        self.plotter_helper.start()
        

    def closeEvent(self, event): # Every measurement is stopped by closing the main window  
        self.window2.close()
        print("closing")
        if self.plotter_helper.meas_type == 0:  # in a calibration/, there is no monitor so only consumer and device must be stopped. 
            self.plotter_helper.consumer.close()
            self.plotter_helper.device.stop_measurement()
            self.plotter2.offset_timer.stop()
            return
        self.plotter_helper.stop()
