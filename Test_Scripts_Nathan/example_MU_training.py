'''
(c) 2022,2023 Twente Medical Systems International B.V., Oldenzaal The Netherlands

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
 * @file ${example_MU_training.py} 
 * @brief This example shows the functionality to do a measurement with force feedback.
.* The functionality plots a force profile which enables force feedback tracking.
.* To do so, the force output of the load cell is normalized and plotted
.* which enables the participant to follow the force profile.
.* The data can be saved in both Poly5 or XDF file type.
 */


'''

from PySide2.QtWidgets import *
import sys
import time
from os.path import join, dirname, realpath
Example_dir = dirname(realpath(__file__)) # directory of this file
modules_dir = join(Example_dir, '..') # directory with all modules
measurements_dir = join(Example_dir, '../measurements') # directory with all measurements
configs_dir = join(Example_dir, '../TMSiSDK\\tmsi_resources') # directory with configurations
sys.path.append(modules_dir)

from TMSiSDK.device import ChannelType
from TMSiSDK.tmsi_sdk import TMSiSDK, DeviceType, DeviceInterfaceType, DeviceState
from TMSiSDK.tmsi_errors.error import TMSiError 
from TMSiSDK.device.devices.saga.saga_API_enums import SagaBaseSampleRate

from Test_Scripts_Nathan.Setup_Experiment import Setup_Experiment


try:
    # Execute a device discovery. This returns a list of device-objects for every discovered device.
    TMSiSDK().discover(dev_type = DeviceType.saga, dr_interface = DeviceInterfaceType.docked, ds_interface = DeviceInterfaceType.usb)
    discoveryList = TMSiSDK().get_device_list(DeviceType.saga)

    if (len(discoveryList) > 0):
        # Get the handle to the first discovered device.
        dev = discoveryList[0]
        
        # Open a connection to the SAGA-system
        dev.open()
        
        # Set the sample rate of the BIP and AUX channels to 4000 Hz
        dev.set_device_sampling_config(base_sample_rate = SagaBaseSampleRate.Decimal,  channel_type = ChannelType.BIP, channel_divider = 2)
        dev.set_device_sampling_config(channel_type = ChannelType.AUX, channel_divider = 2)
        

        # To which aux channel is the lead cell connected?
        Force_channel = 'AUX 1-1'

        grid_type = '8-8-L'

        # Enable AUX 1-2
        AUX_list = [1]
        
        # options:'4-8-L', '6-11-L', '6-11-S', '8-8-L', '8-8-S', '6-11-L-1', '6-11-L-2', '6-11-S-1', '6-11-S-2', '8-8-L-1', '8-8-L-2', '8-8-S-1', '8-8-S-2'
        
        # Load the HD-EMG channel set and configuration
        print("load HD-EMG config")
        if dev.get_num_channels()<64:
            dev.import_configuration(join(configs_dir, "saga32_config_textile_grid_" + grid_type + ".xml"))
        else:
            dev.import_configuration(join(configs_dir, "saga64_config_textile_grid_" + grid_type + ".xml"))
        # Retrieve all channels from the device and update which should be enabled
        ch_list = dev.get_device_channels()
        
        # The counters are used to keep track of the number of AUX and BIP channels 2
        # that have been encountered while looping over the channel list
        AUX_count = 0
        enable_channels = []
        disable_channels = []
        for idx, ch in enumerate(ch_list):
            if (ch.get_channel_type() == ChannelType.AUX):
                if AUX_count in AUX_list:
                    enable_channels.append(idx)
                else:
                    disable_channels.append(idx)
                AUX_count += 1
    
        dev.set_device_active_channels(enable_channels, True)
        dev.set_device_active_channels(disable_channels, False)
        

        # Select here the number of measurement you want to do. 
        nmb_measurement = 3

        for i in range(nmb_measurement):
            app = QApplication.instance()
            
            # Initialise the plotter application if there is no other plotter application
            if not app:
                app = QApplication(sys.argv)

            Experiment_instance = Setup_Experiment(dev, app, force_channel = Force_channel,  grid_type = grid_type)
            Experiment_instance.run()
               
        # Close the connection to the SAGA device
        dev.close()
        

    
except TMSiError as e:
    print(e)
    
        
finally:
    if 'dev' in locals():
        # Close the connection to the device when the device is opened
        if dev.get_device_state() == DeviceState.connected:
            dev.close()