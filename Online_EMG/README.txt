Read_Me Online_EMG

This document will explain the pipeline for Online/live motor unit decomposition. 

Files in this folder:
- EMG_Classes.py: Contains EMG, Offline_EMG and Online_EMG classes
- Example_Decomp_pipeline.py: runs the whole pipeline for live decomposition of motor units.
- Example_Online_EMG.py: runs live decomposition based on decomposition data (optional to do offline_decompostion as well)
- offline_EMG.py: Contains EMG_Decomposition that runs the offline EMG decomposition. It uses the offline_EMG class from EMG_classes.py and the processing_tool.py.
- online_emg_plotter_helper.py: Contains live decomposition preprocessing (applying MUfilters to live data) and calculates the discharge rate.
- online_emg_plotter.py: Contains the plotter's specific settings (initialization of channels) in live decomposition.
- online_emg_chart.py: Contains the specific settings (color, thickness, labels and y ticks) for every channel chart in live decomposition.
- processing_tools.py: Contains all processing tools for both live and offline decomposition.  


EMG_Classes.py: 
Contains EMG, Offline_EMG and Online_EMG classes. EMG() is the parent class of both Offline_EMG and Online_EMG and contains processing settings for both classes. This contains the number of iterations that the algorithm does, before saving the results.
	- Offline_EMG(): contains different functions for the offline decomposition of motor units. The EMG_Decomposition() class off offline_EMG.py contains the pipeline for offline decomposition and will call the functions of this 	class. Offline_EMG() functions use functions off processing_tools.py for several processing steps.
	- Online_EMG(): Extracts found motor unit filters of offline decomposition with preprocess_with_biofeedback(). Callback() applies these MUfilters on live data. After a live decomposition measurement, it applies the motor unit 	filters on the recorded data with the process() function to save the found pulse trains and MUpulses in a _decomp.json file.


Example_Online_EMG.py:
Enables all the right channels and configurations for the measurement. The force channel and grid type must be changed according to the measurement setup.
If with_decomp is set to True, a .poly5 file of a training measurement can be selected and offline decomposition will precede the live decomposition.
After, it initializes and runs a Setup_Experiment instance in which the right kind of measurement will be set up and the live plotter will be launched. 

Called objects:
	Setup_Experiment(): (part of the Test_Scripts_Nathan folder)
	Asks what kind of measurement you want to do: calibration, MVC, training, or live decomposition
	Asks what kind of file type you want to use for the measurement. 
	Initializes the onlineEMGplotterHelper, GUI_experiment and right file writer instances. 
	Launches the app.


	Gui_Experiment():  (part of the Test_Scripts_Nathan folder)
	Creates the windows and sets the force channel and force profile to be displayed.
	Also initiates the execution of a measurement when the main window is closed.

offline_EMG.py: 
Contains the EMG_Decomposition class. This class contains the whole pipeline for offline decomposition based on training data. It saves the EMG decomposition results for biofeedback (live decomposition). 

OnlineEMGplotterhelper.py: (child class of the FilteredSignalPlotterHelper)
__init__():
Initializes the right settings for filtering and creates an onlineEMG() object. 
This online object will first load the data of a selected decomp.json file and extract the motor unit filters.

Functions:
	Initialize():
	Initialises channel names, units and types for the data that must be saved.
	Then the channel names, units and types are extracted for the data to be plotted. These are the force 	channel, force profile, and the amount of motor units found.
	Initializes an extra file writer to write away the data and uses CHANGED file writers.

	initialize_file_writer():
	Initializes a ForceFileWriter, which is an instance of the right file writer. 
	The poly5 and xdf file writers are changed to be able to handle the added force profile. 

	Start(): Waits for user input in the app before starting a live decomposition.
	Includes the preview button to preview the force profile. 

	Start_live(): initializes the force profile for live decomposition.
	Runs the MVC_Extraction.py which opens the file director and 
	extracts the offset and mvc value from a self-chosen file. 
	Initializes the Consumer, ConsumerThread, and Monitor instances and starts the measurement.

	Callback(): 
	Gets the data from the Monitor instance and plots it in 2 different windows. 
	Main window contains the online EMG plotter and the secondary window has the normal signal plotter
	Load cell data gets plotted and saved normalized based on extracted MVC values.
	** NEW DATA FROM CHANNELS 1 TILL -3 (R#C# CHANNELS) WILL BE PUT IN online_emg.callback() FUNCTION AND BINARY DISCHARGE TIMES WILL BE GIVEN BACK. THE DISCHARGE RATE OF THIS NEW DATA BLOCK BE CALCULATED WITH THE 	calculate_disrate() FUNCTION.

	put_sample_data():
	Reshapes the new send data from the device in the right shape and puts it in the SampleData instance.
	Puts it in the Queue of the ForceFileWriter.

	calculate_disrate(): calculates the discharge rate for every new sample over a 1-second window.

	Some extra added functions that are self-explanatory. 


MVC_Extraction.py: (same as in Test_Scripts_Nathan folder)
Loads a .poly5 or .xdf file and calculates the offset and maximum value of the MVC measurement based on found plateaus. The criteria for a plateau are
	- a big change precedes the plateau (np.diff between consecutive samples is large)
	- the level is at least bigger than 


online_emg_plotter.py: (Based of SignalPlotter)
Changes made in initialize_channels_components(), update_chart() and get_channel_info()

	initialize_channels_components():
	Creates the channels for force channel, force profile, and motor units
	Any channel that is no real channel of the device (such as the force profile)
	will get unit = 'a.u.' and ChannelType.Unknown. Initializes the force channel,
	force profile and all motor unit ChannelComponents.

	update_chart():
	#if self._compute_autoscale:  deleted, so the y range gets updated each iteration. 
	The standard scale value for motor units is 50 since this is the max discharge rate that can be detected by the decomposition algorithm.
	The offset is set to the middle value of the force profile. 
	In this way, it follows the force profile with a range of +/- 1.5*self.scale.
	*** IN THE DEMO MEETING, WE AGREED TO USE A STANDARD SCALE WITH OFFSET 0 FOR FORCE PROFILE AND CHANNEL, SO NEEDS TO BE CHANGED ***.
	

	get_channel_info():
	Extracts info of all the channels that need to be saved with the file writer.
	Any channel that is no real channel of the device (such as the force profile and motor units)
	will get unit = 'a.u.' and ChannelType.Unknown. 


online_emg_chart.py:
Changes made in setup_signals(), update_y_ticks(), generate_rainbow_colors():
Changes are discussed in the code

processing_tools.py: Contains all processing tools for both live and offline decomposition. 
Added the get_pulse_trains(), get_mu_filters(), get_online_parameters() and getspikesonline() compared to the code of Lisette.


At the end:
The pipeline creates 2 file for each live recording
	- one file with the original data and the force profile appended to it  (name: Online_EMG+ForceProfile_{date}_{time}.{file type})
	- one file with the decomposition results  (name: Online_EMG+ForceProfile_{date}_{time}_decomp.json), which can be opened with OpenHDEmg


*** IMPORTANT CONSIDERATIONS:
- THE FILTERING METHODS (hpf, lpf and order) IN TRAINING AND LIVE DECOMPOSITION/ONLINE_EMG MUST BE THE SAME, ELSE THE MOTOR UNIT FILTERS WILL NOT WORK ON THE LIVE DATA AND THE RESULTS WILL BE UNRELIABLE.
THEREFORE, THERE IS ALSO NO ADDITIONAL FILTERING IN THE OFFLINE DECOMPOSITION. ANY BAD CHANNELS (HIGH RMS OR IMPEDANCE) MUST BE EXCLUDED BEFORE OFFLINE DECOMPOSITION. THIS CAN BE DONE WITH EMG_reject IN THE Example_Online_EMG.py, WHICH CREATES A MASK IN OFFLINE DECOMPOSITION AND LIVE DECOMPOSITION. THE SAME emg_reject MUST BE USED IN LIVE DECOMPOSITION. THIS NECESSAR	Y BECAUSE ELSE THE MOTOR UNIT FILTERS WILL BE UNRELIABLE OR THE FILTERS WILL NOT HAVE THE RIGHT SHAPE. 
- EMG_REJECT: CONTAINS THE INDICES OF CHANNELS THAT MUST BE EXCLUDED. HOWEVER, SINCE DECOMPOSITION IS ONLY DONE ON THE GRID CHANNEL DATA (R#C# CHANNELS) AND THE NEW PYTHON INTERFACE INCLUDES THE CREF CHANNEL, THE CHANNEL INDEX DIFFERS 1 INDEX FROM THE INDEX IN THE BUFFER ARRAY. FOR EXAMPLE, R1C8 IS INDEX 1 IN THE callback_object['buffer'].dataset, BUT INDEX 0 IN THE DECOMPOSITION ALGORITHM. So index 0 must be added in EMG_reject = [] to exclude R1C8.
- ANY ADDITIONAL AUX OR BIP CHANNELS OUTSIDE OF THE LOAD CELL CHANNEL MUST BE DEACTIVATED, ELSE THE INDEXING OF THE EMG DATA WILL GO WRONG IN THE CALLBACK FUNCTION OF OnlineEMGPlotterHelper. THIS MUST BE CHANGED, HOWEVER, FOR ME THERE IS NO TIME LEFT TO DO IT. 
- FOR .poly5 files A DIFFERENT READER (poly5_force_file_reader in TestScriptsNathan) IS USED THAT DOES NOT REORDER THE CHANNELS BASED ON THE NAME OF THE CHANNELS, BECAUSE ELSE THE MOTOR UNITS FILTERS ALSO NEED TO BE REORDERED. SO THE CHANNEL ORDERING IS FOR BOTH TRAINING, OFFLINE DECOMPOSITION AND LIVE DECOMPOSITION THE SAME AS THE ORDER IN WHICH THE SAGA DEVICE DELIVERS THE DATA TO THE CALLBACK FUNCTION. 
*** 

