READ ME for standard data files

Calibration_file_foot_pedal.poly5: 
- contains only UNI channels, 1 aux (force sensor) and STATUS and COUNTER channels. Must be opened with a normal Poly5reader.
- One calibration measurement with a simple offset measurement for load cell with a contraction after 5 seconds.

MVC_measurement_raw_data.poly5: 
- contains only UNI channels, 1 aux (load cell), STATUS, COUNTER and force profile. Can be opened with poly5_force_file_reader.
- MVC_measurement with raw force channel data with one MVC peak in it. 

Training_measurement_raw_data.poly5: 
- contains only UNI channels, 1 aux (load cell), STATUS, COUNTER and force profile. Must be opened with poly5_force_file_reader.
- Contains raw data of a training measurement in which the force profile is followed. Force channel data is normalized based on the MVC value extracted from MVC_measurement_raw_data.poly5.

Training_decomposition_results.json: 
- contains the training measurement's decomposition results (pulse trains, discharge times). 
- This file can be visualized with the OpenHDEMG library. In the folder OpenHDEMG is a file called 'visualize_data.py'which can be used for visualisation.

Online_measurement_raw_data.poly5: 
- contains only UNI channels, 1 aux (load cell), STATUS, COUNTER and force profile. Must be opened with poly5_force_file_reader.
- Contains raw data of an Online measurement in which the force profile is followed. Force channel data is normalized based on the MVC value extracted from MVC_measurement_raw_data.poly5.

Online_decomposition_results.json: 
- contains the online's measurements decomposition results (pulse trains, discharge times).
- This file can be visualized with the OpenHDEMG library. In the folder OpenHDEMG is a file called 'visualize_data.py' which can be used for the visualisation.

OpenHDEMG/visualize_data.py:
Code to open the results from a .json decomposition file. Must be run with OpenHDEMG library installed in python interpreter. However, do NOT install OpenHDEMG library in the virtual environment (.venv) of TMSi python gitlab as this has other package dependencies. Therefore, install it in a new venv or on the base installation of python. 