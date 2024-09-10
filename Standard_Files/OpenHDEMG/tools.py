import json
import gzip
import pandas as pd 
import numpy as np
from scipy import io
from scipy import signal 
from tkinter import filedialog
import os 
import warnings

def select_file():
    file_path_s11 = filedialog.askopenfilename() # select matlab file 
    file_path_s11 = file_path_s11.replace('/', '\\')
    file_dir_s11 = os.path.dirname(file_path_s11) #get the folder 
    base=os.path.basename(file_path_s11) #get the file name with extension
    name = os.path.splitext(base)[0] #get the file name without extension 
    python_s11 =  os.path.join(file_dir_s11, name +'_decomp.json')
    matlab_s11 =  os.path.join(file_dir_s11, name +'_decomp.mat')

    decomp_matlab = import_matlab(matlab_s11)
    decomp_python = open_json(python_s11)
    
    return decomp_matlab, decomp_python

def open_json(filename):
    # from openHDEMG 
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        jsonemgfile = json.load(f)
    source = json.loads(jsonemgfile["SOURCE"])
    filename = json.loads(jsonemgfile["FILENAME"])    
    fsamp = float(json.loads(jsonemgfile["FSAMP"]))
    ied = float(json.loads(jsonemgfile["IED"]))
    emg_length = json.loads(jsonemgfile["EMG_LENGTH"])
    # number_of_mus = json.loads(jsonemgfile["NUMBER_OF_MUS"])
    ipts = pd.read_json(jsonemgfile["IPTS"], orient = 'split')
    ipts = ipts.transpose().to_numpy()    
    binary_mus_firing = pd.read_json(jsonemgfile["BINARY_MUS_FIRING"], orient = 'split')
    binary_mus_firing = binary_mus_firing.transpose().to_numpy()

    mupulses = json.loads(jsonemgfile["MUPULSES"])
    for num, element in enumerate(mupulses):
        mupulses[num] = np.array(element)

    ref_signal = pd.read_json(jsonemgfile["REF_SIGNAL"], orient='split')
    ref_signal.columns = ref_signal.columns.astype(int)
    ref_signal.index = ref_signal.index.astype(int)
    ref_signal.sort_index(inplace=True)
    ref_signal = ref_signal.to_numpy()

    # EXTRAS
    extras = pd.read_json(jsonemgfile["EXTRAS"], orient='split')
    # idx = sort_mus(mupulses)
    
    emgfile_json = {
    "SOURCE": source,
    "FILENAME": filename,
    "FSAMP": fsamp,
    "IED": ied,
    "EMG_LENGTH": emg_length,
    "NUMBER_OF_MUS": number_of_mus,
    "REF_SIGNAL": ref_signal,
    "IPTS": ipts,
    "BINARY_MUS_FIRING": binary_mus_firing,
    "MUPULSES": mupulses,
    "REF_SIGNAL": ref_signal,
    "IDX": idx, 
    "TARGET": extras, 
    }   
    
    return emgfile_json

def import_matlab(file):
    matlabemgfile = io.loadmat(file)['signal']
    mupulses = matlabemgfile['Dischargetimes'][0][0][0].tolist()
    for i in range(0, len(mupulses)): 
        mupulses[i] = np.squeeze (mupulses[i]) #make it 1D 
        mupulses[i] = mupulses[i]-1 #as indixing is different in python/matalb 
        # mupulses[i] = mupulses[i]-1
    ipts = matlabemgfile['Pulsetrain'][0][0][0][0]
    target = matlabemgfile['target'][0][0][0]
    binary_mus_firing = np.zeros((ipts.shape[0], ipts.shape[1]))
    for i in range(0, len(ipts)):
        binary_mus_firing[i, mupulses[i]] = 1      
    ref_signal = matlabemgfile['path'][0][0][0]
    number_of_mus = len(mupulses)
    fsamp = matlabemgfile['fsamp'][0][0][0][0]
    extras = []
    
    idx = sort_mus(mupulses)
    
    emgfile_matlab = {
    # "SOURCE": source,
    # "FILENAME": filename,
    "FSAMP": fsamp,
    "NUMBER_OF_MUS": number_of_mus,
    "IPTS": ipts,
    "BINARY_MUS_FIRING": binary_mus_firing,
    "MUPULSES": mupulses,
    "REF_SIGNAL": ref_signal,
    "TARGET": target,
    "IDX": idx, 
    "EXTRAS": extras, 
    }   
       
    return emgfile_matlab
    
def sort_mus(mupulses):
    mupulses_idx = [None] * len(mupulses) 
    for trial in range(len(mupulses)):
        mupulses_idx[trial]  = mupulses[trial][0] #get the first discharge times of all MUs 
    mupulses_sort = mupulses_idx.copy()
    mupulses_sort.sort() #sort the first discharge times 
    
    idx = [None] * len(mupulses)
    for i in range(len(mupulses)):
        idx[i] = mupulses_idx.index(mupulses_sort[i]) #find the discharge times on ascending order 
    
    return idx

def plot_mus(ax1, dictionary, order = 1): 
    color = 'tab:blue'
    if order == 1: # plot spike trains in order of recruitment 
        for trial in range(len(dictionary["MUPULSES"])):
            ax1.vlines(dictionary["MUPULSES"][dictionary["IDX"][trial]], trial - 0.5, trial + 0.5, color = color, linewidth = 1)
            ax1.set_yticks(range(len(dictionary["MUPULSES"])))
            ax1.set_yticklabels(dictionary['IDX'])   
    else: # plot spike trains in order of the file 
        for trial in range(len(dictionary["MUPULSES"])):
            ax1.vlines(dictionary["MUPULSES"][trial], trial - 0.5, trial + 0.5, linewidth = 1)    
            ax1.set_yticks(range(len(dictionary["MUPULSES"])))    
    ax1.set_xlim([0, np.shape(dictionary["IPTS"])[1]])
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Spike train number', color = color)
    ax1.set_title('Spike trains')
    ax1.tick_params(axis='y', labelcolor = color)
    
    color = 'tab:red'
    ax2 = ax1.twinx () # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('target (% MVC)', color = color)
    ax2.plot(dictionary["TARGET"], color = color)
    ax2.tick_params(axis='y', labelcolor = color) 
    return ax1, ax2 

def plot_mus_comparison(ax1, dictionary_matlab, dictionary_python, name, order = 1): 
    for trial in range(len(dictionary_matlab["MUPULSES"])): #always in order of recruitment 
        ax1.vlines(dictionary_matlab["MUPULSES"][dictionary_matlab["IDX"][trial]], trial - 0.5, trial, linewidth = 0.5, colors = 'g', label = 'Matlab')        
        ax1.vlines(dictionary_python["MUPULSES"][dictionary_python["IDX"][trial]], trial, trial + 0.5, linewidth = 0.5, colors = 'b', label = 'Python')        
    # ax1.set_xlim([0, np.shape(dictionary_matlab["IPTS"])[1]])
    ax1.legend(['Matlab', 'Python'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('MU')
    ax1.set_ylim([-0.5, len(dictionary_matlab["MUPULSES"])+0.5])
    ax1.set_yticks(range(len(dictionary_matlab["MUPULSES"])))
    ax1.set_title(f'Spike trains comparison MUEdit vs. Python {name}')
    ax1.tick_params(axis='y')
    
    color = 'tab:orange'
    ax2 = ax1.twinx() #Instantiate second axis that shares same x-axis 
    ax2.set_ylabel('target (% MVC)', color = color)
    # ax2.set_xlabel('Time (s)')
    ax2.plot(dictionary_matlab["TARGET"], color = color)
    ax2.tick_params(axis='y', labelcolor = color)
    ax2.set_xlim([0, np.shape(dictionary_matlab["IPTS"])[1]])
    ax2.set_ylim([0, max(dictionary_matlab["TARGET"])+1])
    xticks = np.arange(0, np.shape(dictionary_matlab["IPTS"])[1],5*dictionary_matlab["FSAMP"])
    xlabels = list(range(0,int(np.round(np.shape(dictionary_matlab["IPTS"])[1]/dictionary_matlab['FSAMP'])),5))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels)
    
    return ax1, ax2 

def calc_roa(matlab_dictionary, python_dictionary):
    jitter = 0.0005 #0.5 ms amount difference in time lags  
    jit = round(jitter * matlab_dictionary['FSAMP']) #amount of variation in discharge times 
    distimmp_python = [None]*(np.shape(python_dictionary["BINARY_MUS_FIRING"])[0])
                       
    for i in range(np.shape(python_dictionary["BINARY_MUS_FIRING"])[0]): #for each MU 
        distimmp_python_i = []
        for j in range(1, jit + 1):   
           distimmp_python_i = np.concatenate((python_dictionary["MUPULSES"][i]-j, python_dictionary["MUPULSES"][i]+j))
        distimmp_python_i = np.concatenate((distimmp_python_i, python_dictionary["MUPULSES"][i]))   
        distimmp_python[i]=distimmp_python_i
 

    RoA = np.zeros((len(matlab_dictionary["BINARY_MUS_FIRING"]), len(python_dictionary["BINARY_MUS_FIRING"])))
    common_discharges = np.zeros((len(matlab_dictionary["BINARY_MUS_FIRING"]), len(python_dictionary["BINARY_MUS_FIRING"]))) #check whether np.zeros(len(pulse_trains))
    coordinates_com = np.zeros((max(len(matlab_dictionary["BINARY_MUS_FIRING"]), len(python_dictionary["BINARY_MUS_FIRING"])), 2))
    RoA_com = [None]*len(matlab_dictionary["BINARY_MUS_FIRING"])
    num_com = 0 
    for mu in range(0,len(matlab_dictionary["BINARY_MUS_FIRING"])):
        for nu in range(0, len(python_dictionary['BINARY_MUS_FIRING'])):
            com = np.intersect1d(matlab_dictionary["MUPULSES"][mu], distimmp_python[nu]) #calculate amount of common discharges
            idx = np.append(False, np.diff(com) == 1) #diffference in discharge times of 1 --> meaning its because of the lags 
            if com.size != 0: 
                com = com[~idx] #delete common discharge times because of the lags 
            #concatenate false at beginning of logical array, removes array where diff=1  
            common_discharges[mu,nu] = len(com)/max(len(matlab_dictionary["MUPULSES"][mu]), len(distimmp_python[nu])) #check sizes 
            if common_discharges[mu,nu] > 0.3: 
                coordinates_com[mu,:] = [mu, nu]
                Aj = len(com) #both matlab andpython
                Ij = len(matlab_dictionary["MUPULSES"][mu])-len(com) #identified by matlab, not by python 
                Sj = len(python_dictionary["MUPULSES"][nu])-len(com) #identified by python, but not by matlab 
                RoA[mu,nu] = Aj/(Aj+Ij+Sj)
                RoA_com[num_com] = Aj/(Aj+Ij+Sj)
                num_com +=1 
    return num_com, coordinates_com, RoA_com, common_discharges, RoA

def xcorr(x,y): # https://stackoverflow.com/questions/25830840/python-cross-correlation
    norm_x = x/np.linalg.norm(x)
    norm_y = y/np.linalg.norm(y)
    corr = signal.correlate(norm_x, norm_y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr

def emg_from_json(filepath):
    """
    Load the emgfile or emg_refsig stored in json format.

    Parameters
    ----------
    filepath : str or Path
        The directory and the name of the file to load (including file
        extension .json).
        This can be a simple string, the use of Path is not necessary.

    Returns
    -------
    emgfile : dict
        The dictionary containing the emgfile.

    See also
    --------
    - save_json_emgfile : Save the emgfile or emg_refsig as a JSON file.
    - askopenfile : Select and open files with a GUI.

    Notes
    -----
    The returned file is called ``emgfile`` for convention
    (or ``emg_refsig`` if SOURCE in ["OTB_REFSIG", "CUSTOMCSV_REFSIG", "DELSYS_REFSIG"]).

    Examples
    --------
    For an extended explanation of the imported emgfile use:

    >>> import openhdemg.library as emg
    >>> emgfile = emg.emg_from_json(filepath="path/filename.json")
    >>> info = emg.info()
    >>> info.data(emgfile)
    """

    # Read and decompress json file
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        jsonemgfile = json.load(f)

    """
    print(type(jsonemgfile))
    <class 'dict'>
    """

    # Access the dictionaries and extract the data.
    source = json.loads(jsonemgfile["SOURCE"])
    filename = json.loads(jsonemgfile["FILENAME"])

    if source in ["DEMUSE", "OTB", "CUSTOMCSV", "DELSYS"]:
        # RAW_SIGNAL
        # df are stored in json as a dictionary, it can be directly extracted
        # and converted into a pd.DataFrame.
        # index and columns are imported as str, we need to convert it to int.
        raw_signal = pd.read_json(jsonemgfile["RAW_SIGNAL"], orient='split')
        # Check dtypes for safety, little computational cost
        raw_signal.columns = raw_signal.columns.astype(int)
        raw_signal.index = raw_signal.index.astype(int)
        raw_signal.sort_index(inplace=True)
        # REF_SIGNAL
        ref_signal = pd.read_json(jsonemgfile["REF_SIGNAL"], orient='split')
        ref_signal.columns = ref_signal.columns.astype(int)
        ref_signal.index = ref_signal.index.astype(int)
        ref_signal.sort_index(inplace=True)
        # ACCURACY
        accuracy = pd.read_json(jsonemgfile["ACCURACY"], orient='split')
        try:
            accuracy.columns = accuracy.columns.astype(int)
        except Exception:
            accuracy.columns = [*range(len(accuracy.columns))]
            warnings.warn(
                "Error while loading accuracy, check or recalculate accuracy"
            )
            # TODO error occurring when accuracy was recalculated on empty MUs.
            # Check if the error is present also for other params.
        accuracy.index = accuracy.index.astype(int)
        accuracy.sort_index(inplace=True)
        # IPTS
        ipts = pd.read_json(jsonemgfile["IPTS"], orient='split')
        ipts.columns = ipts.columns.astype(int)
        ipts.index = ipts.index.astype(int)
        ipts.sort_index(inplace=True)
        # MUPULSES
        # It is s list of lists but has to be converted in a list of ndarrays.
        mupulses = json.loads(jsonemgfile["MUPULSES"])
        for num, element in enumerate(mupulses):
            mupulses[num] = np.array(element)
        # FSAMP
        # Make sure to convert it to float
        fsamp = float(json.loads(jsonemgfile["FSAMP"]))
        # IED
        ied = float(json.loads(jsonemgfile["IED"]))
        # EMG_LENGTH
        # Make sure to convert it to int
        emg_length = int(json.loads(jsonemgfile["EMG_LENGTH"]))
        # NUMBER_OF_MUS
        number_of_mus = int(json.loads(jsonemgfile["NUMBER_OF_MUS"]))
        # BINARY_MUS_FIRING
        binary_mus_firing = pd.read_json(
            jsonemgfile["BINARY_MUS_FIRING"],
            orient='split',
        )
        binary_mus_firing.columns = binary_mus_firing.columns.astype(int)
        binary_mus_firing.index = binary_mus_firing.index.astype(int)
        binary_mus_firing.sort_index(inplace=True)
        
        # EXTRAS
        # Don't alter index and columns as these could contain anything.
        extras = pd.read_json(jsonemgfile["EXTRAS"], orient='split')
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

    elif source in ["OTB_REFSIG", "CUSTOMCSV_REFSIG", "DELSYS_REFSIG"]:
        # FSAMP
        fsamp = float(json.loads(jsonemgfile["FSAMP"]))
        # REF_SIGNAL
        ref_signal = pd.read_json(jsonemgfile["REF_SIGNAL"], orient='split')
        ref_signal.columns = ref_signal.columns.astype(int)
        ref_signal.index = ref_signal.index.astype(int)
        ref_signal.sort_index(inplace=True)
        # EXTRAS
        extras = pd.read_json(jsonemgfile["EXTRAS"], orient='split')

        emgfile = {
            "SOURCE": source,
            "FILENAME": filename,
            "FSAMP": fsamp,
            "REF_SIGNAL": ref_signal,
            "EXTRAS": extras,
        }

    else:
        raise Exception("\nFile source not recognised\n")

    return emgfile


# ---------------------------------------------------------------------
# Function to open files from a GUI in a single line of code.