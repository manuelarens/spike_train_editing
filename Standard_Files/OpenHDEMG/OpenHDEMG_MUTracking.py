#%%
from tools import *
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog  
from tkinter import filedialog  
import os
import openhdemg.library as emg
#%%
print('Do you want to select the files?: 0 no, 1 yes')
select_files = int(input()) #select new files or use old directory
list = ['0: s1.1 20', '1: s1.2 20', '2: s2.1 20', '3: s2.2 20', '4: s1.1 40', '5: s1.2 40', '6: s2.1 40', '7: s2.2 40']
listname = ['s1.1-20', 's1.2-20', 's2.1-20', 's2.2-20', 's1.1-40', 's1.2-40', 's2.1-40', 's2.2-40']
filepath = [None]*len(list)
filedir = [None]*len(list)
name = [None]*len(list)
base = [None]*len(list)
python_file = [None]*len(list)
matlab_file = [None]*len(list)
if select_files == 1: 
    for i in range(0, len(list)):
        print('input:', list[i])
        filepath[i] = filedialog.askopenfilename() # select matlab file 
        filepath[i] = filepath[i].replace('/', '\\')
        filedir[i] = os.path.dirname(filepath[i])
        base[i] = os.path.basename(filepath[i])
        name[i] = os.path.splitext(base[i])[0]
        python_file[i]= os.path.join(filedir[i], name[i] +'_decomp.json')
        matlab_file[i]= os.path.join(filedir[i], name[i] +'_decomp.mat')
else: 
    python_file = ['C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S40\\Training_40_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S40\\Training_40_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/40\\Training_40_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/40\\Training_40_decomp.json']
    matlab_file = ['C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S40\\Training_40_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S40\\Training_40_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/40\\Training_40_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/40\\Training_40_decomp.mat']

#%% SELECT WHICH FILE YOU WANT TO ANALYZE
print('select number 1 to track', list)
i = int(input())
print('select number 2 to track', list)
j = int(input())
print('processing', list[i], list[j])

emgfile1 = emg.emg_from_json(python_file[i])
emgfile2 = emg.emg_from_json(python_file[j])

#%%
tracking_res = emg.tracking(
        emgfile1=emgfile1,
        emgfile2=emgfile2,
        firings="all",
        derivation="dd",
        timewindow=50,
        threshold=0.6,
        matrixcode="None",
        orientation=180,
        n_rows=8,
        n_cols=8,
        exclude_belowthreshold=True,
        filter=True,
        show=True,
)

