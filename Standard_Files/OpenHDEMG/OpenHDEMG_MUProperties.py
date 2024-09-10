#%%
from tools import *
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog  
from tkinter import filedialog  
import os
import openhdemg.library as emg


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
        filedir[i] = os.path.dirname(filepath[i]) # get folder of file 
        base[i] = os.path.basename(filepath[i]) # get file name + extension 
        name[i] = os.path.splitext(base[i])[0] # get file name 
        python_file[i]= os.path.join(filedir[i], name[i] +'_decomp.json')
        matlab_file[i]= os.path.join(filedir[i], name[i] +'_decomp.mat')
else: 
    python_file = ['C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/20\\Training_20_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S40\\Training_40_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S40\\Training_40_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/40\\Training_40_decomp.json', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/40\\Training_40_decomp.json']
    matlab_file = ['C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/20\\Training_20_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1/S40\\Training_40_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240116/Session1_2/S40\\Training_40_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S1/40\\Training_40_decomp.mat', 'C:/Users/masse/OneDrive - Universiteit Twente/Master/Stage TMSi/Metingen/HD_EMG_20240117/S2/40\\Training_40_decomp.mat']

#%% SELECT WHICH FILE YOU WANT TO ANALYZE
print('select number to analyze', list)
i = int(input())
print('processing', list[i])
decomp_python = open_json(python_file[i])
decomp_matlab = import_matlab(matlab_file[i])

#%%
emgfile = emg.emg_from_json(python_file[i])
emgfile = emg.sort_mus(emgfile=emgfile)

#%% Plot Discharge rate 
fig1 = emg.plot_idr(emgfile=emgfile)
fig1.suptitle(f'Discharge rates {listname[i]}')
axes = fig1.get_axes()
for ax in axes:
        ax.set_xlim((0,45))
fig1.savefig(listname[i]+'dischargerate.png')

#%% Plot the mu pulses 
fig = emg.plot_mupulses(
        emgfile=emgfile,
        linewidths=1,
        addrefsig=True,
        timeinseconds=True,
        showimmediately=False,
        figsize=[20,15],
)

axes = fig.get_axes()
for ax in axes:
        ax.set_xlim((0, 47))
        ax.set_xlabel('Time(s)', fontsize =15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
axes[0].set_ylabel('Motor unit', fontsize = 15)
axes[1].set_ylabel('MVC(%)', fontsize = 15)

fig.suptitle(f'MU Pulses {listname[i]}', fontsize = 25)
fig.savefig(listname[i]+'mupulses.png')


fig.show()
#%% Plot the spike trains (not binary)
fig = emg.plot_ipts(
        emgfile=emgfile,
        munumber="all",
        addrefsig=True,
        timeinseconds=True,
        figsize=[20, 15],
        showimmediately=True,
)

# %% SPIKE TRIGGERED AVERAGING and plotMUAPs
sorted_rawemg = emg.sort_rawemg(
        emgfile=emgfile,
        code="None",
        orientation=180,
        n_cols = 8,
        n_rows = 8
)

# spike triggered averaging 
sta = emg.sta(
        emgfile=emgfile,
        sorted_rawemg=sorted_rawemg,
        firings="all",
        timewindow=50,
)

# plot muaps 
emg.plot_muaps(sta_dict=[sta[1], sta[2]])

#%% CONDUCTION VELOCITY calculated from the double differential of the EMG signal
sorted_rawemg = emg.sort_rawemg(
        emgfile=emgfile,
        code="None",
        orientation=180,
        n_cols = 8,
        n_rows = 8
)

# use double differential 
dd = emg.double_diff(sorted_rawemg)

# spike triggered averaging 
sta = emg.sta(
        emgfile=emgfile,
        sorted_rawemg=dd,
        firings=[0, 50],
        timewindow=50,
)

# calculate cross-correlation of spike triggered averaging 
xcc_sta = emg.xcc_sta(sta)

# open gui to calculate conduction velocity 
emg.MUcv_gui(
        emgfile=emgfile, 
        sorted_rawemg=sorted_rawemg, 
        n_firings=[0, 50], 
        muaps_timewindow=50, 
        figsize=[20, 15])
#%% SAVE the figure of the double differentials of all MUs 
for j in range(0, len(sta)):
        fig = emg.plot_muaps_for_cv(
                sta_dict=sta[j],
                xcc_sta_dict=xcc_sta[j],
                showimmediately=True,
        )
        fig.suptitle(f'CV {listname[i]}, MU{j}')
        fig.savefig(listname[i]+'MU'+str(j)+'estimationconductionvelocity.png')

emg.compute_cv()

#%% calculate mu threshold 
mus_thresholds = emg.compute_thresholds(
        emgfile=emgfile,
        event_="rt_dert",
        type_="rel",
        mvc = 100
)
mus_thresholds

# %% Compute the beginning and end of the steady state phase 
plat_thr = 0.01 # amount of the aimed force at the plateao 
ind_decomposed = np.where(emgfile["EXTRAS"] >= emgfile["EXTRAS"].max().values[0] * plat_thr)[0]
ind_steady = np.where(emgfile["EXTRAS"] >= emgfile["EXTRAS"].max().values[0])[0]
start_decomp = ind_decomposed[0]
end_decomp = ind_decomposed[-1]

start_steady = ind_steady[0]
end_steady = ind_steady[-1]

print('decomp', start_decomp, end_decomp, 'steady', start_steady, end_steady)
#%% Use the computed indices of the beginning and the end of the steady phase and the whole decomposition phase 
mus_dr_steady = emg.compute_dr(emgfile=emgfile, event_="steady", start_steady = start_steady, end_steady = end_steady)
mus_dr_steady
MUSdr_steady = round(mus_dr_steady.loc[:,"DR_all_steady"],2).to_string(index=False)

mus_dr_decomp = emg.compute_dr(emgfile=emgfile, event_="steady", start_steady = start_decomp, end_steady = end_decomp)
MUSdr_decomp = round(mus_dr_decomp.loc[:,"DR_all_steady"],2).to_string(index=False)
print(MUSdr_decomp)
#%%
df =emg.basic_mus_properties(emgfile=emgfile, start_steady = start_steady, end_steady = end_steady, mvc = 100)
df 
# %%
cov_isi_steady = emg.compute_covisi(emgfile=emgfile, start_steady = start_steady, end_steady = end_steady)
cov_isi_decomp = emg.compute_covisi(emgfile=emgfile, start_steady = start_decomp, end_steady= end_decomp)
COVisi_decomp = round(cov_isi_decomp.loc[:,"COVisi_steady"],2).to_string(index=False)
print('COVisi_decomp=', COVisi_decomp, )

COVisi_steady = round(cov_isi_steady.loc[:,"COVisi_steady"],2).to_string(index=False)
print('COVisi_decomp, steady=', COVisi_steady)
