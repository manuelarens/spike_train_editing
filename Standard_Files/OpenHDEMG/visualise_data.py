from tools import *
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog  
from tkinter import filedialog  
import os
import numpy as np
import openhdemg.library as emg


"""
Make sure the openhdemg library is downloaded. Else use 'pip install openhdemg' in Terminal to install it in your current environment. 
Do not download it in TMSi python environment as it has contradicting package dependencies.
Extract filepath of the .json decomposition file you want to analyze. 
"""
# CHOOSE FILE HERE:
python_file = r'C:/Users/natha/OneDrive - Universiteit Twente/Universiteit/Master/Internship/Python Interface 5.2.0/tmsi-python-interface/measurements/Standard_Files/Training_decomposition_results.json'

emgfile = emg_from_json(python_file)
# emgfile = emg.sort_mus(emgfile=emgfile)

# Plot Discharge rate 
fig = emg.plot_idr(emgfile=emgfile)
fig.suptitle(f'Discharge rates')
axes = fig.get_axes()
for ax in axes:
        ax.set_xlim((0,45))
fig.show()

#Plot the mu pulses 
fig1 = emg.plot_mupulses(
        emgfile=emgfile,
        linewidths=1,
        addrefsig=True,
        timeinseconds=True,
        showimmediately=False,
        figsize=[20,15]
)

axes = fig1.get_axes()
for ax in axes:
        ax.set_xlim((0, 60))
        ax.set_xlabel('Time(s)', fontsize =15)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
axes[0].set_ylabel('Motor unit', fontsize = 15)
axes[1].set_ylabel('MVC(%)', fontsize = 15)

fig1.suptitle(f'MU Pulses ', fontsize = 25)
fig1.show()


fig2 = emg.plot_ipts(
        emgfile=emgfile,
        munumber="all",
        addrefsig=True,
        timeinseconds=True,
        figsize=[20, 15],
        showimmediately=True,
)
