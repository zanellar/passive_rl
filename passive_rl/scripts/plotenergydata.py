
import os
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from mjrlenvs.scripts.eval.plotdata import mean_std_plt
from passive_rl.scripts.logenergydata import *
  


def plt_run_emin( run_folder_path, show=True, save=True, save_path="plot.pdf", smooth=True):    
    mean_std_plt(
        data = df_run_episodes_energy(run_folder_path, smooth, etype="min") , 
        title = 'Average Episode Min Tank Level',
        xaxis = "Episodes", 
        value = "Energy",  
        estimator = np.mean,
        save = save, 
        show = show, 
        save_path = save_path
    )

 
def plt_run_emax(run_folder_path, show=True, save=True, save_path="plot.pdf", smooth=True):  
    mean_std_plt(
        data = df_run_episodes_energy(run_folder_path, smooth, etype="max"), 
        title = 'Average Episode Max Tank Level',
        xaxis = "Episodes", 
        value = "Energy",  
        estimator = np.mean,
        save = save, 
        show = show, 
        save_path = save_path
    )

 
def plt_run_efinal(run_folder_path, show=True, save=True, save_path="plot.pdf", smooth=True):  
    mean_std_plt(
        data = df_run_episodes_energy(run_folder_path, smooth, etype="final"), 
        title = 'Average Episode Final Tank Level',
        xaxis = "Episodes", 
        value = "Energy",  
        estimator = np.mean,
        save = save, 
        show = show, 
        save_path = save_path
    )

 
def plt_multirun_emin(env_run_paths, show=True, save=True, save_path="plot.pdf", smooth=True):  
    mean_std_plt(
        data = df_multiruns_episodes_energy(env_run_paths, smooth, etype="min"), 
        title = 'Average Episode Min Tank Level Multirun',
        xaxis = "Episodes", 
        value = "Energy",  
        hue = "Runs",
        estimator = np.mean,
        save = save, 
        show = show, 
        save_path = save_path
    )

 
