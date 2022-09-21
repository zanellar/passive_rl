import numpy as np
import pandas as pd 
import json 
import os
from scipy.interpolate import interp1d, make_interp_spline, BSpline 
  
def dataload( file_path): 
    with open(file_path, 'r') as f: 
        data = json.loads(f.read())  
    return data

def num_episodes(data):  
    return len(data["energy_tank_min"])

def num_steps(data):   
    return int(data["num_tot_steps"])

def get_steps(data):   
    return int(data["num_ep_steps"])
        
def get_etankmin(data):  
    return data["energy_tank_min"] 

def get_etankmax(data):  
    return data["energy_tank_max"] 

def get_etankfinal(data):  
    return data["energy_tank_final"] 
    
def _smooth(timeframe, values, timesteps): 
    newtimeframe = np.linspace(timeframe.min(), timeframe.max(), timesteps)
    # smooth_returns = interp1d(timeframe, returns, kind='cubic')  
    # returns = smooth_returns(newtimeframe)  
    spl = make_interp_spline(timeframe, values, k=3)  # type: BSpline
    values = spl(newtimeframe)
    timeframe = newtimeframe
    return timeframe, values 

def df_episodes_energy(data, smooth=False, etype="min"):
    ''' DataFrame with the energy corresponding to each episode of the loaded training'''
    if etype == "min":
        energy = get_etankmin(data)
    elif etype == "max":
        energy = get_etankmax(data)
    elif etype == "final":
        energy = get_etankfinal(data)
    else:
        exit(f"Energy '{etype}', not saved in logs")

    timeframe = np.arange(num_episodes(data))
    if smooth:
        timeframe, energy = _smooth(timeframe,energy,num_steps(data))
    return pd.DataFrame(dict(Episodes = timeframe, Energy = energy))  

def df_run_episodes_energy( run_folder_path, smooth=False, etype = "min"): 
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 

    for file_name in os.listdir(saved_logs_path):  
        name, ext = os.path.splitext(file_name)   
        if name.startswith("energy_") and ext==".json": 
            file_path = os.path.join(saved_logs_path, file_name)
            data = dataload(file_path)
            df = df_episodes_energy(data, smooth,etype)
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df])
    return comb_df

def df_multiruns_episodes_energy( run_paths_list, smooth=False, etype = "min"):  
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given list of runs'''
    comb_df = pd.DataFrame()   
    for run_folder_path in run_paths_list: 
        df = df_run_episodes_energy(run_folder_path, smooth, etype) 
        env_id, run_id  = run_folder_path.split("/")[-2:]  
        df["Runs"] = [env_id+"_"+run_id]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df])
    return comb_df 
     
def df_test_run_energy(run_folder_path, smooth=False, etype = "min"): 
    ''' DataFrame with the energy corresponding to each episode of all the tests in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_energy_test_path = os.path.join(run_folder_path, "energy_eval_run.json")  
    data = dataload(saved_energy_test_path)
    energy_values = []
    for v in data.values():
        energy_values += v
    df = pd.DataFrame(dict(Tests = np.arange(len(energy_values)), Energy = energy_values))    
    return df

def df_test_multirun_energy(run_paths_list, smooth=False, etype = "min"):
    comb_df = pd.DataFrame()   
    for run_folder_path in run_paths_list: 
        df = df_test_run_energy(run_folder_path, smooth, etype) 
        env_id, run_id  = run_folder_path.split("/")[-2:]  
        df["Runs"] = [env_id+"_"+run_id]*len(df["Tests"])
        comb_df = pd.concat([comb_df, df])
    return comb_df 
