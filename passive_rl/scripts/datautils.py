import numpy as np
import pandas as pd 
import json 
import os
from scipy.interpolate import interp1d, make_interp_spline, BSpline 
  
def dataload( file_path): 
    with open(file_path, 'r') as f: 
        data = json.loads(f.read())  
    return data
 
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
    
def multirun_steps(run_paths_list):
    num_steps_list = []
    for i, run_folder_path in enumerate(run_paths_list): 
        saved_logs_path = os.path.join(run_folder_path, "logs") 
        run_num_steps_list = []
        for file_name in os.listdir(saved_logs_path):  
            name, ext = os.path.splitext(file_name)   
            if name.startswith("energy_") and ext==".json": 
                file_path = os.path.join(saved_logs_path, file_name)
                data = dataload(file_path) 
                run_num_steps_list += [num_steps(data)]
        num_steps_list += [run_num_steps_list]
    return num_steps_list

def _interpolate(timeframe, values, timesteps): 
    newtimeframe = np.linspace(timeframe.min(), timeframe.max(), int(timesteps )) 
    spl = make_interp_spline(timeframe, values, k=3)  # type: BSpline
    values = spl(newtimeframe)
    timeframe = newtimeframe
    return timeframe, values 

def _smooth(values, weight=0.9):
    smoothed = np.array(values)
    for i in range(1, smoothed.shape[0]):
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * smoothed[i]
    return smoothed

###########################################################################
###########################################################################
###########################################################################

def df_episodes_energy(data, smooth=False, interpolate=False, etype="min", etank_init=None):
    ''' DataFrame with the energy corresponding to each episode of the loaded training'''
    if etype == "min":
        energy = get_etankmin(data)
    elif etype == "max":
        energy = get_etankmax(data)
    elif etype == "final":
        energy = get_etankfinal(data)
    else:
        exit(f"Energy '{etype}', not saved in logs")

    if etank_init is not None:
        energy_exiting = etank_init - np.array(energy)  
        norm_level =  1 - energy_exiting/etank_init 
    else:
        energy_exiting = np.zeros(len(energy)) 
        norm_level = np.zeros(len(energy)) 

    num_episodes = len(energy)
    timeframe = np.arange(num_episodes)
    if interpolate:
        timeframe, energy = _interpolate(timeframe,energy,num_steps(data))
        timeframe, energy_exiting = _interpolate(timeframe,energy_exiting,num_steps(data))
        timeframe, norm_level = _interpolate(timeframe,norm_level,num_steps(data))
    if smooth:
        energy = _smooth(energy)
        energy_exiting = _smooth(energy_exiting)
        norm_level = _smooth(norm_level)
    return pd.DataFrame(dict(Episodes = timeframe, Energy = energy, Level=norm_level, Exiting=energy_exiting ))  

def df_run_episodes_energy( run_folder_path, smooth=False, interpolate=False, etype = "min", etank_init=None): 
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 

    for file_name in os.listdir(saved_logs_path):  
        name, ext = os.path.splitext(file_name)   
        if name.startswith("energy_") and ext==".json": 
            file_path = os.path.join(saved_logs_path, file_name)
            data = dataload(file_path) 
            df = df_episodes_energy(data=data, smooth=smooth, interpolate=interpolate, etype=etype, etank_init=etank_init) 
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df

def df_multiruns_episodes_energy( run_paths_list, smooth=False, interpolate=False, etype = "min", run_label_list=[], etank_init_list=[]):  
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given list of runs'''
    comb_df = pd.DataFrame()   
    for i, run_folder_path in enumerate(run_paths_list): 
        if len(etank_init_list)>0:
            etank_init = etank_init_list[i]
        else:
            etank_init = None
        df = df_run_episodes_energy(run_folder_path=run_folder_path, smooth=smooth, interpolate=interpolate, etype=etype, etank_init=etank_init) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else:
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id 
        df["Runs"] = [run_label]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 
     
###########################################################################

def df_test_run_etank(run_folder_path, etank_init=None ): 
    ''' DataFrame with the energy corresponding to each episode of all the tests in the given run'''
    print(f"Loading logs from: {run_folder_path}")
    saved_etank_test_path = os.path.join(run_folder_path, "etank_eval_run.json")  
    data = dataload(saved_etank_test_path)
    etank_values = []
    tank_levels = []
    for etank in data.values():
        if etank_init is not None:
            energy_exiting = etank_init - np.array(etank) 
            norm_level =  1 - energy_exiting/etank_init  
        else:
            norm_level = np.zeros(len(etank))  
 
        etank_values = np.concatenate([etank_values,np.array(etank)])
        tank_levels = np.concatenate([tank_levels,norm_level])
    df = pd.DataFrame(dict(Tests = np.arange(len(etank_values)), Energy = etank_values, Level = tank_levels))    
    return df

def df_test_multirun_etank(run_paths_list, etank_init_list=[], run_label_list=[]):
    comb_df = pd.DataFrame()   
    for i,run_folder_path in enumerate(run_paths_list): 
        if len(etank_init_list)>0:
            etank_init = etank_init_list[i]
        else:
            etank_init = None
        df = df_test_run_etank(run_folder_path, etank_init ) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else: 
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id  
        df["Runs"] = [run_label]*len(df["Tests"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 

def df_test_run_energy(run_folder_path, smooth=False ): 
    ''' DataFrame with the energy corresponding to each episode of all the tests in the given run'''
    print(f"Loading logs from: {run_folder_path}")
    saved_energy_test_path = os.path.join(run_folder_path, "energy_eval_run.json")  
    data = dataload(saved_energy_test_path)
    run_df = pd.DataFrame()
    for model_id in data.keys():   
        for episode in data[model_id].keys():
            episode_energy = data[model_id][episode]
            num_steps = len(episode_energy) 
            timeframe = np.arange(num_steps)  
            total_energy = [sum(episode_energy[:k+1]) for k in timeframe ] 
            if smooth:
                episode_energy = _smooth(episode_energy) 
            print(f"model:{model_id} - ep:{episode}")
            df = pd.DataFrame(dict(Steps = timeframe, Energy = episode_energy, TotalEnergy = total_energy, Tests=f"{model_id}_ep{episode}" ))    
            run_df = pd.concat([run_df, df], ignore_index=True)
    return run_df  

def df_test_multirun_energy( run_paths_list, smooth=False,   run_label_list=[] ):  
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given list of runs'''
    comb_df = pd.DataFrame()   
    for i, run_folder_path in enumerate(run_paths_list):  
        df = df_test_run_energy(run_folder_path=run_folder_path, smooth=smooth ) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else:
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id 
        df["Runs"] = [run_label]*len(df["Tests"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 
 
 
###########################################################################
###########################################################################
###########################################################################

def df_episodes_error(data, smooth=False, interpolate=False, cumulative_error=False ):
    ''' DataFrame with the errors corresponding to each episode of the loaded training'''
    if cumulative_error: 
        errors = data["err_pos_tot_episode"]   
    else:
        errors = data["err_pos_final_episode"]  
    num_episodes = len(errors)
    timeframe = np.arange(num_episodes)
    if interpolate:
        timeframe, errors = _interpolate(timeframe,errors,num_steps(data))
    elif smooth:
        errors = _smooth(errors)
    return pd.DataFrame(dict(Episodes = timeframe, Errors = errors))  
 
def df_run_episodes_error( run_folder_path, smooth=False, interpolate=False, cumulative_error=False ): 
    ''' DataFrame with the errors corresponding to each episode of all the trainings in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 

    for file_name in os.listdir(saved_logs_path):  
        name, ext = os.path.splitext(file_name)   
        if name.startswith("errors_") and ext==".json": 
            file_path = os.path.join(saved_logs_path, file_name)
            data = dataload(file_path) 
            df =  df_episodes_error(data, cumulative_error=cumulative_error, smooth=smooth, interpolate=interpolate)
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df
 

def df_multiruns_episodes_error( run_paths_list, smooth=False, interpolate=False , run_label_list=[], cumulative_error=False ):  
    ''' DataFrame with the errors corresponding to each episode of all the trainings in the given list of runs'''
    comb_df = pd.DataFrame()   
    for i, run_folder_path in enumerate(run_paths_list): 
        df = df_run_episodes_error(run_folder_path, cumulative_error=cumulative_error, smooth=smooth, interpolate=interpolate) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else:
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id 
        df["Runs"] = [run_label]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 
 
###########################################################################

def df_test_run_errors(run_folder_path, smooth=False, interpolate=False): 
    ''' DataFrame with the errors corresponding to each episode of all the tests in the given run''' 
    print(f"Loading logs from: {run_folder_path}")
    saved_errors_test_path = os.path.join(run_folder_path, "errors_eval_run.json")  
    data = dataload(saved_errors_test_path)
    errors_values = [] 
    for v in data.values(): 
        val = np.array(v)
        errors_values = np.concatenate([errors_values,val]) 
    df = pd.DataFrame(dict(Tests = np.arange(len(errors_values)), Errors = errors_values ))    
    return df

def df_test_multirun_errors(run_paths_list, smooth=False, interpolate=False, run_label_list=[] ):
    comb_df = pd.DataFrame()   
    for i,run_folder_path in enumerate(run_paths_list): 
        df = df_test_run_errors(run_folder_path, smooth) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else: 
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id  
        df["Runs"] = [run_label]*len(df["Tests"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 

  