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

def _smooth(timeframe, values, timesteps): 
    newtimeframe = np.linspace(timeframe.min(), timeframe.max(), int(timesteps ))
    # print(f"smoothing: {len(timeframe)} -> {len(newtimeframe)}") 
    # smooth_returns = interp1d(timeframe, returns, kind='cubic')  
    # returns = smooth_returns(newtimeframe)  
    spl = make_interp_spline(timeframe, values, k=3)  # type: BSpline
    values = spl(newtimeframe)
    timeframe = newtimeframe
    return timeframe, values 

###########################################################################
###########################################################################
###########################################################################

def df_episodes_energy(data, smooth=False, etype="min", etank_init=None):
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
        enrgy_exiting = etank_init-np.array(energy)
        norm_level =  1 - enrgy_exiting/etank_init 
    else:
        enrgy_exiting = np.zeros(len(energy)) 
        norm_level = np.zeros(len(energy)) 

    num_episodes = len(energy)
    timeframe = np.arange(num_episodes)
    if smooth:
        timeframe, energy = _smooth(timeframe,energy,num_steps(data))
        timeframe, enrgy_exiting = _smooth(timeframe,enrgy_exiting,num_steps(data))
        timeframe, norm_level = _smooth(timeframe,norm_level,num_steps(data))
    return pd.DataFrame(dict(Episodes = timeframe, Energy = energy, Level=norm_level, Exiting=enrgy_exiting ))  

def df_run_episodes_energy( run_folder_path, smooth=False, etype = "min", etank_init=None): 
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 

    for file_name in os.listdir(saved_logs_path):  
        name, ext = os.path.splitext(file_name)   
        if name.startswith("energy_") and ext==".json": 
            file_path = os.path.join(saved_logs_path, file_name)
            data = dataload(file_path) 
            df = df_episodes_energy(data=data, smooth=smooth, etype=etype, etank_init=etank_init) 
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df

def df_multiruns_episodes_energy( run_paths_list, smooth=False, etype = "min", run_label_list=[], etank_init_list=[]):  
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given list of runs'''
    comb_df = pd.DataFrame()   
    for i, run_folder_path in enumerate(run_paths_list): 
        if len(etank_init_list)>0:
            etank_init = etank_init_list[i]
        else:
            etank_init = None
        df = df_run_episodes_energy(run_folder_path=run_folder_path, smooth=smooth, etype=etype, etank_init=etank_init) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else:
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id 
        df["Runs"] = [run_label]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 
     
###########################################################################
###########################################################################
###########################################################################

def df_episodes_error(data, smooth=False, final_error=True ):
    ''' DataFrame with the errors corresponding to each episode of the loaded training'''
    if final_error:
        errors = data["err_pos_final_episode"]   
    else:
        errors = data["err_pos_tot_episode"]   
    num_episodes = len(errors)
    timeframe = np.arange(num_episodes)
    if smooth:
        timeframe, errors = _smooth(timeframe,errors,num_steps(data))
    return pd.DataFrame(dict(Episodes = timeframe, Errors = errors))  

def df_run_episodes_error( run_folder_path, smooth=False, final_error=True ): 
    ''' DataFrame with the errors corresponding to each episode of all the trainings in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_logs_path = os.path.join(run_folder_path, "logs") 

    for file_name in os.listdir(saved_logs_path):  
        name, ext = os.path.splitext(file_name)   
        if name.startswith("errors_") and ext==".json": 
            file_path = os.path.join(saved_logs_path, file_name)
            data = dataload(file_path) 
            df =  df_episodes_error(data, final_error=final_error)
            df["Trainings"] = [name]*len(df["Episodes"])
            comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df

def df_multiruns_episodes_error( run_paths_list, smooth=False , run_label_list=[], final_error=True ):  
    ''' DataFrame with the energy corresponding to each episode of all the trainings in the given list of runs'''
    comb_df = pd.DataFrame()   
    for i, run_folder_path in enumerate(run_paths_list): 
        df = df_run_episodes_error(run_folder_path, smooth=smooth, final_error=final_error ) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else:
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id 
        df["Runs"] = [run_label]*len(df["Trainings"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 

###########################################################################
###########################################################################
###########################################################################

def df_test_run_energy(run_folder_path, etank_init=None, smooth=False, etype = "min"): 
    ''' DataFrame with the energy corresponding to each episode of all the tests in the given run'''
    comb_df = pd.DataFrame()  
    print(f"Loading logs from: {run_folder_path}")
    saved_energy_test_path = os.path.join(run_folder_path, "energy_eval_run.json")  
    data = dataload(saved_energy_test_path)
    energy_values = []
    tank_levels = []
    for v in data.values():
        if etank_init is not None:
            norm_level = 1+(np.array(v) - etank_init)/etank_init 
        else:
            norm_level = np.zeros(len(v))
        val = np.array(v)
        energy_values = np.concatenate([energy_values,val])
        tank_levels = np.concatenate([tank_levels,norm_level])
    df = pd.DataFrame(dict(Tests = np.arange(len(energy_values)), Energy = energy_values, Level = tank_levels))    
    return df

def df_test_multirun_energy(run_paths_list, etank_init_list=[], smooth=False, etype = "min", run_label_list=[]):
    comb_df = pd.DataFrame()   
    for i,run_folder_path in enumerate(run_paths_list): 
        if len(etank_init_list)>0:
            etank_init = etank_init_list[i]
        else:
            etank_init = None
        df = df_test_run_energy(run_folder_path, etank_init, smooth, etype) 
        if len(run_label_list) > 0:
            run_label = run_label_list[i]
        else: 
            env_id, run_id  = run_folder_path.split("/")[-2:]  
            run_label = env_id+"_"+run_id  
        df["Runs"] = [run_label]*len(df["Tests"])
        comb_df = pd.concat([comb_df, df], ignore_index=True)
    return comb_df 

 