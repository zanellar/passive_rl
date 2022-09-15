import numpy as np
import pandas as pd 
import json 
import os
from scipy.interpolate import interp1d, make_interp_spline, BSpline 
 

class LogEnergyData():
    def __init__(self, file_path=None) -> None:
        if file_path is not None:
            with open(file_path, 'r') as f:
                self.data = json.loads(f.read())   

    def load(self, file_path): 
        with open(file_path, 'r') as f: 
            self.data = json.loads(f.read())  

    def num_episodes(self):  
        return len(self.data["energy_tank_min"])

    def num_steps(self):   
        return int(self.data["num_tot_steps"])

    def get_steps(self):   
        return int(self.data["num_ep_steps"])
         
    def get_etankmin(self):  
        return self.data["energy_tank_min"] 

    def get_etankmax(self):  
        return self.data["energy_tank_max"] 

    def get_etankfinal(self):  
        return self.data["energy_tank_final"] 
        
    def _smooth(self,timeframe,data): 
        newtimeframe = np.linspace(timeframe.min(), timeframe.max(), self.num_steps())
        # smooth_returns = interp1d(timeframe, returns, kind='cubic')  
        # returns = smooth_returns(newtimeframe)  
        spl = make_interp_spline(timeframe, data, k=3)  # type: BSpline
        data = spl(newtimeframe)
        timeframe = newtimeframe
        return timeframe, data 
 
    def df_episodes_energy(self, smooth=False):
        data = self.get_etankmin()
        timeframe = np.arange(self.num_episodes())
        if smooth:
            timeframe, data = self._smooth(timeframe,data)
        return pd.DataFrame(dict(Episodes = timeframe, Energy = data))  
  
    def df_runs_episodes_energy(self, folder_path, smooth=False): 
        comb_df = pd.DataFrame()  
        for file_name in os.listdir(folder_path):  
            name, ext = os.path.splitext(file_name)   
            if name.startswith("energy_") and ext==".json": 
                file_path = os.path.join(folder_path,file_name)
                self.load(file_path)
                df = self.df_episodes_energy(smooth)
                df["Runs"] = [name]*len(df["Episodes"])
                comb_df = pd.concat([comb_df, df])
        return comb_df

