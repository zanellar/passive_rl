
import os 
import time
import pandas as pd
import json
from mjrlenvs.scripts.cbs.logcbs import BaseLogCallback
 
class SaveEnergyLogsCallback(BaseLogCallback):
    def __init__(self, save_all=False ): 
        super().__init__(prefix="energy")  
        self.save_all = save_all 
        self.energy_tank_min = None  
        self.energy_tank_max = None   

    def _on_training_start(self) -> None:
        self.episodes_data = dict(energy_tank=[], energy_exchanged=[])
        self.training_data = dict(episodes_data=[], energy_tank_min=[], energy_tank_max=[], energy_tank_final=[], num_ep_steps=[], num_tot_steps=[])
        self.rollouts = 0   
        self.num_ep_steps = 0
        return True

    def _on_rollout_start(self) -> None:  
        return True

    def _on_step(self) -> bool: 
        sample = self.training_env.env_method("get_sample") 
        _, _, _, _, info = sample[0]

        energy_tank = info["energy_tank"]
        energy_exchanged = info["energy_exchanged"]

        if self.energy_tank_min is None:
            self.energy_tank_min = energy_tank 
        self.energy_tank_min = min(self.energy_tank_min, energy_tank)

        if self.energy_tank_max is None:
            self.energy_tank_max = energy_tank
        self.energy_tank_max = max(self.energy_tank_max, energy_tank)

        if self.save_all:
            self.episodes_data["energy_tank"].append(energy_tank)
            self.episodes_data["energy_exchanged"].append(energy_exchanged)  
        return True
    
    def _on_rollout_end(self) -> None:  
        self.rollouts += 1

        sample = self.training_env.env_method("get_sample") 
        _, _, _, _, info = sample[0] 
        energy_tank = info["energy_tank"]

        if self._is_episode_end():
            self.training_data["energy_tank_min"].append(self.energy_tank_min) 
            self.training_data["energy_tank_max"].append(self.energy_tank_max) 
            self.training_data["energy_tank_final"].append(energy_tank) 
            self.training_data["num_ep_steps"]= self.num_timesteps-self.num_ep_steps
            self.num_ep_steps = self.num_timesteps
            if self.save_all:
                self.training_data["episodes_data"].append(self.episodes_data)  
            self.episodes += 1 
            self.energy_tank_min = None  
            self.energy_tank_max = None 
        return True

    def _on_training_end(self) -> bool:  
        self.training_data["num_tot_steps"]= self.num_timesteps
        with open(self.file_path, 'w') as f:
            json.dump(self.training_data, f)
 
          