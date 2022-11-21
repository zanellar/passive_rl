
import os 
import time
import pandas as pd
import json
from mjrlenvs.scripts.cbs.logcbs import BaseLogCallback
 
class ErrorsCallback(BaseLogCallback):
    def __init__(self, save_all=False ): 
        super().__init__(prefix="errors")  
        self.save_all = save_all 
        self.err_pos_tot_episode = 0   
        self.err_pos = 0   

    def _on_training_start(self) -> None:
        self.episodes_data = dict(err_pos=[], cos_pos=[], tanh_vel=[])
        self.training_data = dict(episodes_data=[], err_pos_tot_episode=[],err_pos_final_episode=[])
        self.rollouts = 0    
        return True

    def _on_rollout_start(self) -> None:  
        return True

    def _on_step(self) -> bool: 
        sample = self.training_env.env_method("get_sample") 
        obs, action, reward, done, info = sample[0] 

        if done: 
            return True


        sin_pos = obs[0] 
        cos_pos = obs[1] 
        tanh_vel = obs[2]  
        self.err_pos = abs(1. - sin_pos)
        self.err_pos_tot_episode += self.err_pos 
  
        if self.save_all:
            self.episodes_data["err_pos"].append(self.err_pos)
            self.episodes_data["cos_pos"].append(cos_pos)  
            self.episodes_data["tanh_vel"].append(tanh_vel)  
        return True
    
    def _on_rollout_end(self) -> None:  
        self.rollouts += 1 

        if self._is_episode_end(): 
            self.logger.record('errors/pos_tot_episode', self.err_pos_tot_episode)  # -> tensorboard 
            self.training_data["err_pos_tot_episode"].append(self.err_pos_tot_episode)   
            self.logger.record('errors/pos_final_episode', self.err_pos)  # -> tensorboard 
            self.training_data["err_pos_final_episode"].append(self.err_pos)   
            if self.save_all:
                self.training_data["episodes_data"].append(self.episodes_data)   
            self.err_pos_tot_episode = 0
        return True

    def _on_training_end(self) -> bool:   
        with open(self.file_path, 'w') as f:
            json.dump(self.training_data, f)
 
          