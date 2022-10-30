from math import fabs
import os 
from ast import Try
from pickle import FALSE
from statistics import mean
import numpy as np 
import json
from stable_baselines3 import HER, SAC, TD3, DDPG    
from mjrlenvs.scripts.env.envutils import wrapenv 
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from mjrlenvs.scripts.eval.tester import TestRun 
from passive_rl.scripts.pkgpaths import PkgPath  
 
 

class TestRunEBud(TestRun):

    def __init__(self, run_args, render=None, test_id="" ) -> None:
        super().__init__(run_args, render=render)  
        test_id = "_"+test_id if test_id != "" else test_id
        new_testing_output_folder_path = self.testing_output_folder_path + test_id  
        os.rename(src=self.testing_output_folder_path, dst=new_testing_output_folder_path)
        self.testing_output_folder_path = new_testing_output_folder_path
    
    def eval_ebud_model(self, model_id="random", n_eval_episodes=30, render=False, save=False): 
        self._loadmodel(model_id) 
        obs = self.env.reset() 
        episode_emin = None
        emin_list = []
        episode_err = 0
        err_list = []
        i = 0
        while i<=n_eval_episodes: 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.env.step(action)    
            sin_pos = obs[0][0]  
            episode_err += abs(1. - sin_pos)
            energy = info[0]["energy_tank"]
            episode_emin = min(energy,episode_emin) if episode_emin is not None else energy 
            if render:
                self.env.render() # BUG not working cam selection
            if done:
                i +=1 
                obs = self.env.reset()
                err_list.append(episode_err)
                episode_err = 0 
                emin_list.append(episode_emin)
                episode_emin = None 
        
        if save:
            file_path =  os.path.join(self.testing_output_folder_path, f"energy_{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(emin_list)  
            file_path =  os.path.join(self.testing_output_folder_path, f"errors_{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(err_list)  

        return dict(emin=emin_list, err=err_list)

    def eval_ebud_run(self, n_eval_episodes=30, render=False, save=False, plot=False, addname=""):  
        data_emin = {}
        data_err = {}
        run_training_logs_folder_path = os.path.join(self.training_output_folder_path,"logs")
        run_eval_emindata = []
        run_eval_errdata = []
        for file_name in os.listdir(run_training_logs_folder_path):  
            name = os.path.splitext(file_name)[0]
            prefix, model_id = name.split(sep="_", maxsplit=1)
            if prefix == "log":  
                print(f"Evaluating {model_id}")
                model_eval_data = self.eval_ebud_model(model_id=model_id, n_eval_episodes=n_eval_episodes, render=render, save=False) 
                run_eval_emindata += model_eval_data["emin"]
                run_eval_errdata += model_eval_data["err"]
                data_emin[model_id] = run_eval_emindata
                data_err[model_id] = run_eval_errdata
        if plot:
            pass #TODO statannotation 
        if save:  
            file_path =  os.path.join(self.testing_output_folder_path, "energy_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data_emin, f) 
            file_path =  os.path.join(self.testing_output_folder_path, "errors_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data_err, f) 
         
        return dict(emin=data_emin, err=data_err)
 