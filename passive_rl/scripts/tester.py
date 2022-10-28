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
    
    def eval_emin_model(self, model_id="random", n_eval_episodes=30, render=False, save=False): 
        self._loadmodel(model_id) 
        obs = self.env.reset() 
        emin = None
        emin_list = []
        i = 0
        while i<=n_eval_episodes: 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.env.step(action)  
            energy = info[0]["energy_tank"]
            emin = min(energy,emin) if emin is not None else energy 
            if render:
                self.env.render() # BUG not working cam selection
            if done:
                i +=1 
                obs = self.env.reset()
                emin_list.append(emin)
                emin = None 
        
        if save:
            file_path =  os.path.join(self.testing_output_folder_path, f"{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(emin_list)  

        return emin_list 

    def eval_emin_run(self, n_eval_episodes=30, render=False, save=False, plot=False, addname=""):  
        data = {}
        run_training_logs_folder_path = os.path.join(self.training_output_folder_path,"logs")
        emin_full_list = []
        for file_name in os.listdir(run_training_logs_folder_path):  
            name = os.path.splitext(file_name)[0]
            prefix, model_id = name.split(sep="_", maxsplit=1)
            if prefix == "energy":  
                print(f"Evaluating {model_id}")
                emin_model_list = self.eval_emin_model(model_id=model_id, n_eval_episodes=n_eval_episodes, render=render, save=False) 
                emin_full_list += emin_model_list
                data[model_id] = emin_model_list
        if plot:
            pass #TODO statannotation 
        if save:  
            file_path =  os.path.join(self.testing_output_folder_path, "energy_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data, f) 
         
        return emin_full_list

    ##################################################################################################################################

    def eval_err_model(self, model_id="random", n_eval_episodes=30, render=False, save=False): 
        self._loadmodel(model_id) 
        obs = self.env.reset() 
        episode_err = 0
        err_list = []
        i = 0
        while i<=n_eval_episodes: 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.env.step(action)   
            sin_pos = obs[0] 
            cos_pos = obs[1] 
            tanh_vel = obs[2]   
            episode_err += abs(1. - sin_pos)
            if render:
                self.env.render() # BUG not working cam selection
            if done:
                i +=1 
                obs = self.env.reset()
                err_list.append(episode_err)
                episode_err = 0 
        
        if save:
            file_path =  os.path.join(self.testing_output_folder_path, f"{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(err_list)  

        return err_list 

    def eval_err_run(self, n_eval_episodes=30, render=False, save=False, plot=False, addname=""):  
        data = {}
        run_training_logs_folder_path = os.path.join(self.training_output_folder_path,"logs")
        err_full_list = []
        for file_name in os.listdir(run_training_logs_folder_path):  
            name = os.path.splitext(file_name)[0]
            prefix, model_id = name.split(sep="_", maxsplit=1)
            if prefix == "errors":  
                print(f"Evaluating {model_id}")
                err_model_list = self.eval_err_model(model_id=model_id, n_eval_episodes=n_eval_episodes, render=render, save=False) 
                err_full_list += err_model_list
                data[model_id] = err_model_list
        if plot:
            pass #TODO statannotation 
        if save:  
            file_path =  os.path.join(self.testing_output_folder_path, "errors_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data, f) 
         
        return err_full_list

     
     