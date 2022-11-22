 
import os  
import json
from traceback import print_tb 
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
    
    def eval_model(self, model_id="random", n_eval_episodes=30, cumulative_error=False, render=False, save=False): 
        self._loadmodel(model_id) 
        obs = self.env.reset() 
        energy = []
        emin_list = []
        episode_err = 0
        err_list = []
        returns_list = []
        episode_return = 0
        i = 0 
        while i<n_eval_episodes: 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action) 

            energy_tank = info[0]["energy_tank"]   
 
            if done:

                obs = self.env.reset()
                returns_list.append(episode_return) 
                episode_return = 0 
                err_list.append(episode_err)
                episode_err = 0  
                emin_list.append(min(energy) if len(energy)>0 else energy_tank)
                energy = [] 
                i +=1  

            else:       

                # return
                episode_return += reward.item()
    
                # position error 
                if self.args.NORMALIZE_ENV is not None:
                    unnormalized_obs = self.env.get_original_obs()
                else:
                    unnormalized_obs = obs

                sin_pos = unnormalized_obs[0][0]  
                position_error = abs(1. - sin_pos) 

                if cumulative_error:
                    episode_err += position_error
                else:
                    episode_err = position_error

                # minimal energy in tank 
                energy.append(energy_tank)

            if render:
                self.env.render() # BUG not working cam selection
  
        if save:
            file_path =  os.path.join(self.testing_output_folder_path, f"returns_{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(returns_list)  
            file_path =  os.path.join(self.testing_output_folder_path, f"energy_{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(emin_list)  
            file_path =  os.path.join(self.testing_output_folder_path, f"errors_{model_id}.txt") 
            with open(file_path, 'w') as file:  
                file.write(err_list)    

        return dict(emin=emin_list, err=err_list, ret=returns_list)

    def eval_run(self, n_eval_episodes=30, render=False, save=False, plot=False, addname="", cumulative_error=False):  
        data_emin = {}
        data_err = {}
        data_ret = {}
        run_training_logs_folder_path = os.path.join(self.training_output_folder_path,"logs")
        run_eval_emindata = []
        run_eval_errdata = []
        run_eval_retdata = []
        for file_name in os.listdir(run_training_logs_folder_path):  
            name = os.path.splitext(file_name)[0]
            prefix, model_id = name.split(sep="_", maxsplit=1)
            if prefix == "log":  
                print(f"Evaluating {model_id}")
                model_eval_data = self.eval_model(model_id=model_id, n_eval_episodes=n_eval_episodes, render=render, cumulative_error=cumulative_error, save=False) 
                run_eval_retdata += model_eval_data["ret"]  
                run_eval_emindata += model_eval_data["emin"] 
                run_eval_errdata += model_eval_data["err"]
                data_ret[model_id] = model_eval_data["ret"]  
                data_emin[model_id] = model_eval_data["emin"] 
                data_err[model_id] = model_eval_data["err"]

        if plot:
            pass #TODO   

        if save:  
            file_path =  os.path.join(self.testing_output_folder_path, "returns_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data_ret, f) 

            file_path =  os.path.join(self.testing_output_folder_path, "energy_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data_emin, f) 

            file_path =  os.path.join(self.testing_output_folder_path, "errors_eval_run.json") 
            with open(file_path, 'w') as f:
                json.dump(data_err, f) 
         
        return dict(emin=run_eval_emindata, err=run_eval_errdata, ret=run_eval_retdata)
 