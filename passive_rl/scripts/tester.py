 
import os  
import json
from traceback import print_tb
from imageio import save 
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from mjrlenvs.scripts.eval.tester import TestRun 
from passive_rl.scripts.pkgpaths import PkgPath  
 
 

class TestRunEBud(TestRun):

    def __init__(self, run_args, render=None, test_id="" ) -> None:
        super().__init__(run_args, render=render)  
        test_id = "_"+test_id if test_id != "" else test_id
        new_testing_output_folder_path = self.testing_output_folder_path + test_id  
        try:
            os.rename(src=self.testing_output_folder_path, dst=new_testing_output_folder_path)
        except:
            os.rmdir(self.testing_output_folder_path)
        self.testing_output_folder_path = new_testing_output_folder_path
    
    def eval_model(self, model_id="random", n_eval_episodes=30, cumulative_error=False, render=False, save=False): 
        self._loadmodel(model_id) 
        obs = self.env.reset() 
        energy_tank_list = []
        episode_energy = {}
        energy_exchanged_list = []
        etankmin_list = []
        episode_err = 0
        err_list = []
        returns_list = []
        episode_return = 0
        i = 0 
        
        while i<n_eval_episodes: 
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action) 

            energy_tank = info[0]["energy_tank"]    
            energy_exchanged = info[0]["energy_exchanged"]   
              
 
            if done:
                
                obs = self.env.reset()
                returns_list.append(episode_return) 
                episode_return = 0 
                err_list.append(episode_err)
                episode_err = 0  
                etankmin_list.append(min(energy_tank_list) if len(energy_tank_list)>0 else energy_tank)
                energy_tank_list = [] 
                episode_energy[str(i)] = energy_exchanged_list 
                energy_exchanged_list = []
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
                energy_tank_list.append(energy_tank)

                # energy exchanged during the episode
                energy_exchanged_list.append(energy_exchanged)

            if render:
                self.env.render() # BUG not working cam selection

            if save:  

                file_path =  os.path.join(self.testing_output_folder_path, "returns_eval.json") 
                with open(file_path, 'w') as f:
                    json.dump({model_id:returns_list}, f) 

                file_path =  os.path.join(self.testing_output_folder_path, "etank_eval.json") 
                with open(file_path, 'w') as f:
                    json.dump({model_id:etankmin_list}, f) 

                file_path =  os.path.join(self.testing_output_folder_path, "energy_eval.json") 
                with open(file_path, 'w') as f:
                    json.dump({model_id:episode_energy}, f)  

                file_path =  os.path.join(self.testing_output_folder_path, "errors_eval.json") 
                with open(file_path, 'w') as f:
                    json.dump({model_id:err_list}, f)  
    
        return dict(etankmin=etankmin_list, err=err_list, ret=returns_list, episode_energy = episode_energy)

    def eval_run(self, n_eval_episodes=30, render=False, save=False, plot=False, addname="", cumulative_error=False):  

        data_etankmin = {}
        data_energy = {}
        data_err = {}
        data_ret = {}

        run_training_logs_folder_path = os.path.join(self.training_output_folder_path,"logs")

        run_eval_etankmindata = []
        run_eval_errdata = []
        run_eval_retdata = []

        for file_name in os.listdir(run_training_logs_folder_path):  

            name = os.path.splitext(file_name)[0]
            prefix, model_id = name.split(sep="_", maxsplit=1)

            if prefix == "log":  
                print(f"Evaluating {model_id}")
                model_eval_data = self.eval_model(model_id=model_id, n_eval_episodes=n_eval_episodes, render=render, cumulative_error=cumulative_error ) 
                
                run_eval_retdata += model_eval_data["ret"]  
                run_eval_etankmindata += model_eval_data["etankmin"] 
                run_eval_errdata += model_eval_data["err"]

                data_ret[model_id] = model_eval_data["ret"]  
                data_etankmin[model_id] = model_eval_data["etankmin"] 
                data_energy[model_id] = model_eval_data["episode_energy"]

                data_err[model_id] = model_eval_data["err"]
 
        if save:  
            file_path =  os.path.join(self.testing_output_folder_path, "returns_eval.json") 
            with open(file_path, 'w') as f:
                json.dump(data_ret, f) 

            file_path =  os.path.join(self.testing_output_folder_path, "etank_eval.json") 
            with open(file_path, 'w') as f:
                json.dump(data_etankmin, f) 

            file_path =  os.path.join(self.testing_output_folder_path, "energy_eval.json") 
            with open(file_path, 'w') as f:
                json.dump(data_energy, f) 

            file_path =  os.path.join(self.testing_output_folder_path, "errors_eval.json") 
            with open(file_path, 'w') as f:
                json.dump(data_err, f) 
         
        return dict(etankmin=run_eval_etankmindata, err=run_eval_errdata, ret=run_eval_retdata)
 