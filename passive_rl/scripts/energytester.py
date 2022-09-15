import os 
from ast import Try
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy   
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from passive_rl.scripts.pkgpaths import PkgPath
from mjrlenvs.scripts.env.envutils import wrapenv 

from passive_rl.scripts.plotenergydata import PlotEnergyData  
 
 

class TestEbudAgent():

    def __init__(self, run_args, rendering=None) -> None:

        self.args = run_args 
        
        self.env = self.args.ENV_EVAL 
        self.env = wrapenv(self.env, self.args, load_norm_env = True)  

        self.callback_list = CallbackList([])
        self.cb = None

        ###### INPUT FOLDERS PATH
        out_train_folder = PkgPath.OUT_TRAIN_FOLDER if self.args.OUT_TRAIN_FOLDER is None else self.args.OUT_TRAIN_FOLDER  
        self.training_output_folder_path = os.path.join(out_train_folder,f"{self.args.ENVIRONMENT}/{self.args.RUN_ID}")

        print(f"Loading logs from: {self.training_output_folder_path}")
        self.saved_energy_logs_path = os.path.join(self.training_output_folder_path,"logs") 

        ###### OUTPUT FOLDERS PATH
        out_test_folder = PkgPath.OUT_TEST_FOLDER if self.args.OUT_TEST_FOLDER is None else self.args.OUT_TEST_FOLDER
        self.testing_output_folder_path = os.path.join(out_test_folder,f"{self.args.ENVIRONMENT}/{self.args.RUN_ID}") 


        self.save_testing_evals_path = os.path.join(self.testing_output_folder_path,"evals")
        if not os.path.exists(self.save_testing_evals_path):
            os.makedirs(self.save_testing_evals_path)

        self.save_testing_plots_path = os.path.join(self.testing_output_folder_path,"plots")
        if not os.path.exists(self.save_testing_plots_path):
            os.makedirs(self.save_testing_plots_path)
   

    def plot(self, model_id=None, y="energy",save=True, show=True, save_name=None): 
        if model_id is not None:
            file_name = f"energy_{model_id}.json"
        else:
            file_name = None    
        plotter = PlotEnergyData(self.saved_energy_logs_path, file_name=file_name)    
        if save_name is None:
            save_name = os.path.join(self.save_testing_plots_path,f"{self.args.RUN_ID}.pdf")
        plotter.plt_emin(save=save, show=show, save_name=save_name) 
 



 