import os 
from ast import Try
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy   
from stable_baselines3.common.callbacks import CallbackList, BaseCallback 
from passive_rl.scripts.pkgpaths import PkgPath
from mjrlenvs.scripts.env.envutils import wrapenv 

from passive_rl.scripts.plotenergydata import *  
 
 

class TestRunEBud():

    def __init__(self, run_args, rendering=None) -> None:

        self.args = run_args 
        
        self.env = self.args.ENV_EVAL 
        self.env = wrapenv(self.env, self.args, load_norm_env = True)  

        self.callback_list = CallbackList([])
        self.cb = None

        ###### INPUT FOLDERS PATH
        out_train_folder = PkgPath.OUT_TRAIN_FOLDER if self.args.OUT_TRAIN_FOLDER is None else self.args.OUT_TRAIN_FOLDER  
        self.training_output_folder_path = os.path.join(out_train_folder, self.args.ENVIRONMENT, self.args.RUN_ID)

        ###### OUTPUT FOLDERS PATH
        out_test_folder = PkgPath.OUT_TEST_FOLDER if self.args.OUT_TEST_FOLDER is None else self.args.OUT_TEST_FOLDER
        self.testing_output_folder_path = os.path.join(out_test_folder, self.args.ENVIRONMENT, self.args.RUN_ID) 


        self.save_testing_evals_path = os.path.join(self.testing_output_folder_path,"evals")
        if not os.path.exists(self.save_testing_evals_path):
            os.makedirs(self.save_testing_evals_path)

        self.save_testing_plots_path = os.path.join(self.testing_output_folder_path,"plots")
        if not os.path.exists(self.save_testing_plots_path):
            os.makedirs(self.save_testing_plots_path)
   

    def plot(self, save=True, show=True ):   
        print("Plotting Average MIN Energy Episode")
        plt_run_emin(
            run_folder_path = self.training_output_folder_path,
            save = save, 
            show = show, 
            save_path = os.path.join(self.save_testing_plots_path, f"emin_{self.args.RUN_ID}.pdf") 
        ) 

        print("Plotting Average MAX Energy Episode")
        plt_run_emax(
            run_folder_path = self.training_output_folder_path,
            save = save, 
            show = show, 
            save_path = os.path.join(self.save_testing_plots_path, f"emax_{self.args.RUN_ID}.pdf") 
        ) 

        print("Plotting Average FINAL Energy Episode")
        plt_run_efinal(
            run_folder_path = self.training_output_folder_path,
            save = save, 
            show = show, 
            save_path = os.path.join(self.save_testing_plots_path, f"efinal_{self.args.RUN_ID}.pdf") 
        ) 
 



class TestMultiRunEBud():

    def __init__(self, out_train_folder = None, out_test_folder = None) -> None:
        self.out_train_folder = PkgPath.OUT_TRAIN_FOLDER if out_train_folder is None else out_train_folder  
        self.out_test_folder = PkgPath.OUT_TEST_FOLDER if out_test_folder is None else out_test_folder  
        self.save_testing_plots_path = os.path.join(self.out_test_folder, "multirun", "plots")
        if not os.path.exists(self.save_testing_plots_path):
            os.makedirs(self.save_testing_plots_path)

    def evalmultirun(self):
        pass

    def plot(self, env_run_ids, save=True, show=True, plot_name=None):
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        env_run_paths = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]
        plt_multirun_emin(
            env_run_paths, 
            show  = show, 
            save = save, 
            save_path = os.path.join(self.save_testing_plots_path, f"emin_multirun_{plot_name}.pdf"), 
            smooth=True
        )