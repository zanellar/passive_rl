from cProfile import run
import os  
from passive_rl.scripts.datautils import *  
from passive_rl.scripts.pkgpaths import PkgPath

from mjrlenvs.scripts.plot.plotutils import mean_std_plt,stat_boxes_plt

from mjrlenvs.scripts.plot.plotter import Plotter

class PlotterEBud(Plotter):

    def __init__(self, out_train_folder=None, out_test_folder=None, plot_folder=None) -> None:
        super().__init__(out_train_folder=out_train_folder, out_test_folder=out_test_folder, plot_folder=plot_folder)
    

    def plot_avg_train_run_energy(self, env_name, run_id, save=True, show=True, select="all"):   

        self.training_output_folder_path = os.path.join(self.out_train_folder, env_name, run_id)

        if select == "min" or select == "all": 
            mean_std_plt(
                data = df_run_episodes_energy(run_folder_path=self.training_output_folder_path, smooth=True, etype="min") , 
                title = 'Average Train Episode Min Tank Level',
                xaxis = "Episodes", 
                value = "Energy",  
                estimator = np.mean,
                save = save, 
                show = show, 
                save_path = os.path.join(self.save_training_plots_folder_path, f"emin_train_{run_id}.pdf") 
            ) 

        if select == "max" or select == "all": 
            mean_std_plt(
                data = df_run_episodes_energy(run_folder_path=self.training_output_folder_path, smooth=True, etype="max") , 
                title = 'Average Train Episode Max Tank Level',
                xaxis = "Episodes", 
                value = "Energy",  
                estimator = np.mean,
                save = save, 
                show = show, 
                save_path = os.path.join(self.save_training_plots_folder_path, f"emax_train_{run_id}.pdf") 
            )  

        if select == "final" or select == "all": 
            mean_std_plt(
                data = df_run_episodes_energy(run_folder_path=self.training_output_folder_path, smooth=True, etype="final") , 
                title = 'Average Train Episode Final Tank Level',
                xaxis = "Episodes", 
                value = "Energy",  
                estimator = np.mean,
                save = save, 
                show = show, 
                save_path = os.path.join(self.save_training_plots_folder_path, f"efinal_train_{run_id}.pdf") 
            )   

    def plot_avg_train_multirun_energy(self, env_run_ids, save=True, show=True, plot_name=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]
        mean_std_plt(
            data = df_multiruns_episodes_energy(run_paths_list=run_paths_list, smooth=True, etype="min"), 
            title = 'Average Train Episode Min Tank Level Multirun',
            xaxis = "Episodes", 
            value = "Energy",  
            hue = "Runs",
            estimator = np.mean,
            save = save, 
            show = show, 
            save_path = os.path.join(self.save_multirun_training_plots_path, f"emin_train_multirun_{plot_name}.pdf")
        )
 

    # def plot_avg_test_run_energy(self, env_name, run_id, save=True, show=True ):   
    #     self.testing_output_folder_path = os.path.join(self.out_test_folder, env_name, run_id) 
    #     #TODO
 
    def plot_boxes_test_multirun_energy(self, env_run_ids, save=True, show=True, plot_name=None): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        stat_boxes_plt(
            data = df_test_multirun_energy(run_paths_list=run_paths_list), 
            title = 'Test Min Tank Level Multirun',
            xaxis = "Runs", 
            value = "Energy",  
            show  = show, 
            save = save, 
            save_path = os.path.join(self.save_multirun_training_plots_path, f"emin_test_multirun_{plot_name}.pdf") 
        )
