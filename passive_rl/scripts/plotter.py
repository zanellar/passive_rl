 
import os  
import matplotlib.pyplot as plt
import seaborn as sns 

from passive_rl.scripts.datautils import *  
from passive_rl.scripts.pkgpaths import PkgPath 
from mjrlenvs.scripts.plot.plotter import Plotter

class PlotterEBud(Plotter):

    def __init__(self, out_train_folder=None, out_test_folder=None, plot_folder=None) -> None:
        super().__init__(out_train_folder=out_train_folder, out_test_folder=out_test_folder, plot_folder=plot_folder)
    

    def plot_avg_train_run_energy(self, env_name, run_id, labels=[], save=True, show=True, select="all"):   

        self.training_output_folder_path = os.path.join(self.out_train_folder, env_name, run_id)

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 

        if select == "min" or select == "all": 
            data = df_run_episodes_energy(run_folder_path=self.training_output_folder_path, smooth=True, etype="min") 
            save_path = os.path.join(self.save_training_plots_folder_path, f"emin_train_{run_id}.pdf")  
 
        if select == "max" or select == "all":  
            data = df_run_episodes_energy(run_folder_path=self.training_output_folder_path, smooth=True, etype="max") 
            save_path = os.path.join(self.save_training_plots_folder_path, f"emax_train_{run_id}.pdf")  

        if select == "final" or select == "all": 
            save_path = os.path.join(self.save_training_plots_folder_path, f"efinal_train_{run_id}.pdf")  
            data = df_run_episodes_energy(run_folder_path=self.training_output_folder_path, smooth=True, etype="final")   
        
        ax = sns.lineplot(
            data = data, 
            x =  "Episodes", 
            y = "Energy",  
            estimator = np.mean,
            errorbar='sd'
        )  

        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False, labels=labels)

        if save:
            plt.savefig(save_path, bbox_inches='tight', format="pdf") 
        if show: 
            plt.show()

    def plot_avg_train_multirun_energy(self, env_run_ids, labels=[], save=True, show=True, plot_name=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids))

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]

        data = df_multiruns_episodes_energy(run_paths_list=run_paths_list, smooth=True, etype="min") 
        save_path = os.path.join(self.save_multirun_training_plots_path, f"emin_train_multirun_{plot_name}.pdf")
 
        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 
        ax = sns.lineplot(
            data = data, 
            x =  "Episodes", 
            y = "Energy",  
            hue = "Runs",
            # estimator = np.mean,
            # errorbar='sd'
        )    
        
        # ax.set_xticklabels(range())
        # ax.set_xlabel('Steps') 
        # ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False, labels=labels)

        if save:
            plt.savefig(save_path, bbox_inches='tight', format="pdf") 
        if show: 
            plt.show()
 

    # def plot_avg_test_run_energy(self, env_name, run_id, save=True, show=True ):   
    #     self.testing_output_folder_path = os.path.join(self.out_test_folder, env_name, run_id) 
    #     #TODO
 
    def plot_stat_test_multirun_energy(self, env_run_ids, labels=[], save=True, show=True, plot_name=None, type="hist"): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_energy(run_paths_list=run_paths_list) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, f"emin_test_multirun_{plot_name}.pdf")

        plt.figure() 
        sns.set(style="darkgrid", font_scale=1.5) 
        if type == "hist":
            exit(data)
            ax = sns.histplot(
                data = data, 
                x = "Energy", 
                hue =  "Runs", 
                kde = True 
            )  
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=len(labels), frameon=False, labels=labels)

        if type == "box": 
            ax = sns.boxplot(
                data = data, 
                x = "Runs", 
                y = "Energy", 
                orient = "v"
            )
            ax.set_xticklabels(labels)
  
 
        if save:
            plt.savefig(save_path, bbox_inches='tight', format="pdf") 
        if show: 
            plt.tight_layout() 
            plt.show()
 
