
import os
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from mjrlenvs.scripts.eval.plotdata import mean_std_plt
from passive_rl.scripts.logenergydata import LogEnergyData


class PlotEnergyData():
 

    def __init__(self, logs_folder_path, file_name=None) -> None:  
        self.logs_folder_path =  logs_folder_path 
        self.data = LogEnergyData()    
        if file_name is not None: 
            self.data.load(file_path=os.path.join(logs_folder_path,file_name))
     
    
    def plt_emin(self, save=True, show=True, save_name="plot.pdf", smooth=True):  
        
        df_logs = self.data.df_runs_episodes_energy(self.logs_folder_path, smooth)

        plt.figure()
        
        mean_std_plt(
            df_logs, 
            title='Average Episode Tank Level',
            xaxis="Episodes", 
            value="Energy",  
            estimator=np.mean
        )

        if save:
            plt.savefig(save_name, bbox_inches='tight', format="pdf") 
        if show: 
            plt.show()  
 
