 
from ntpath import join
import numpy as np 
import gym
from passive_rl.scripts.ebudenv import EBudBaseEnv,EBudAwEnv
 
from mjrlenvs.scripts.env.envgymbase import EnvGymBase
 

class HalfCheetahEBud(EBudBaseEnv): 

    def __init__(self,     
                energy_tank_init = 10, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank    
                debug = False,
                energy_terminate = False  
                ):
        super(HalfCheetahEBud, self).__init__(
            env = gym.make('HalfCheetah-v3')  ,
            energy_tank_init = energy_tank_init, # initial energy in the tank
            energy_tank_threshold = energy_tank_threshold, # minimum energy in the tank  
            debug = debug,
            energy_terminate = energy_terminate,
            non_neg_reward = False,
            reward_offset = 0
        )
 
    def get_joints(self):  
        return np.array(self.env.sim.get_state()[1][-6:]) 

    def get_torques(self):
        return self.action
            


class HalfCheetahEBudAw(EBudAwEnv): 

    def __init__(self,     
                energy_tank_init = 10, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank    
                debug = False,
                energy_terminate = False 
                ):
        super(HalfCheetahEBudAw, self).__init__(
            env = gym.make('HalfCheetah-v2'),
            energy_tank_init = energy_tank_init, # initial energy in the tank
            energy_tank_threshold = energy_tank_threshold, # minimum energy in the tank  
            debug = debug,
            energy_terminate = energy_terminate  
        )
 
    def get_joints(self):
        obs = self.env.get_obs()
        joints = np.arcsin(obs[0])
        return joints

    def get_torques(self):
        return self.action

    def upgrade_reward(self,reward):  
        if self.reward_id == 0:
            new_reward = reward - 0.5*(self.energy_tank_init-self.energy_tank)
        elif self.reward_id == 1:
            new_reward = reward/(reward + 0.5*abs(self.energy_tank-self.energy_tank_init))
        return new_reward
            