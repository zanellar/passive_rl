 
from ntpath import join
import numpy as np 
import gym
from passive_rl.scripts.ebudenv import EBudBaseEnv,EBudAwEnv
 
from mjrlenvs.scripts.env.envgymbase import EnvGymBase


# class HalfCheetah(EnvGymBase): 

#     def __init__(self):
#         super(HalfCheetah, self).__init__()
    
#         self.env = gym.make('HalfCheetah-v3')  
#         self.observation_space = self.env.observation_space # BUG 
#         self.action_space = self.env.action_space # BUG 
  
#     def step(self, action):
#         self.action = action  
#         _obs, _reward, _done, _info = self.env.step(action) 
#         return _obs, max(0, _reward), _done, _info
    
#     def reset(self ):  
#         return self.env.reset() 

#     def render(self ): 
#         self.env.render()

#     def get_joints(self):
#         return np.array(self.env.sim.get_state()[1][-6:]) 


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
            non_neg_reward = True,
            reward_offset = 5
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
            