from cmath import inf
from faulthandler import is_enabled
import numpy as np
import math 
import gym
# from gym.envs.mujoco.reacher import ReacherEnv 
from passive_rl.envs.reacher.reacher_base import ReacherEnv 
 
class ReacherEBud(gym.Env): 

    def __init__(self,    
                max_episode_length = 50, 
                energy_tank_init = 7, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                debug = False,
                mjmodel = "reacher_original"
                ):
        super(ReacherEBud, self).__init__()

        self.debug = debug
        self.t = 0
        self.T = max_episode_length

        ################# Init Learning Environment ####################

        # Vanilla Environment
        self._env = ReacherEnv(mjmodel)   
 
        # Observations 
        self.observation_space = self._env.observation_space 

        # Actions  
        self.action_space = self._env.action_space 

        # Vanilla State
        self.state = self._env._get_obs()  
        self.action = np.zeros(self._env.action_space.shape)

        ################# Init Energy-budgeting Framework ####################
 
        self.energy_tank_init = energy_tank_init
        self.energy_tank = energy_tank_init  
        self.energy_tank_threshold = energy_tank_threshold

        self.energy_joints = [0,0] # list containing the energy going out from the 2 motors
        self.energy_exchanged = 0 # initializing total energy going out at a time step to 0 
  
        self.energy_stop_ct = 0
    
    def _update_energy_tank(self, state_new):
        ''' 
        Etank(t) = Etank(t-1) - Eex(t)
        Ex(t) = sum(tau(t-1,i)*(q(t,i) - q(t-1,i)), i=1,2)
        '''
        
        joints_old = np.arccos(self.state[:2])
        joints = np.arccos(state_new[:2])   
        d_joints = joints - joints_old
        self.energy_joints = self.action*d_joints  
        self.energy_exchanged = sum(self.energy_joints)
        self.energy_tank -= self.energy_exchanged

        is_empty = self.energy_tank <= self.energy_tank_threshold
        return is_empty
  
    def step(self, action):  

        # Vanilla Environment Step
        state_new, _reward, _done, _info = self._env.step(action) 

        # Energy Budgeting  
        energy_done = self._update_energy_tank(state_new) 
 
        horizon_done = self.t>=self.T
        done = energy_done or _done or horizon_done

        if energy_done:
            self.energy_stop_ct += 1
         
        info = dict(energy_exchanged = self.energy_exchanged,
                    energy_tank = self.energy_tank, 
                    _info = _info)   

        self.action = action
        self.state = state_new 
        self.t += 1 

        # print(self.t, self.energy_tank, action,  self.energy_joints) 

        return state_new, _reward, done, info

    def reset(self, goal=None): 

        self._env.reset() 
        self.t = 0  
        self.energy_tank = self.energy_tank_init
        self.energy_exchanged = 0.0  
        return self.state 

    def render(self, mode=None): 
        self._env.render()

    def close(self):
        self._env.close()
          
    def seed(self, seed=None):
        return self._env.seed(seed)

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     return self._env.compute_reward(achieved_goal, desired_goal, info)