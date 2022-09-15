# Passive RL #
Framework to train a RL agent with passivity 

## TO DO 

- [ ] Find Agent's parameters for environment in gym-mujoco
- [ ] Test adding obstacles in the xml envs
- [ ] train a manipulator like UR5 or Panda  

## Installation ##

Setup virtual environment

```
conda env create -f rl.yml
```


Add the following line to .bashrc

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kvn/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```


```
pip install -e .
```


## Mujoco ##

### Contact Forces bug
Issue: https://github.com/openai/mujoco-py/pull/487
Need to modify 'mujoco_py/mjviewer.py' in the conda env as follow:
https://github.com/openai/mujoco-py/pull/487/commits/ab026c1ff8df54841a549cfd39374b312e8f00dd

## Nota Bene
 
if action is velocity we can get the torque as

'torque = (action - self.action)/self._env.dt'

or with a buffer: https://github.com/ARISE-Initiative/robosuite/blob/874ce964640f66440a695582a1375df1aff247ac/robosuite/robots/single_arm.py#L271
