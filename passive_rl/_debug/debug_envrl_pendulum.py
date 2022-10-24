 
from cProfile import label
from cgi import print_directory
from passive_rl.envs.pendulum.pendulum import PendulumEBud
from passive_rl.scripts.pkgpaths import PkgPath 
import matplotlib.pyplot as plt

env = PendulumEBud(
  max_episode_length=500, 
  energy_tank_init=1000, 
  energy_tank_threshold=0, 
  init_joint_config=[-1.57], 
  debug=False, 
  energy_terminate=False, 
  folder_path = PkgPath.ENV_DESC_FOLDER,
  env_name ="pendulum_f0"
) 

obs = env.reset()
env.render() 
input("[enter]")

energy_tank_f0 = []
energy_exchanged_f0 = []
posf0 = []
for i in range(500):
    action = [0.3]
    obs, reward, done, info = env.step(action)
    # env.render() 
    energy_tank_f0 += [info["energy_tank"]] 
    energy_exchanged_f0 += [info["energy_exchanged"]] 
    posf0 += [env.get_joints()] 
    if done: 
      obs = env.reset() 
      
env = PendulumEBud(
  max_episode_length=500, 
  energy_tank_init=1000, 
  energy_tank_threshold=0, 
  init_joint_config=[-1.57], 
  debug=False, 
  energy_terminate=False, 
  folder_path = PkgPath.ENV_DESC_FOLDER,
  env_name ="pendulum_f1"
) 

obs = env.reset()
env.render() 
# input("[enter]")
energy_tank_f1 = []
energy_exchanged_f1 = []
posf1 = []
for i in range(500):
    action = [0.3]
    obs, reward, done, info = env.step(action)
    # env.render() 
    energy_tank_f1 += [info["energy_tank"]]
    energy_exchanged_f1 += [info["energy_exchanged"]] 
    posf1 += [env.get_joints()] 
    print(info["energy_exchanged"],info["energy_tank"])
    if done: 
      obs = env.reset() 
      
env.close() 

plt.figure(0)
plt.plot(range(500),posf1,label="friction = 1") 
plt.plot(range(500),posf0,label="friction = 0") 
plt.legend()
plt.show()

plt.figure(1) 
plt.plot(range(500),energy_tank_f0,label="friction = 0") 
plt.plot(range(500),energy_tank_f1,label="friction = 1")
plt.legend()
plt.show()

plt.figure(2)
# plt.plot(range(500),energy_exchanged_f0,label="friction = 0") 
# plt.plot(range(500),energy_exchanged_f1,label="friction = 1")
plt.legend()
plt.show()