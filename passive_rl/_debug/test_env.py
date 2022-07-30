from tkinter import W
import mujoco_py
import os
import math

'''
Test of the environment using `mujoco-py`
'''

# mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'xmls', 'ur5.xml')
xml_path = '/home/riccardo/projects/passive_rl/data/xml/reacher_test.xml'

# model
model = mujoco_py.load_model_from_path(xml_path)
# simulator
sim = mujoco_py.MjSim(model)

# simulate 1 step
sim.step()
 

# render
viewer = mujoco_py.MjViewer(sim)

for i in range(10000000):
    viewer.render()  
    sim.step()  
 