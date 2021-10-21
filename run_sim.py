import tree_sim as ts
import numpy as np
import matplotlib.pyplot as plt

sim = ts.TreeSim()
sim.createTree(np.array([16,16,0],dtype=int),ts.random_stategy,10.0)
MAX_VALUE=6000
ts.drawFrame(0,sim,path="test_output")
energy=[]
power=[]
values=[]
for i in range(1,MAX_VALUE):
    energy.append(sim.trees[0].energy)
    values.append(i)
    power.append(sim.trees[0].power)
    if(not sim.step([])):
        print("Tree Died at:",i)
        break
plt.close()
fig, axs = plt.subplots(2)
axs[0].plot(values,energy)
axs[1].plot(values,power)
ts.drawFrame(i,sim,path="test_output")
print("Frame {} Done".format(i))
print("Done!") 
plt.show()