import tree_sim_python as ts
import numpy as np
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

INIT_TREE_POWER = 10.0
MAX_STEPS = 700
EVELUATION_POP_SIZE = 10

rs = ts.RandomStratagy()
TREE_INDEX = 0
scores = []

def create_dummy_model(inputs,outputs):
    model = create_neural_network_model(input_size=inputs.shape[1], output_size=outputs.shape[1])
    return model
 
def create_neural_network_model(input_size, output_size):
    network = input_data(shape=[None, input_size], name='input')
    network = tflearn.fully_connected(network, 32)
    network = tflearn.fully_connected(network, 32)
    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, name='targets')
    model = tflearn.DNN(network, tensorboard_dir='tflearn_logs')
 
    return model

training_data = np.loadtxt("training_data.txt")
outputs = np.reshape(training_data[:,3],(len(training_data),1))
mask =  np.ones(training_data.shape[1],dtype=bool)
mask[3]=False
inputs = training_data[:,mask]
model = create_dummy_model(inputs,outputs)

model = model.load('tree_trained.tflearn')
print(model)
for _ in range(EVELUATION_POP_SIZE):
    sim = rs.createRandomTreeSim(ts.Vec3i(15,15,0),10)
    sim.setLogTree(0)
    
    for _ in range(MAX_STEPS):
        cell = sim.getRandomCell(TREE_INDEX)
        current_ob = np.reshape(cell.getObservationNoMove(sim.model),(-1,10))
        print(current_ob)
        move = model.predict(current_ob)
        print()
        if(not sim.step(sim.getObervation(TREE_INDEX)),move,cell):
            #print("Died!")
            break
    ts.drawFrame(_,sim)
    input("Press any key to continue...")
    scores.append( sim.getPowerLog()[-1] )
scores = np.array(scores)
average_score =np.mean(scores)
plt.hist(scores)
print("Average Score: ",average_score)
plt.show()
"""

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
"""