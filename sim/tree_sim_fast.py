from numpy.lib.function_base import kaiser
from numpy.lib.histograms import histogram
import tree_simulator as ts
import numpy as np
import tree_render as tr

def mean_filter(values,size):
    dx = []
    for i in range(len(values))[size:-size]:
        n = 0
        for k in range(i-size,i+1+size):
            n+=values[k]
        dx.append(n/size)
    return dx
    

def derivitive(values):
    dx = []
    for i in range(len(values))[1:-1]:
        dx.append( (values[i+1] - values[i-1])/2 )
    return dx

MAX_STEPS = 700
POWER_TARGET = 0
TARGET_MIN_LIFE = MAX_STEPS

rs = ts.RandomStratagy()
fig,axs = tr.plt.subplots(3)

max_powers = []

POP_SIZE = 10000
observations_per_game = []
moves_per_game= []
for n in range(POP_SIZE):
    energies = None
    powers = None
    while(energies==None or len(energies)<TARGET_MIN_LIFE or powers[-1]<POWER_TARGET ):
        sim = rs.createRandomTreeSim(ts.Vec3i(15,15,0),10)
        sim.setLogTree(0)
        
        for _ in range(MAX_STEPS):
            if(not sim.step(sim.getObervation(0))):
                #print("Died!")
                break

        energies = sim.getEnergyLog()
        powers = sim.getPowerLog()
    max_powers.append(powers[-1])

    if(powers[-1]>80):
        observations = np.array(sim.getObervationLog())
        moves_per_game+=list(observations[:,3])
        observations_per_game+=list(observations)
        """tr.renderModel(sim)
        axs[0].plot(energies)
        axs[1].plot(powers)
        axs[2].plot(mean_filter(derivitive(mean_filter(powers,25)),11))
        tr.plt.show()"""
        

    if(n%int(POP_SIZE/100)==0):
        print("{}%\r".format(100*n/POP_SIZE))





move_distabution = np.histogram(moves_per_game,bins=np.arange(0,19,1) )[0]/len(moves_per_game)
print(move_distabution)

print("Done!")
#model = sim.getModel()
#print(model,sim.MODEL_X_MAX(),sim.MODEL_Y_MAX(),sim.MODEL_Z_MAX())
#tr.renderModel(sim)
#tr.plt.show()


#fig,axs = tr.plt.subplots(3)

np.savetxt("../training_data_2.txt",np.array(observations_per_game))

power_hist = histogram(max_powers,bins=80)
mean_power = power_hist[1][:-1][power_hist[0]==np.max(power_hist[0])][0]
print(mean_power)

axs[0].hist(max_powers,bins=80)
#axs[0].plot(energies)
axs[1].plot(powers)
axs[2].plot(mean_filter(derivitive(mean_filter(powers,15)),11))
tr.plt.show()