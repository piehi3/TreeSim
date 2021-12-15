import numpy as np
import matplotlib.pyplot as plt

def addCubeToRender(cell,colors,filled):
    colors[cell.pos[0],cell.pos[1],cell.pos[2]]=cell.getColor()
    filled[cell.pos[0],cell.pos[1],cell.pos[2]]=True

def renderTree(tree,colors,filled):
    for cell in tree.cells:
        addCubeToRender(cell,colors,filled)

def getColor(kind):
    if(kind==0):
        return np.array([0.0,1.0,0.0])
    elif(kind==1):
        return np.array([78/255,53/255,36/255])
    else:
        return np.array([0,0,0])

def renderModel(sim):
    colors = np.zeros((sim.MODEL_X_MAX(), sim.MODEL_Y_MAX(), sim.MODEL_Z_MAX(),3))
    filled = np.zeros((sim.MODEL_X_MAX(), sim.MODEL_Y_MAX(), sim.MODEL_Z_MAX()), dtype=bool)
    
    model = sim.getModel()
    for x in range(sim.MODEL_X_MAX()):
        for y in range(sim.MODEL_Y_MAX()):
            for z in range(sim.MODEL_Z_MAX()):
                colors[x][y][z] = getColor(model[x][y][z])
                filled[x][y][z] = model[x][y][z] >= 0
    

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.voxels(filled,facecolors=colors,  edgecolors='k')

def drawFrame(i,sim,path="output"):
    colors = np.zeros((sim.MODEL_MAX_X, sim.MODEL_MAX_Y, sim.MODEL_MAX_Z,3))
    filled = np.zeros((sim.MODEL_MAX_X, sim.MODEL_MAX_Y, sim.MODEL_MAX_Z), dtype=bool)

    for tree in sim.trees:
        renderTree(tree,colors,filled)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.voxels(filled,facecolors=colors,  edgecolors='k')

    plt.savefig(path+"/{}.png".format(i))
    #plt.show()