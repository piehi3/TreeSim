from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random as rand

from numpy.core.fromnumeric import shape, size

LEAF=0
BRANCH=1
KILL=9
ERROR_KIND = -1

LEAF_POWER=1.5
LEAD_BUILD=5.0
LEAF_MAINTAINE=0.5

BRANCH_BUILD=0.25
BRANCH_MAINTAINE=0.1

def getKindNear(model,cell,pos):
    p = cell.pos+pos
    if(np.any(p>model.shape)):
        return ERROR_KIND
    return model[p[0],p[1],p[2]]
    
class Tree(object): pass

class Cell(object):
    def __init__(self,tree,kind,opasity,pos,power_input,power_max_output,energy_to_build,color=[0.0,0.0,0.0]):
        super().__init__()
        self.kind = kind
        self.pos = pos
        self.alive=True
        self.power_input = power_input
        self.power_max_output = power_max_output
        self.opasity = opasity
        self.tree=tree
        self.energy_to_build = energy_to_build
        self.color = np.array(color)

    def getObservation(self,model):
        return [self.pos[0],self.pos[1],self.pos[2],
                getKindNear(model,self,np.array([1,0,0])),
                getKindNear(model,self,np.array([-1,0,0])),
                getKindNear(model,self,np.array([0,1,0])),
                getKindNear(model,self,np.array([0,-1,0])),
                getKindNear(model,self,np.array([0,0,1])),
                getKindNear(model,self,np.array([0,0,-1])),
                self.tree.power]

    def getPowerOutput(self,model):
        return 0

    def getPowerInput(self,model):
        return self.power_input

    def getPowerNet(self,model):
        if(self.alive):
            return self.getPowerOutput(model) - self.getPowerInput(model)
        else:
            return 0

    def getKind(self):
        return self.kind

    def getOpasity(self):
        return self.opasity

    def getTree(self):
        return self.tree

    def getEnergyToBuild(self):
        return self.energy_to_build

    def getColor(self):
        return self.color

    def killCell(self):
        self.alive=False


class Leaf(Cell):
    def __init__(self, tree, pos):
        super().__init__(tree,LEAF, LEAF_MAINTAINE, pos,0.2,LEAF_POWER,LEAD_BUILD,[0.0,1.0,0.0])

    def getPowerOutput(self,model):
        optcity_multiplier = 1.0
        for i in range(self.pos[2],model.shape[2]):
            cell = model[self.pos[0],self.pos[1],i]
            if(cell==None):
                continue
            optcity_multiplier*=cell.getOpasity()
            if(optcity_multiplier<=0.0):
                break

        return self.power_max_output*optcity_multiplier


class Branch(Cell):
    def __init__(self, tree,pos):
        super().__init__(tree, BRANCH, BRANCH_MAINTAINE,pos,0.1,0.0,BRANCH_BUILD,[234/255,221/255,202/255])     

class Tree(object):
    def __init__(self,energy=0):
        self.cells = []
        self.newCells=[]
        self.last_move=-1
        self.last_observation=None
        self.energy=energy
        self.power=0
        self.growth_stratagy=None

    def setStratagy(self,s):
        self.growth_stratagy=s

    def growCycle(self,model,prev_obs,max_iter=100):
        e1 = self.energy
        for i in range(len(self.cells)):
            cell = self.cells[i]
            self.energy+=cell.getPowerNet(model)
        if(self.growth_stratagy==None):
            return False
        #for cell in self.cells:
        sucessful_action=False
        #while(not sucessful_action):
        if(True):
            cell = rand.choice(self.cells)
            if(not cell.alive):
                #continue
                return True
            #for _ in range(max_iter):
            move = self.growth_stratagy(self,cell,model,prev_obs)
            pos = np.array([0.0,0.0,0.0],dtype=int)
            kind=-1
            self.last_move=move
            if(move==0):#TODO maybe do this cleaner
                pos[0]=1
                kind=BRANCH
            elif(move==1):
                pos[0]=-1
                kind=BRANCH
            elif(move==2):
                pos[1]=1
                kind=BRANCH
            elif(move==3):
                pos[1]=-1
                kind=BRANCH
            elif(move==4):
                pos[2]=1
                kind=BRANCH
            elif(move==5):
                pos[2]=-1
                kind=BRANCH
            elif(move==6):
                pos[0]=1
                kind=LEAF
            elif(move==7):
                pos[0]=-1
                kind=LEAF
            elif(move==8):
                pos[1]=1
                kind=LEAF
            elif(move==9):
                pos[1]=-1
                kind=LEAF
            elif(move==10):
                pos[2]=1
                kind=LEAF
            elif(move==11):
                pos[2]=-1
                kind=LEAF
            elif(move==12):
                pos[0]=1
                kind=KILL
            elif(move==13):
                pos[0]=-1
                kind=KILL
            elif(move==14):
                pos[1]=1
                kind=KILL
            elif(move==15):
                pos[1]=-1
                kind=KILL
            elif(move==16):
                pos[2]=1
                kind=KILL
            elif(move==17):
                pos[2]=-1
                kind=KILL
            sucessful_action = self.addCell(model,kind,pos+cell.pos,cell)
        self.cells+=self.newCells
        self.newCells=[]
        self.power=self.energy-e1
        self.last_observation = cell.getObservation(model)
        if(self.energy<0):
            return False
        return True

    def addCell(self,model,cell_type,pos,growth_cell):
        if(growth_cell!=None and growth_cell.getKind()==LEAF):
            #print("Invalid Growth Cell Type")
            return False

        if(growth_cell!=None and np.sum(np.abs(growth_cell.pos-pos)) != 1 ):
            #print("Distance From Growth Cell Too Lrage")
            return False
        
        if(model[pos[0],pos[1],pos[2]] != None):
            if(cell_type==KILL):
                model[pos[0],pos[1],pos[2]].killCell()
                model[pos[0],pos[1],pos[2]] = None
                return True
            #print("Position Not Empty")
            return False

        new_cell=None
        if(cell_type==LEAF):
            new_cell = Leaf(self,pos)
        elif(cell_type==BRANCH):
            new_cell = Branch(self,pos)
        else:
            #print("Unknow Cell Type")
            return False

        delta_energy = new_cell.getEnergyToBuild()
        if(delta_energy>self.energy):
            #print("Not Enought Energy To Build",self.energy)
            return False
        self.energy-=delta_energy
        if(growth_cell==None):
            self.cells.append(new_cell)
        else:
            self.newCells.append(new_cell)
        model[pos[0],pos[1],pos[2]] = new_cell
        #print("Cell Added, Energy Left:",self.energy)
        return True

class TreeSim(object):
    def __init__(self,MODEL_MAX_Z=32,MODEL_MAX_Y=32,MODEL_MAX_X=32):
        super().__init__()
        self.MODEL_MAX_X=MODEL_MAX_X
        self.MODEL_MAX_Y=MODEL_MAX_Y
        self.MODEL_MAX_Z=MODEL_MAX_Z
        self.model = np.empty((MODEL_MAX_X,MODEL_MAX_Y,MODEL_MAX_Z),dtype=Cell)
        self.trees=[]

    def createTree(self,pos,stategy,init_energy=100):
        t1 = Tree(100.0)
        t1.setStratagy(stategy)
        t1.addCell(self.model,BRANCH,pos,None)
        self.trees.append(t1)

    def step(self,prev_obs):
        running = False
        for tree in self.trees:
            running = tree.growCycle(self.model,prev_obs) or running
        return running

def addCubeToRender(cell,colors,filled):
    colors[cell.pos[0],cell.pos[1],cell.pos[2]]=cell.getColor()
    filled[cell.pos[0],cell.pos[1],cell.pos[2]]=True

def renderTree(tree,colors,filled):
    for cell in tree.cells:
        addCubeToRender(cell,colors,filled)
    
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

def random_stategy(tree,cell,model,prev_obs):
    if(cell.getKind()==LEAF):
        return False
    w = [0.5, #gorw_xp Branch
         0.5, #gorw_xm
         0.5, #gorw_yp
         0.5, #gorw_ym
         5.0, #gorw_zp
         0.01, #gorw_zm
         1.0, #gorw_xp Leaf
         1.0, #gorw_xm
         1.0, #gorw_yp
         1.0, #gorw_ym
         1.0, #gorw_zp
         0.0, #gorw_zm
         0.1, #kill_xp Leaf
         0.1, #kill_xm
         0.1, #kill_yp
         0.1, #kill_ym
         0.0, #kill_zp
         1.0] #kill_zm

    r = rand.random()*sum(w)

    if(r<w[0]):
        return 0
    for i in range(1,len(w)):
        if(r<sum(w[:i+1])):
            return i
    return -1