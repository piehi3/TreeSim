import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random as rand

from numpy.lib.npyio import loadtxt
from numpy.lib.twodim_base import fliplr

LEAF=0
BRANCH=1
KILL=9
ERROR_KIND = -1
NONE_KIND = -2
ERROR_MOVE = -99

LEAF_POWER=1.5
LEAD_BUILD=5.0
LEAF_MAINTAINE=0.5

BRANCH_BUILD=0.25
BRANCH_MAINTAINE=0.1

def getKindNear(model,cell,pos):
    p = cell.pos+pos
    if(np.any(p.to_array()>model.shape)):
        return ERROR_KIND
    if(model[p[0],p[1],p[2]]==None):
        return NONE_KIND
    return model[p[0],p[1],p[2]].kind
    
class Vec3i():
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self,a):
        return Vec3i(self.x+a.x,self.y+a.y,self.z+a.z)
        
    def __sub__(self,a):
        return Vec3i(self.x-a.x,self.y-a.y,self.z-a.z)

    def __getitem__(self,index):
        if(index==0):
            return self.x
        elif(index==1):
            return self.y
        elif(index==2):
            return self.z 
        print("index {} out of bounds for vec of size 3".format(index))

    def __gt__(self,a):
        return np.array([self.x>a.x,self.y>a.y,self.z>a.z],dtype=bool)

    def __lt__(self,a):
        return np.array([self.x<a.x,self.y<a.y,self.z<a.z],dtype=bool)

    def __eq__(self,a):
        return np.array([self.x==a.x,self.y==a.y,self.z==a.z],dtype=bool)

    def to_array(self):
        return np.array([self.x,self.y,self.z],dtype=int)

    def __setitem__(self,index,value):
        if(index==0):
            self.x = value
            return
        elif(index==1):
            self.y = value
            return
        elif(index==2):
            self.z = value
            return
        print("index {} out of bounds for vec of size 3".format(index))

    def __str__(self) -> str:
        return "[{:i}, {:i}, {:i}]".format(self.x,self.y,self.z)

    def sum(vec):
        return vec.x+vec.y+vec.z

    def abs(vec):
        return Vec3i(abs(vec.x),abs(vec.y),abs(vec.z))

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

    def getObservation(self,model,move_used=ERROR_MOVE):
        return [self.pos[0],self.pos[1],self.pos[2],
                move_used,
                getKindNear(model,self,Vec3i(1,0,0)),
                getKindNear(model,self,Vec3i(-1,0,0)),
                getKindNear(model,self,Vec3i(0,1,0)),
                getKindNear(model,self,Vec3i(0,-1,0)),
                getKindNear(model,self,Vec3i(0,0,1)),
                getKindNear(model,self,Vec3i(0,0,-1)),
                self.tree.power]

    def getObservationNoMove(self,model):
        return np.array([self.pos[0],self.pos[1],self.pos[2],
                getKindNear(model,self,Vec3i(1,0,0)),
                getKindNear(model,self,Vec3i(-1,0,0)),
                getKindNear(model,self,Vec3i(0,1,0)),
                getKindNear(model,self,Vec3i(0,-1,0)),
                getKindNear(model,self,Vec3i(0,0,1)),
                getKindNear(model,self,Vec3i(0,0,-1)),
                self.tree.power],dtype=float)

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

    def getRandomCell(self):
        return rand.choice(self.cells)

    def growCycle(self,model,prev_obs,move_override=ERROR_MOVE,cell_override=None,max_iter=100):
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
            if(cell_override==None):
                cell = self.getRandomCell()
            else:
                cell = cell_override
            if(not cell.alive):
                #continue
                return True
            #for _ in range(max_iter):
            if(move_override==ERROR_MOVE):
                move = self.growth_stratagy(self,cell,model,prev_obs)
            else:
                move = move_override
            pos = Vec3i(0,0,0)
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
        self.last_observation = cell.getObservation(model,move)
        if(self.energy<0):
            return False
        return True

    def addCell(self,model,cell_type,pos,growth_cell):
        if(growth_cell!=None and growth_cell.getKind()==LEAF):
            #print("Invalid Growth Cell Type")
            return False

        if(growth_cell!=None and Vec3i.sum(Vec3i.abs(growth_cell.pos-pos)) != 1 ):
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
        self.log_tree = 0
        self.power_log = []
        self.energy_log = []
        self.model = np.empty((MODEL_MAX_X,MODEL_MAX_Y,MODEL_MAX_Z),dtype=Cell)
        self.trees=[]

    def setLogTree(self,log_tree):
        self.log_tree = log_tree

    def getRandomCell(self,tree_index=0):
        return self.trees[tree_index].getRandomCell()

    def getObervation(self,tree_index):
        return self.trees[tree_index].last_observation

    def getPowerLog(self):
        return self.power_log

    def getEnergyLog(self):
        return self.energy_log


    def createTree(self,pos,stategy,init_energy=100):
        t1 = Tree(100.0)
        t1.setStratagy(stategy)
        t1.addCell(self.model,BRANCH,pos,None)
        self.trees.append(t1)

    def step(self,prev_obs,move_override=ERROR_MOVE,cell_override=None):
        running = False
        for tree in self.trees:
            running = tree.growCycle(self.model,prev_obs,move_override,cell_override) or running
        self.power_log.append( self.trees[self.log_tree].power )
        self.energy_log.append( self.trees[self.log_tree].energy )
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
    plt.show()

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

class RandomStratagy():
    def __init__(self):
        self.statagy = random_stategy

    def createRandomTreeSim(self,pos, init_energy):
        sim = TreeSim()
        sim.createTree(pos,self.statagy,init_energy)
        return sim
