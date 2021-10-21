#include "TreeSim.hxx"

static getKindNear(Model &model, Cell* cell,)
def getKindNear(model,cell,pos):
    p = cell.pos+pos
    if(np.any(p>model.shape)):
        return ERROR_KIND
    return model[p[0],p[1],p[2]]