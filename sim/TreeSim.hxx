#include <array>
#include <vector>
#include "vec2.hxx"

enum KIND {LEAF=0,BRANCH=1,KILL=9,ERROR_KIND = -1}

//Leaf varriable
const double LEAF_POWER=1.5
const double LEAD_BUILD=5.0
const double LEAF_MAINTAINE=0.5

//Branch varriables
const double BRANCH_BUILD=0.25
const double BRANCH_MAINTAINE=0.1

const double MODEL_X=32
const double MODEL_Y=32
const double MODEL_Z=32

class Tree;//forward declaation 
class Cell;

typedef std::array<Cell, MODEL_X,MODEL_Y,MODEL_Z> Model;

class Pos{
public:
    int x;
    int y;
}

class Cell {

};

