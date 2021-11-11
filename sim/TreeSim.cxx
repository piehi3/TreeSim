#include "TreeSim.hxx"
#include <algorithm>
#include <iterator>

static KIND getKindNear(Model &model, Cell* cell, Vec3i pos){
    
    Vec3i p = cell->getPos()+pos;
    if(p.x<0 || p.y<0 || p.z<0 ||
       p.x>=MODEL_X || p.y>=MODEL_Y || p.z>=MODEL_Z)
        return ERROR_KIND;
    if(model[p[0]][p[1]][p[2]]==nullptr)
        return EMPTY;
    return model[p[0]][p[1]][p[2]]->getKind();
}

//Generatic Cell Stuff

Cell::Cell(Tree* _tree,KIND _kind,double _opasity,Vec3i _pos
    ,double _power_input,double _power_max_output,double _energy_to_build):
    kind(_kind),pos(_pos),alive(true),power_input(_power_input),power_max_output(_power_max_output)
    ,opasity(_opasity),tree(_tree),energy_to_build(_energy_to_build){}

Observation Cell::getObservation(Model &model){
       
        return {double(pos.x),double(pos.y),double(pos.z),
                double(getKindNear(model,this,Vec3i(1,0,0))),
                double(getKindNear(model,this,Vec3i(-1,0,0))),
                double(getKindNear(model,this,Vec3i(0,1,0))),
                double(getKindNear(model,this,Vec3i(0,-1,0))),
                double(getKindNear(model,this,Vec3i(0,0,1))),
                double(getKindNear(model,this,Vec3i(0,0,-1))),
                tree->getPower()};
}

//double Cell::getPowerOutput(Model &model){return 0;}

double Cell::getPowerInput(Model &model){ return power_input;}

double Cell::getPowerNet(Model &model){
    if(alive)
        return getPowerOutput(model) - getPowerInput(model);
    else
        return 0;
}

KIND Cell::getKind(){
    return kind;
}

double Cell::getOpasity(){
    return opasity;
}

Tree* Cell::getTree(){
    return tree;
}

bool Cell::isAlive(){
    return alive;
}

double Cell::getEnergyToBuild(){
    return energy_to_build;
}

//def Cell::getColor(self):
//    return self.color

void Cell::killCell(){
    alive=false;
}

Vec3i Cell::getPos(){
    return pos;
}

Branch::Branch(Tree* tree, Vec3i pos) : Cell(tree, BRANCH, BRANCH_OPASITY, pos, BRANCH_MAINTAINE, 0, BRANCH_BUILD){}

Leaf::Leaf(Tree* tree, Vec3i pos) : Cell(tree, LEAF, LEAF_OPASITY, pos, LEAF_MAINTAINE, LEAF_POWER, LEAD_BUILD){}

double Leaf::getPowerOutput(Model &model){
    double optcity_multiplier = 1.0;
    Vec3i p = this->getPos();
    for(int i = p.y; i < MODEL_Z; i++){
        Cell* cell = model[p.x][i][p.y];
        if(cell==nullptr)
            continue;
        optcity_multiplier*=cell->getOpasity();
        if(optcity_multiplier<=0.0)
            break;
    
    }
    return this->power_max_output*optcity_multiplier;
}

bool Tree::growCycle(Model &model,Observation ob, int max_iter){
        int e1 = energy;
        
        for(Cell* cell: cells)
            energy+=cell->getPowerNet(model);
        //for cell in self.cells:
        bool sucessful_action=false;
        //while(not sucessful_action):
        std::vector<Cell*> chosen_cell;
        std::sample(cells.begin(),cells.end(), std::back_inserter(chosen_cell),1,std::mt19937{std::random_device{}()});
        
        for(Cell* cell:cells){
            if(!cell->isAlive())
                return true;
            
            //#for _ in range(max_iter):
            int move = (*growth_strat)(this,cell,model,ob);
            Vec3i pos=getPosFromMove(move%6);
            KIND kind=ERROR_KIND;
            last_move=move;
            //before here
            switch (int(move/6))
            {
            case 0:
                kind=BRANCH;
                break;
            
            case 1:
                kind=LEAF;
                break;

            case 2:
                kind=KILL;
                break;
            
            default:
                break;
            }
            
            bool action_successful = addCell(model,kind,pos+cell->getPos(),cell);
            cells.insert(cells.end(), new_cells.begin(),new_cells.end());
            new_cells.clear();
            power=energy-e1;
            last_observation = cell->getObservation(model);
        }
       
        if(energy<0)
            return false;
        return true;
}

bool Tree::addCell(Model &model,KIND kind,Vec3i pos,Cell* growth_cell){
    if(growth_cell!=nullptr && growth_cell->getKind()==LEAF)
        return false;

    if(growth_cell!=nullptr && norm2(abs(growth_cell->getPos()-pos)) != 1 ){
        //#print("Distance From Growth Cell Too Lrage")
        return false;
    }
    
    if(model[pos.x][pos.y][pos.z] != nullptr){
        if(kind==KILL){
            model[pos.x][pos.y][pos.z]->killCell();
            model[pos.x][pos.y][pos.z] = nullptr;
            return true;
        }
        //#print("Position Not Empty")
        return false;
    }

    Cell* new_cell;
    switch (kind)
    {
    case LEAF:
        new_cell = new Leaf(this,pos);
        break;
    case BRANCH:
        new_cell = new Branch(this,pos);
        break;
    
    default:
        return false;
    }

    double delta_energy = new_cell->getEnergyToBuild();
    if(delta_energy>this->energy){
        //#print("Not Enought Energy To Build",self.energy)
        return false;
    }
    this->energy-=delta_energy;
    if(growth_cell==nullptr){
        cells.push_back(new_cell);
    }else{
        new_cells.push_back(new_cell);
    }
    model[pos.x][pos.y][pos.z] = new_cell;
    //#print("Cell Added, Energy Left:",self.energy)
    return true;
}

Observation Tree::getLastObservation(){
    return last_observation;
}

Tree::Tree(double _energy){
    cells = std::vector<Cell*>();
    new_cells = std::vector<Cell*>();
    last_move=-1;
    energy=_energy;
    power=0;
}

double Tree::getPower(){
    return power;
}

double Tree::getEnergy(){
    return energy;
}

void Tree::setStratagy(Stratagy* strat){
    this->growth_strat = strat;
}

TreeSim::TreeSim(){
    model = Model();
    trees = std::vector<Tree*>();
}

void TreeSim::createTree(Vec3i pos,Stratagy* stategy, double init_energy){
    tree = new Tree(init_energy);
    tree->setStratagy(stategy);
    tree->addCell(model,BRANCH,pos,nullptr);
    trees.push_back(tree);
}

Observation TreeSim::getObervation(int i){
    return trees[i]->getLastObservation();
}

bool TreeSim::step(Observation prev_observation){
    bool running = false;
    for(Tree* tree : trees){
        running = tree->growCycle(model,prev_observation) || running;
        
    }
    return running;
}