#include "TreeSim.hxx"
#include "Vec3i.hxx"
#include "Stratagies.hxx"

int main(int args,char** argv){
    Stratagy* strat = new RandomStratagy();
    
    TreeSim sim = TreeSim();
    int steps = 1000;
    sim.setLogTree(0);
    sim.createTree(Vec3i(15,15,0), strat, 10.0);
    for(size_t i= 0; i < steps; i++){
        sim.step(sim.getObervation(0));
    }
    
}