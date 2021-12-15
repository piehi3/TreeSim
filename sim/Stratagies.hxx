#ifndef __STRAT
#define __STRAT

#include <time.h>
#include "Vec3i.hxx"
#include "TreeSim.hxx"

class RandomStratagy: public Stratagy{
private:
const std::vector<double> induvidual_probablities = {0.03234894, 0.04083404, 0.04173617, 0.03171915, 0.33842553, 0.00055319,
 0.08165106, 0.08093617, 0.07897872, 0.0826383,  0.10344681, 0.,
 0.00268936, 0.00771064, 0.00551489, 0.00636596, 0.,         0.06445106,};
/*const std::vector<double> induvidual_probablities = {0.5, //gorw_xp Branch
         0.5, //gorw_xm
         0.5, //gorw_yp
         0.5, //gorw_ym
         5.0, //gorw_zp
         0.01,//gorw_zm
         1.0, //gorw_xp Leaf
         1.0, //gorw_xm
         1.0, //gorw_yp
         1.0, //gorw_ym
         1.5, //gorw_zp
         0.0, //gorw_zm
         0.1, //kill_xp Leaf
         0.1, //kill_xm
         0.1, //kill_yp
         0.1, //kill_ym
         0.0, //kill_zp
         1.0}; //kill_zm*/

std::vector<double> probabilities;

public:
    inline RandomStratagy(){
        double s = 0.0;
        srand(time(NULL));
        for(double prob:induvidual_probablities){
            s+=prob;
            probabilities.push_back(s);
        }
    }

    inline int operator()(Tree *tree,Cell *cell, Model &model,Observation prev_obs) override{
        if(cell->getKind()==LEAF)
        return false;

        double r = probabilities.back()*rand()/RAND_MAX;

        for(int i = 0; i < probabilities.size(); i++){
            if(r<probabilities[i])
                return i;
        }
        return -1;
    }

    inline TreeSim* createRandomTreeSim(Vec3i pos, double energy){
        TreeSim* tree_sim = new TreeSim();
        tree_sim->createTree(pos, this, energy);
        return tree_sim;
    }
};

#endif