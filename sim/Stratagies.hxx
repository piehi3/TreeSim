#ifndef __STRAT
#define __STRAT

#include "Vec3i.hxx"
#include "TreeSim.hxx"

class RandomStratagy: public Stratagy{
private:
const std::vector<double> induvidual_probablities = {0.5, //gorw_xp Branch
         0.5, //gorw_xm
         0.5, //gorw_yp
         0.5, //gorw_ym
         5.0, //gorw_zp
         0.01,//gorw_zm
         1.0, //gorw_xp Leaf
         1.0, //gorw_xm
         1.0, //gorw_yp
         1.0, //gorw_ym
         1.0, //gorw_zp
         0.0, //gorw_zm
         0.1, //kill_xp Leaf
         0.1, //kill_xm
         0.1, //kill_yp
         0.1, //kill_ym
         0.0, //kill_zp
         1.0}; //kill_zm

std::vector<double> probabilities;

public:
    inline RandomStratagy(){
        double s = 0.0;
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
};

#endif