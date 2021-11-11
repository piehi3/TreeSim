#ifndef __TREE_SIM
#define __TREE_SIM

#include <array>
#include <vector>
#include <random>
#include <iostream>
#include "Vec3i.hxx"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class Stratagy;

enum KIND {LEAF=0,BRANCH=1,KILL=9,ERROR_KIND = -1,EMPTY=-2};

//Leaf varriable
static const double LEAF_POWER=1.5;
static const double LEAD_BUILD=5.0;
static const double LEAF_MAINTAINE=0.5;
static const double LEAF_OPASITY=0.2;

//Branch varriables;
static const double BRANCH_BUILD=0.25;
static const double BRANCH_MAINTAINE=0.1;
static const double BRANCH_OPASITY=0.0;

static const int MODEL_X=32;
static const int MODEL_Y=32;
static const int MODEL_Z=32;

static const int OBSERVATION_SIZE=10;

class Tree;//forward declaation 
class Cell;

//CXX14 standar to simplify
typedef std::array<std::array<std::array<Cell*, MODEL_X>,MODEL_Y>,MODEL_Z> Model;

typedef std::array<double,OBSERVATION_SIZE> Observation;

class Pos{
public:
    int x;
    int y;
};

inline Vec3i getPosFromMove(int move){
    switch (move)
    {
    case 0:
        return Vec3i(1,0,0);
    case 1:
        return Vec3i(-1,0,0);

    case 2:
        return Vec3i(0,1,0);
    case 3:
        return Vec3i(0,-1,0);

    case 4:
        return Vec3i(0,0,1);
    case 5:
        return Vec3i(0,0,-1);
    
    default:
        return Vec3i(0,0,0);
    }
}

class Cell {
private:
    KIND kind;
    Vec3i pos;
    bool alive;
    double power_input;
    double opasity;
    Tree* tree;
    double energy_to_build;
protected:
    double power_max_output;
    //self.color = np.array(color) do this in python
public:
    Cell(){};
    Cell(Tree* tree,KIND kind,double opasity,Vec3i pos
    ,double power_input,double power_max_output,double energy_to_builds);
    Observation getObservation(Model &model);
    virtual double getPowerOutput(Model &model) = 0;
    virtual double getPowerInput(Model &model);
    virtual double getPowerNet(Model &model);
    KIND getKind();
    double getOpasity();
    Vec3i getPos();
    Tree* getTree();
    double getEnergyToBuild();
    bool isAlive();
    //double getColor();implmeneted in python section
    void killCell();

};

/*void py_init_cell(py::module& tm) {
    py::class_<Cell>(tm, "Cell")
        .def(py::init<Tree*,KIND,double,Vec3i,double,double,double>())
        .def("getObservation", &Cell::getObservation)
        .def("getPowerOutput", &Cell::getPowerOutput)
        .def("getPowerInput", &Cell::getPowerInput)
        .def("getPowerNet", &Cell::getPowerNet);
}*/

class Leaf: public Cell{
public:
    Leaf(Tree* tree, Vec3i pos);
    double getPowerOutput(Model &model) override;
};

class Branch: public Cell{
public:
    Branch(Tree* tree, Vec3i pos);
    inline double getPowerOutput(Model &model) override {return 0.0;};
};


class Tree{
private:
    std::vector<Cell*> cells;
    std::vector<Cell*> new_cells;
    int last_move;
    Observation last_observation;
    double energy;
    double power;
    Stratagy* growth_strat;
public:
    Tree(double energy);
    double getEnergy();
    double getPower();
    void setStratagy(Stratagy* strat);//TODO: how to do this with pybind11??
    bool growCycle(Model &model,Observation ob, int max_iter=100);
    bool addCell(Model &model,KIND kind,Vec3i pos,Cell* growth_cell);
    Observation getLastObservation();

};

void py_init_tree(py::module& tm) {
    py::class_<Tree>(tm, "Tree")
        .def(py::init<double>())
        .def("getEnergy", &Tree::getEnergy)
        .def("getPower", &Tree::getPower)
        .def("getLastObservation", &Tree::getLastObservation);
}

class TreeSim{
private:
    Model model;
    Tree* tree;//main tree for singel network testing
    std::vector<Tree*> trees;

public:
    TreeSim();
    void createTree(Vec3i pos,Stratagy* stategy, double init_energy=10.0);
    bool step(Observation prev_observation);
    Observation getObervation(int i);
};

void py_init_treesim(py::module& tm) {
    py::class_<TreeSim>(tm, "TreeSim")
        .def(py::init<>())
        .def("createTree", &TreeSim::createTree)
        .def("step", &TreeSim::step)
        .def("getObervation", &TreeSim::getObervation);
}

class Stratagy{
public:
    virtual int operator()(Tree *tree,Cell *cell, Model &model,Observation prev_obs) = 0;
};

#endif