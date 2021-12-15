#include "Stratagies.hxx"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void py_init_cell(py::module& tm) {
    py::class_<Branch>(tm, "Cell")
        .def(py::init<>());
}

void py_init_branch(py::module& tm) {
    py::class_<Branch>(tm, "Branch")
        .def(py::init<>())
        .def("getKind", &Branch::getKind);
}

void py_init_leaf(py::module& tm) {
    py::class_<Leaf>(tm, "Leaf")
        .def(py::init<>())
        .def("getKind", &Leaf::getKind);
}

void py_init_tree(py::module& tm) {
    py::class_<Tree>(tm, "Tree")
        .def(py::init<double>())
        .def("getEnergy", &Tree::getEnergy)
        .def("getPower", &Tree::getPower)
        .def("getLastObservation", &Tree::getLastObservation);
}

void py_init_treesim(py::module& tm) {
    py::class_<TreeSim>(tm, "TreeSim")
        .def(py::init<>())
        .def("step", &TreeSim::step)
        .def("getModel", &TreeSim::getModel)
        .def("getEnergyLog", &TreeSim::getEnergyLog)
        .def("getPowerLog", &TreeSim::getPowerLog)
        .def("setLogTree", &TreeSim::setLogTree)
        .def("getObervation", &TreeSim::getObervation)
        .def("getObervationLog", &TreeSim::getObervationLog)
        .def("MODEL_X_MAX", &TreeSim::MODEL_X_MAX)
        .def("MODEL_Y_MAX", &TreeSim::MODEL_Y_MAX)
        .def("MODEL_Z_MAX", &TreeSim::MODEL_Z_MAX);
}


void py_init_vec3i(py::module& m) {
    py::class_<Vec3i>(m, "Vec3i")
        .def(py::init<>())
        .def(py::init<const int, const int,const int>())
        .def_readwrite("x", &Vec3i::x)
        .def_readwrite("y", &Vec3i::y)
        .def_readwrite("z", &Vec3i::z);
}

void py_init_randomstratagy(py::module& m) {
    py::class_<RandomStratagy>(m, "RandomStratagy")
        .def(py::init<>())
        .def("createRandomTreeSim", &RandomStratagy::createRandomTreeSim);
}

PYBIND11_MODULE(tree_simulator, m) {
    m.doc() = "A simulator module";

    //py_init_cell(m);
    py_init_tree(m);
    py_init_treesim(m);
    py_init_vec3i(m);
    py_init_randomstratagy(m);
    py_init_branch(m);
    py_init_leaf(m);

}
