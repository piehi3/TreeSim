#include "TreeSim.cxx"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(tree_simulator, m) {
    m.doc() = "A simulator module";

    //py_init_cell(m);
    py_init_tree(m);
    py_init_treesim(m);

}
