cmake_minimum_required(VERSION 3.16.3)
project(tree_simulator)

set (CMAKE_CXX_STANDARD 17)
#set(pybind11_DIR "/home/$USER/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")

set(SOURCE
  #TreeSim.cxx
  Vec3i.cxx
)

find_package(pybind11 2.4.3 REQUIRED)
pybind11_add_module(tree_simulator SHARED module.cpp ${SOURCE})
