cmake_minimum_required(VERSION 3.16.3)
project(tree_simulator)


project(tree_simulator LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 11)#17
set(CMAKE_CXX_EXTENSIONS OFF)

#set(USE_LLVM ON)
#set(CMAKE_SYSTEM_NAME Windows)
# which compilers to use
#set(CMAKE_CXX_COMPILER /home/idwcorni/llvm-mingw-aarch64/bin/aarch64-w64-mingw32-clang++)
#set(CMAKE_C_COMPILER /home/idwcorni/llvm-mingw-aarch64/bin/aarch64-w64-mingw32-clang)

#set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

#set(CMAKE_FIND_ROOT_PATH /home/idwcorni/llvm-mingw-aarch64/aarch64-w64-mingw32)
#set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

#set(pybind11_INCLUDE "/home/$USER/.local/lib/python3.8/site-packages/pybind11/include")

set(SOURCE
  TreeSim.cxx
  Vec3i.cxx
)

find_package(pybind11 2.4.3 REQUIRED)

#include_directories(tree_simulator PRIVATE "/home/$USER/.local/lib/python3.8/site-packages/pybind11/include")
pybind11_add_module(tree_simulator STATIC module.cpp ${SOURCE})