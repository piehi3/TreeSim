# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/splat/source/repos/tree_sim/sim

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/splat/source/repos/tree_sim/sim/build

# Include any dependencies generated for this target.
include CMakeFiles/tree_simulator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tree_simulator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tree_simulator.dir/flags.make

CMakeFiles/tree_simulator.dir/module.cpp.o: CMakeFiles/tree_simulator.dir/flags.make
CMakeFiles/tree_simulator.dir/module.cpp.o: ../module.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/splat/source/repos/tree_sim/sim/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tree_simulator.dir/module.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tree_simulator.dir/module.cpp.o -c /mnt/c/Users/splat/source/repos/tree_sim/sim/module.cpp

CMakeFiles/tree_simulator.dir/module.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tree_simulator.dir/module.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/splat/source/repos/tree_sim/sim/module.cpp > CMakeFiles/tree_simulator.dir/module.cpp.i

CMakeFiles/tree_simulator.dir/module.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tree_simulator.dir/module.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/splat/source/repos/tree_sim/sim/module.cpp -o CMakeFiles/tree_simulator.dir/module.cpp.s

CMakeFiles/tree_simulator.dir/Vec3i.cxx.o: CMakeFiles/tree_simulator.dir/flags.make
CMakeFiles/tree_simulator.dir/Vec3i.cxx.o: ../Vec3i.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/splat/source/repos/tree_sim/sim/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tree_simulator.dir/Vec3i.cxx.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tree_simulator.dir/Vec3i.cxx.o -c /mnt/c/Users/splat/source/repos/tree_sim/sim/Vec3i.cxx

CMakeFiles/tree_simulator.dir/Vec3i.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tree_simulator.dir/Vec3i.cxx.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/splat/source/repos/tree_sim/sim/Vec3i.cxx > CMakeFiles/tree_simulator.dir/Vec3i.cxx.i

CMakeFiles/tree_simulator.dir/Vec3i.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tree_simulator.dir/Vec3i.cxx.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/splat/source/repos/tree_sim/sim/Vec3i.cxx -o CMakeFiles/tree_simulator.dir/Vec3i.cxx.s

# Object files for target tree_simulator
tree_simulator_OBJECTS = \
"CMakeFiles/tree_simulator.dir/module.cpp.o" \
"CMakeFiles/tree_simulator.dir/Vec3i.cxx.o"

# External object files for target tree_simulator
tree_simulator_EXTERNAL_OBJECTS =

tree_simulator.cpython-38-aarch64-linux-gnu.so: CMakeFiles/tree_simulator.dir/module.cpp.o
tree_simulator.cpython-38-aarch64-linux-gnu.so: CMakeFiles/tree_simulator.dir/Vec3i.cxx.o
tree_simulator.cpython-38-aarch64-linux-gnu.so: CMakeFiles/tree_simulator.dir/build.make
tree_simulator.cpython-38-aarch64-linux-gnu.so: CMakeFiles/tree_simulator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/splat/source/repos/tree_sim/sim/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library tree_simulator.cpython-38-aarch64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tree_simulator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tree_simulator.dir/build: tree_simulator.cpython-38-aarch64-linux-gnu.so

.PHONY : CMakeFiles/tree_simulator.dir/build

CMakeFiles/tree_simulator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tree_simulator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tree_simulator.dir/clean

CMakeFiles/tree_simulator.dir/depend:
	cd /mnt/c/Users/splat/source/repos/tree_sim/sim/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/splat/source/repos/tree_sim/sim /mnt/c/Users/splat/source/repos/tree_sim/sim /mnt/c/Users/splat/source/repos/tree_sim/sim/build /mnt/c/Users/splat/source/repos/tree_sim/sim/build /mnt/c/Users/splat/source/repos/tree_sim/sim/build/CMakeFiles/tree_simulator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tree_simulator.dir/depend

