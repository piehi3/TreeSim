cd build
cmake ..
make
cd ..
rm -f tree_simulator.cpython-38-aarch64-linux-gnu.so
cp build/tree_simulator.cpython-38-aarch64-linux-gnu.so .
