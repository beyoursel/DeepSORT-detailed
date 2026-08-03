#!/bin/bash
rm -rf build
mkdir build
cd build
cmake .. -DUSE_TENSORRT=ON
make -j8
cd ..


