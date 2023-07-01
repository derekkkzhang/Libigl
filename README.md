# libigl - A simple C++ geometry processing library


[![](https://github.com/libigl/libigl/workflows/Build/badge.svg?event=push)](https://github.com/libigl/libigl/actions?query=workflow%3ABuild+branch%3Amain+event%3Apush)
[![](https://anaconda.org/conda-forge/igl/badges/installer/conda.svg)](https://anaconda.org/conda-forge/igl)

![](https://libigl.github.io/libigl-teaser.png)

Documentation, tutorial, and instructions at <https://libigl.github.io>.

Libigl is a header-only library. Its functions can be directedly used by including a igl header file
(e.g. #include <igl/readOFF.h> #include <igl/copyleft/cgal/mesh_boolean.h>). The core libigl functionality only depends on the C++ Standard Library and Eigen. This project forks the main Libigl project with CGAL dependencies build together.

The Computational Geometry Algorithms Library (CGAL) is a C++ library that
aims to provide easy access to efficient and reliable algorithms in
computational geometry.

## Before Compilation

1. Some dependencies are required to run Libigl and CGAL. It is better to download all required libraries and setup Windows environment before compilation. Please refer to this article for a detailed guide on how to set up the system environment <https://blog.csdn.net/datoucai1998/article/details/113853102> \

2. Basically, Boost library is needed for CGAL which is also a dependent library for this project. Libigl itself downloads all necessary dependencies automatically. However, to successfully build the project, we need to change two header files in the included libraries: **mpfr.h & mpf2mpfr.h**. Simply download these two header files from the main project folder, and use them to replace the files under **build/_deps/mpfr-src/include** (this directory will be created after initial build).

3. As we will build the project via CMake(and use its gui), make sure to download and install CMake in your computer <https://cmake.org/download/>. Select the file **cmake-3.24.2-windows-x86_64.msi** and download. Remember to check "Add CMake to the system PATH for all users"(default) so you do not need to manually set up the system environment for CMake.


## Run CMake
1. Go to the folder that you want to build Libigl project (e.g. C:\WAAM\Software) and type **cmd** in the address bar.\
2. Then clone this Github repo to your local folder by entering: **git clone https:// <span></span> github.com/derekkkzhang/WAAM_Modelling.git**
3. Enter the follwing commands for the initial build: \
mkdir build\
cd build\
cmake-gui ..\
(DO NOT ignore the two dots)\
We will use CMake interface to build the project. Follow the instructions from <https://blog.csdn.net/JWtricker/article/details/121484508?spm=1001.2014.3001.5502> and remember to check **"LIBIGL_WITH_CGAL"** as we will build Libigl with CGAL as a dependency. We only need Chapter 6 tutorials, so you can uncheck other tutorials in the CMake-gui interface.
4. It will take a minute to generate the VS2019 solution. Then we need to go to the build folder and replace the two mpft header files as mentioned in the previous step.
5. Open the VS2019 solution, run ALL_BUILD
6. Finally, select tutorial 609_Boolean and right click and "set as Startup project". Then test the tutorial 609 by using local meshes.
