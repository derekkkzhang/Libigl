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

| 🚨 Important |
|:---|
| The latest version of libigl (v2.4.0) introduces some **breaking changes** to its CMake build system. Please read our [changelog](https://libigl.github.io/changelog/) page for instructions on how to update your project accordingly. |
