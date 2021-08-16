#!/bin/bash

set -x -eo pipefail

: ${BUILDDIR:=build}

cmake -S . -B "${BUILDDIR}"/ -G Ninja \
    -DCMAKE_CXX_FLAGS_INIT='-fuse-ld=mold' \
    -DCMAKE_REQUIRE_FIND_PACKAGE_pybind11=ON \
    -DCMAKE_REQUIRE_FIND_PACKAGE_Python=ON \
    -Dpybind11_ROOT="$(pybind11-config --cmakedir)"

cmake --build "${BUILDDIR}"/ --target pyhogpp

SITEPATH=$(python -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')

mkdir -pv "${SITEPATH}"/
cp -v --target-directory="${SITEPATH}"/ "${BUILDDIR}"/hogpp.cpython-*.so
