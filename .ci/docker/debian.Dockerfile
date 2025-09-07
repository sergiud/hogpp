FROM debian:trixie-slim

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
ccache \
cmake \
furo \
g++ \
gcovr \
libboost-test-dev \
libeigen3-dev \
libfmt-dev \
libopencv-dev \
mold \
ninja-build \
python3-dev \
python3-numpy \
python3-pillow \
python3-pybind11 \
python3-pytest \
python3-pytest-xdist \
python3-sphinx \
python3-sphinx-copybutton \
python3-sphinxcontrib.bibtex
