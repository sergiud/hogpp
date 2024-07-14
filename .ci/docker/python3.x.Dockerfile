ARG Python_VERSION=3.12

FROM python:${Python_VERSION}-slim-bookworm

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
ccache \
cmake \
g++ \
libboost-test-dev \
libeigen3-dev \
libfmt-dev \
libopencv-dev \
mold \
ninja-build

COPY requirements.txt ./

RUN pip install -r requirements.txt
