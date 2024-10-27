ARG Python_VERSION=python:3.13-slim-bookworm

FROM ${Python_VERSION}

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
ccache \
cmake \
g++ \
libeigen3-dev \
libfmt-dev \
mold \
ninja-build

COPY docs/requirements.txt ./docs/
COPY python/requirements.txt ./python/
COPY requirements.txt ./

RUN pip install -r requirements.txt
