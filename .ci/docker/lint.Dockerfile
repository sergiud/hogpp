# syntax=docker/dockerfile:1
FROM debian:bookworm-slim AS base

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
ca-certificates

FROM base AS tools

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
gnupg \
wget

RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

FROM base AS deploy

RUN echo deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main \
>/etc/apt/sources.list.d/llvm.list

COPY --from=tools /etc/apt/trusted.gpg.d/apt.llvm.org.asc /etc/apt/trusted.gpg.d/apt.llvm.org.asc

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
--no-install-recommends --no-install-suggests \
clang-format-19 \
fd-find
