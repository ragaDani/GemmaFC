ARG DOCKER_BASE=ubuntu:20.04
FROM ${DOCKER_BASE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get --no-install-recommends install -yq \
    git \
    cmake \
    build-essential \
    libgl1-mesa-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-gfx-dev \
    libboost-all-dev \
    libdirectfb-dev \
    libst-dev \
    mesa-utils \
    xvfb \
    x11vnc \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Google Research Football currently depends on gym<=0.21.0. Newer
# setuptools rejects that package metadata during Docker builds, so keep the
# official install flow but pin Python packaging tools to a compatible range.
RUN python3 -m pip install --upgrade \
    "pip==23.3.2" \
    "setuptools==65.5.0" \
    "wheel==0.37.1" \
  && python3 -m pip install psutil

COPY . /gfootball
RUN cd /gfootball && python3 -m pip install --no-build-isolation .
RUN python3 -m pip install six

WORKDIR /gfootball
