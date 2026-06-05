# Reproducible build for sbayesrc-gpu.
#
# Build:
#   docker build -t sbayesrc-gpu -f docs/Dockerfile .
#
# Run:
#   docker run --gpus all --rm -it -v $(pwd)/data:/data sbayesrc-gpu \
#       gctb --gwfm RC --use-gpu --ldm-eigen /data/ukbEUR_Imputed ...

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    wget \
    libboost-all-dev \
    libeigen3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Build GCTB with GPU patch.
# Pin GCTB to a known commit so the patch applies cleanly.
WORKDIR /opt
RUN git clone https://github.com/jianzeng/GCTB.git GCTB && \
    cd GCTB && git checkout master   # pin a specific SHA in production

# Copy the GPU patch + dispatch sources.
COPY src/gctb-gpu-patch.patch  /opt/sbayesrc-gpu/
COPY src/sbrc_gpu.cu           /opt/sbayesrc-gpu/
COPY src/sbrc_gpu.hpp          /opt/sbayesrc-gpu/

RUN cd /opt/GCTB && \
    git apply /opt/sbayesrc-gpu/gctb-gpu-patch.patch && \
    cp /opt/sbayesrc-gpu/sbrc_gpu.cu  scr/ && \
    cp /opt/sbayesrc-gpu/sbrc_gpu.hpp scr/

# Build. GPU_ARCH should match your hardware: sm_80 (A100), sm_90 (H100), sm_86 (RTX 30xx).
ARG GPU_ARCH=sm_90
ENV GPU_ARCH=${GPU_ARCH}
ENV EIGEN3_INCLUDE_DIR=/usr/include/eigen3
ENV BOOST_LIB=/usr/include

RUN cd /opt/GCTB/scr && \
    make USE_GPU=1 GPU_ARCH=${GPU_ARCH} -j$(nproc)

# Make the binary available on PATH.
RUN cp /opt/GCTB/scr/gctb /usr/local/bin/gctb

CMD ["gctb"]
