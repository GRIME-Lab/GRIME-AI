FROM almalinux:8.10
LABEL org.opencontainers.image.source https://github.com/GRIME-Lab/GRIME-AI

RUN dnf install -y epel-release
RUN dnf install -y -q mesa-libGLU-devel mesa-libGL-devel tcsh bzip2 libglvnd-glx xcalc
RUN dnf groupinstall -y base-x

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C /usr/local -xvj bin/micromamba

ARG GRIME_PACKAGE='grime-ai'
ARG PYTHON_VERSION="3.12"

RUN CONDA_OVERRIDE_CUDA=12.9 micromamba create -y -p /opt/grimeai -c grimelab -c conda-forge ${GRIME_PACKAGE} python=${PYTHON_VERSION} 'pytorch=*=cuda*' && \
    micromamba clean -y -a

RUN /opt/grimeai/bin/download-sam2-checkpoints
RUN command -v /opt/grimeai/bin/download-yolo11-models && /opt/grimeai/bin/download-yolo11-models do-something || true

RUN useradd -m grimeai

ENV PATH=/opt/grimeai/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/grimeai/lib:$LD_LIBRARY_PATH
ENV QT_XCB_GL_INTEGRATION=none
ENV QT_QUICK_BACKEND=software
ENV HOME=/home/grimeai
WORKDIR /home/grimeai

USER grimeai
