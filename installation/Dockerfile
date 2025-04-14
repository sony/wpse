FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   libgl1 \
                   python3.10 \
                   python3-pip \
                   python3.10-venv
RUN rm -rf /var/lib/apt/lists

COPY requirements.txt requirements.txt
COPY requirements_torch.txt requirements_torch.txt
COPY requirements_rapids.txt requirements_rapids.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements_torch.txt
RUN python3 -m pip install --no-cache-dir -r requirements_rapids.txt

RUN rm requirements.txt
RUN rm requirements_torch.txt
RUN rm requirements_rapids.txt