FROM tensorflow/tensorflow:2.4.1-gpu

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y \
    git  \
    libgl1-mesa-glx \
    python3-pip  \
    python3.7 \
    rsync \
    software-properties-common

RUN ln -fs /usr/bin/python3.7 /usr/local/bin/python
RUN pip3 install --upgrade pip && \
  pip3 install \
  pandas \
  rasterio \
  matplotlib \
  tqdm \
  opencv-python \
  scipy

WORKDIR /capstone_fall20_irrigation
