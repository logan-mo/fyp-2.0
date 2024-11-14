FROM python:3.10

RUN mkdir build

RUN apt update && apt install build-essential dkms libpoppler-cpp-dev pkg-config -y

ARG nvidia_binary_version="525.105.17"
ARG nvidia_binary="NVIDIA-Linux-x86_64-525.105.17.run"
ARG nvidia_binary_url="https://us.download.nvidia.com/tesla/525.105.17/NVIDIA-Linux-x86_64-525.105.17.run"
RUN wget -q ${nvidia_binary_url} \
    && chmod +x ${nvidia_binary} \
    && ./${nvidia_binary} --accept-license --ui=none --no-kernel-module --no-questions \
    && rm -rf ${nvidia_binary}

RUN apt update && apt install -y cmake                           \
    && apt install -y python3-pip                                  
COPY requirements.txt ./build/requirements.txt
RUN pip3 install -r /build/requirements.txt                               
RUN apt remove -y python3-pip                                              \
    && apt autoremove --purge -y                                              \
    && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/*.list

EXPOSE 8000

RUN mkdir ./build/saved_models
COPY app.py ./build/app.py
COPY saved_models/ ./build/saved_models/

ARG env_port
ENV deployment_port=$env_port 

WORKDIR /build
CMD ["uvicorn","app:app","--host","0.0.0.0","--port",${deployment_port}]