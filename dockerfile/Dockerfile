FROM nvcr.io/nvidia/pytorch:23.12-py3 as base

WORKDIR /app

RUN set -eux && \
    apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

RUN apt-get update && apt-get -y install sudo
RUN sudo su -

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

RUN apt-get -y install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev gdb
RUN pip uninstall xgboost transformer_engine flash_attn -y
RUN pip install vllm==0.3.2

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

FROM base as dind
# build dind capbability on top of the dind

RUN apt-get update -q && \
    apt-get install -yq \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"  && \
    apt-get update -q && \
    apt-get install -yq \
    docker-ce \
    docker-ce-cli \
    containerd.io 

# https://github.com/docker/docker/blob/master/project/PACKAGERS.md#runtime-dependencies
RUN set -eux; \
    apt-get update -q && \
    apt-get install -yq \
    e2fsprogs \
    iptables \
    xfsprogs \
    xz-utils \
    pigz \
    wget


# set up subuid/subgid so that "--userns-remap=default" works out-of-the-box
RUN set -x \
    && addgroup --system dockremap \
    && adduser --system -ingroup dockremap dockremap \
    && echo 'dockremap:165536:65536' >> /etc/subuid \
    && echo 'dockremap:165536:65536' >> /etc/subgid


RUN set -eux; \
    wget -O /usr/local/bin/dind "https://raw.githubusercontent.com/docker/docker/master/hack/dind"; \
    chmod +x /usr/local/bin/dind


##### Install nvidia docker #####
# Add the package repositories
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update -qq && \
    apt-get install -yq nvidia-docker2 && \
    sed -i '2i \ \ \ \ "default-runtime": "nvidia",' /etc/docker/daemon.json

COPY dind-entrypoint.sh /usr/local/bin/

VOLUME /var/lib/docker
EXPOSE 2375

ENTRYPOINT ["dind-entrypoint.sh"]
CMD []