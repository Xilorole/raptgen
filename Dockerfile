FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /root

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libcairo2 build-essential libbz2-dev libdb-dev  \
    libreadline-dev libffi-dev libgdbm-dev liblzma-dev   libncursesw5-dev \
    libsqlite3-dev libssl-dev zlib1g-dev uuid-dev python3 python3-dev curl git
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3
RUN curl https://pyenv.run | bash
RUN git clone https://github.com/Xilorole/raptgen.git

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=/root/.pyenv/bin:$PATH

RUN eval "$(pyenv init --path)"
RUN eval "$(pyenv init -)"
RUN eval "$(pyenv virtualenv-init -)"
RUN pyenv install 3.7.11
RUN pip install pipenv
WORKDIR /root/raptgen
RUN pipenv install 
