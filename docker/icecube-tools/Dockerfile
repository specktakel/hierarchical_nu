FROM python:2.7-slim

# Basic tools
RUN apt-get update && apt-get install -y git \
    dpkg-dev make g++ gcc binutils libx11-dev \
    libxpm-dev libxft-dev libxext-dev python-dev gfortran \
    cmake python-pip wget

WORKDIR /opt/

# ROOT 5.34
ENV ROOTSYS="/opt/root"
ENV PATH="$ROOTSYS/bin:$PATH"
ENV LD_LIBRARY_PATH="$ROOTSYS/lib:$LD_LIBRARY_PATH"

RUN cd /opt && wget https://root.cern.ch/download/root_v5.34.36.source.tar.gz &&\
    tar xvzf root_v5.34.36.source.tar.gz &&\
    rm -f root_v5.34.36.source.tar.gz
RUN cd root && mkdir root_build
WORKDIR /opt/root_build
RUN cmake -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -Dfail-on-missing=ON \
    -Dminuit2=ON \
    -Dbuiltin_zlib=ON \
    -Dminimal=ON \
    -Dtmva=OFF \
    -Dmathmore=OFF \
    -Dpython=ON \
    -Dopengl=OFF \
    -Droofit=ON \
    -Dxml=OFF \
    -Dssl=OFF \
    -Dcastor=OFF \
    -Dmysql=OFF \
    -Doracle=OFF \
    -Dpgsql=OFF \
    -Dsqlite=OFF \
    -Dpythia6=OFF \
    -Dpythia8=OFF \
    -Dfftw3=OFF \
    -Dxrootd=OFF \
    -Dgfal=OFF \
    -Ddavix=OFF \
    -Dfitsio=OFF \
    -Dx11=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=/opt/root  ../root && \
     make && make install && cd ..

ENV PATH=$ROOTSYS/bin:$PATH
ENV PYTHONDIR=$ROOTSYS
ENV LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$ROOTSYS/bindings/pyroot:$LD_LIBRARY_PATH
ENV PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH:$ROOTSYS/bindings/pyroot

# python
RUN pip install numpy scipy ipython jupyter pandas matplotlib tables root_numpy rootpy

WORKDIR /opt/

# Meerkat 1.3.0
RUN cd /opt && wget https://meerkat.hepforge.org/downloads/meerkat-1.3.0.tar.gz &&\
    tar xvzf meerkat-1.3.0.tar.gz &&\
    rm -f meerkat-1.3.0.tar.gz
WORKDIR /opt/Meerkat
RUN make

ENV MEERKAT="/opt/Meerkat/lib"

# Run jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--allow-root"]
