#!/bin/bash

set -e

mkdir metis_related
cd metis_related
git clone https://github.com/KarypisLab/GKlib.git
cd GKlib
make
make install
cd ..

git clone https://github.com/KarypisLab/METIS.git
cd METIS
make config shared=1 cc=cc prefix=~/local i64=1
make install
cd ..

git clone --branch dgl https://github.com/KarypisLab/ParMETIS.git
cd ParMETIS
make config cc=cc prefix=~/local
make install

cd ../..

export PATH=$PATH:$HOME/local/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/lib/