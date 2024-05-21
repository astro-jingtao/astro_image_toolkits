#!/bin/bash
#PBS -N cube_2dp
#PBS -lselect=1:ncpus=5:mem=92gb

cd $PBS_O_WORKDIR
source activate base
python test_fft.py