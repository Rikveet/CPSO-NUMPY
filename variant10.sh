#!/bin/bash
# The following line can be included if you have multi-thread programs make sure the #PBS stays at the front of the line
#PBS -l select=1:ncpus=2:mem=7gb
dataset=$1

/usr/bin/scl enable rh-python38 'python3 ./experiments/CPSO-NUMPY/main.py '$dataset' mcpso layer'