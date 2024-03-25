# Implementation of SIR using Repast4py

## Installation
```
conda create --name repast python==3.10 pip
env CC=mpicxx pip install repast4py
coa activate repast
mpirun -n 2 python sir.py params/sir-light.yaml
```
