#!/bin/sh
#SBATCH --job-name=reinforcement1
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/stochastic_search/error_propagation

cd /gpfs/u/home/CGHI/CGHIarch/barn/stochastic_search/
python SS_main.py breast barn 0 1
python SS_main.py matsc_dataset1 barn 0 1
python SS_main.py matsc_dataset2 barn 0 1
python SS_main.py brain barn 0 1
python SS_main.py bone barn 0 1