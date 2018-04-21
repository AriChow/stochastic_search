#!/bin/sh
#SBATCH --job-name=SS_breast
#SBATCH -t 35:59:00
#SBATCH -D /gpfs/u/home/CGHI/CGHIarch/barn/stochastic_search/

cd /gpfs/u/home/CGHI/CGHIarch/barn/stochastic_search/
python SS_main.py breast barn 0 5