#!/bin/sh
#SBATCH --job-name=reinforcement1
#SBATCH -t 35:59:00
#SBATCH -D /home/aritra/Documents/research/EP_project/error_propagation

cd /home/aritra/Documents/research/stochastic_search/
python SS_main.py breast Documents/research 0 5
python SS_main.py matsc_dataset1 Documents/research 0 5
python SS_main.py matsc_dataset2 Documents/research 0 5
python SS_main.py brain Documents/research 0 5
# python SS_main.py bone Documents/research 0 5