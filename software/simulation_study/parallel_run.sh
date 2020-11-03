#!/bin/bash

n_tasks=2
n_jobs=4
n_subjobs=1

#SBATCH -J ssp
#SBATCH -D ./
#SBATCH -p small
#SBATCH -t 00:10:00
#SBATCH -N $n_tasks
#SBATCH --ntastks-per-node=1
#SBATCH --cpus-per-task $n_jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.capel@tum.de

module purge
module load gcc/10 anaconda/3/2020.02
module load parallel/201807

export TMPDIR=/ptmp/fran

cat seeds.txt | parallel --res output/p -j $n_tasks "srun --exclusive -N 1 -n 1 python simulation_study.py $n_jobs $n_subjobs {}" 

