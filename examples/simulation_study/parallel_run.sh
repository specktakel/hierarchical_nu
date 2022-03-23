#!/bin/bash
#SBATCH -J sspt
#SBATCH -D ./
#SBATCH --partition=general
#SBATCH -t 24:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=capel.francesca@gmail.com

n_tasks=2
n_jobs=72
n_subjobs=1

module purge
module load gcc/10
module load anaconda/3/2021.11
module load parallel/201807

export TMPDIR=/ptmp/fran
#export OMP_NUM_THREADS=1

cat seeds.txt | parallel --res output/p -j $n_tasks "srun --exclusive -N 1 -n 1 python simulation_study.py $n_jobs $n_subjobs {}"
