#!/bin/bash
#SBATCH -J fit
#SBATCH -D ./
#SBATCH --array=1-10
#SBATCH -o ./logs/job.out%A_%a
#SBATCH -e ./errs/job.err%A_%a
#SBATCH --partition=general
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=512
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kuhlmann@mpp.mpg.de


n_jobs=1
n_subjobs=10
module purge
module load gcc/11
module load anaconda/3/2021.11
source /raven/u/jdk/hnu/bin/activate
export TMPDIR=/ptmp/jdk

srun --exclusive -N 1 -n 1 python simulation_study.py $n_jobs $n_subjobs $SLURM_ARRAY_TASK_ID
