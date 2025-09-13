#!/bin/bash
#SBATCH --job-name="my-job"    # Sensible name for the job
#SBATCH --account=studiegrupper-cogito # Account for consumed resources
#SBATCH --nodes=1              # Allocate 1 nodes for the job
#SBATCH -c28                   # Number of cores (can vary)
#SBATCH --time=00-00:10:00     # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=CPUQ

module purge
module load Python/3.8.6-GCCcore-10.2.0

python multicore.py 