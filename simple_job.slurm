#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="test-slurm-job"
#SBATCH --time=00:20:00
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_simple.txt
#SBATCH --error=slurm_outputs/output_simple.err
#SBATCH --mail-user=sverrnys@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"


echo "SLURM job executed"

# "sbatch ./yolo.slurm"  to run the job
