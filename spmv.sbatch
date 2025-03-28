#!/bin/bash
#SBATCH -A m4341
#SBATCH -C gpu                    # Use GPU nodes (A100s)
#SBATCH -q regular                # Queue type 
#SBATCH -t 00:10:00               # Time limit (HH:MM:SS)
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH -c 32                     # 32 CPU cores per task
#SBATCH --gpus-per-task=1        # 1 GPU per task

# Bind CPU threads to cores (Not sure if this is needed)
export SLURM_CPU_BIND="cores"

# Load required modules
module load cuda/11.8  # Adjust CUDA version if needed
moduel load cmake/3.30.2


# Go to build directory
cd $SLURM_SUBMIT_DIR

# Run the SpMV test with srun, needs to be in the build directory 
srun ./test_spmv ./matrices/delaunay_n22/delaunay_n22.mtx
srun ./test_spmv ./matrices/stomach/stomach.mtx
srun ./test_spmv ./matrices/n16.mtx

