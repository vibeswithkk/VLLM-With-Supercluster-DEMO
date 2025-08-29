#!/bin/bash
#SBATCH --job-name=vllm_single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=vllm_single_gpu_%j.out
#SBATCH --error=vllm_single_gpu_%j.err

# Single GPU Inference Job Script
# This script demonstrates how to run VLLM inference on a single GPU

echo "Starting VLLM Single GPU Inference Job"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load required modules (adjust based on your cluster environment)
module load cuda/11.8
module load gcc/9.4.0
module load python/3.9

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Print system information
echo "System Information:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Run the single GPU inference example
echo "Running single GPU inference example..."
echo "--------------------------------------"

# Compile the C++ example (if needed)
if [ ! -f ./build/single_gpu_infer ]; then
    echo "Compiling single GPU inference example..."
    mkdir -p build
    nvcc -o build/single_gpu_infer examples/01_single_gpu_infer.cpp \
         kernels/layernorm.cu \
         -Iengine \
         -lcublas -lcublasLt \
         -std=c++14 -O3
fi

# Run the inference example
echo "Executing inference..."
./build/single_gpu_infer

# Run Python example if available
echo "Running Python example..."
echo "------------------------"
if [ -f "notebooks/00_env_check.ipynb" ]; then
    echo "Environment check notebook found"
    # Convert and run the notebook
    jupyter nbconvert --to notebook --execute notebooks/00_env_check.ipynb --output notebooks/00_env_check_executed.ipynb
else
    echo "No Python notebooks found to execute"
fi

# Print completion information
echo "====================================="
echo "Job completed at: $(date)"
echo "Job duration: $(($(date +%s) - $START_TIME)) seconds"