# Usage Guide for VLLM with Supercluster Demo

This document explains how to use this educational repository to learn about high-performance LLM inference engines.

## Repository Structure

The repository is organized as follows:

```
├── kernels/                 # Core CUDA kernel implementations
├── engine/                  # Core inference engine components
├── distributed/             # Multi-GPU and multi-node components
├── bindings/                # Python bindings
├── examples/                # C++ example implementations
├── notebooks/               # Interactive experiments and demos
├── tutorials/               # Educational notebooks
├── scripts/                 # Cluster job submission scripts
├── docs/                    # Documentation
├── README.md                # Main documentation
├── CMakeLists.txt           # Build configuration
├── requirements.txt         # Python dependencies
└── build.sh                 # Build script
```

## Learning Path

To get the most out of this repository, follow this learning path:

1. **Environment Check** (`notebooks/00_env_check.ipynb`)
   - Verify your setup
   - Understand the project structure

2. **CUDA Basics** (`tutorials/01_intro_to_cuda.ipynb`)
   - Learn fundamental CUDA concepts
   - Understand thread hierarchy and memory management

3. **Tensor Core Basics** (`tutorials/02_tensor_cores_basics.ipynb`)
   - Learn about Tensor Cores and mixed precision
   - Understand performance optimization techniques

4. **GEMM Optimizations** (`tutorials/03_cublaslt_gemm.ipynb`)
   - Explore cuBLASLt for optimized matrix multiplication
   - Learn about GEMM performance tuning

5. **Paged Attention** (`tutorials/04_paged_attention.ipynb`)
   - Understand paged attention mechanism
   - Learn memory management optimizations

6. **Multi-GPU Communication** (`tutorials/05_nccl_multi_gpu.ipynb`)
   - Learn NCCL for multi-GPU communication
   - Understand AllReduce and AllToAll operations

7. **CUDA Graphs** (`tutorials/06_cuda_graphs_inference.ipynb`)
   - Learn about CUDA graphs for reduced overhead
   - Understand kernel fusion and optimization

8. **Scaling to Supercluster** (`tutorials/07_scaling_supercluster.ipynb`)
   - Learn about distributed inference
   - Understand supercluster scaling challenges

## Building the Project

### Prerequisites

- CUDA toolkit (11.0 or later)
- CMake (3.18 or later)
- C++ compiler with C++14 support
- Python (3.7 or later)
- PyBind11

### Building on Linux/Mac

```bash
# Make the build script executable
chmod +x build.sh

# Run the build script
./build.sh
```

### Building on Windows

```bash
# Run the build script
bash build.sh
```

### Manual Build Process

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j4
```

## Running Examples

After building, you can run the examples:

```bash
# Run single GPU inference example
./build/single_gpu_infer

# Run multi-GPU inference example (if multiple GPUs available)
./build/multi_gpu_infer

# Run latency benchmark
./build/latency_bench
```

## Using Python Bindings

The repository includes Python bindings for easier experimentation:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install the Python module
cd build
python -m pip install .
```

Then in Python:

```python
import vllm_supercluster_demo as vllm

# Use the functions
result = vllm.layer_norm(input_tensor, weight, bias)
```

## Running Notebooks

The Jupyter notebooks can be run interactively:

```bash
# Start Jupyter notebook
jupyter notebook

# Navigate to the notebooks/ directory and open any notebook
```

## Cluster Usage

For cluster environments, SLURM scripts are provided:

```bash
# Submit single GPU job
sbatch scripts/slurm/sbatch_single_gpu.sh

# Submit multi-GPU job
sbatch scripts/slurm/sbatch_multi_gpu.sh

# Submit multi-node job
sbatch scripts/slurm/sbatch_multinode.sh
```

## Educational Value

This repository is designed as a learning resource to understand:

1. **CUDA Programming**: How to write efficient GPU kernels
2. **Memory Management**: Techniques for efficient GPU memory usage
3. **Performance Optimization**: Methods to optimize inference performance
4. **Distributed Computing**: How to scale inference across multiple GPUs/nodes
5. **System Architecture**: Understanding of modern inference engine design

## Note on Hardware Requirements

This repository demonstrates concepts that typically require:
- High-end GPUs for full functionality
- Multi-GPU setups for distributed examples
- HPC clusters for supercluster scaling

The code is designed to be educational and may require modification for execution on consumer hardware. The primary goal is to understand the concepts and implementation patterns rather than to run on any specific hardware configuration.