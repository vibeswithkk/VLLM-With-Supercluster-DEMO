# VLLM with Supercluster Demo

<div align="center">
  
  ![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
  ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
  ![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
  ![macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)
  
  [![License](https://img.shields.io/badge/License-Learning%20License-green?style=for-the-badge)](LICENSE)
  [![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](#project-status)
  [![Contributions](https://img.shields.io/badge/Contributions-Welcome-blue?style=for-the-badge)](CONTRIBUTING.md)
  
</div>

## Abstract

This educational repository demonstrates the architectural principles and implementation techniques of high-performance Large Language Model (LLM) inference engines, from fundamental CUDA kernels to distributed inference across superclusters. Designed as a comprehensive learning resource, the project bridges the gap between academic research and production systems by providing clear, well-documented implementations of core components found in state-of-the-art inference engines like vLLM.

The repository showcases enterprise-grade distributed computing with NCCL integration, secure session management, performance monitoring, and cross-platform build support. It is structured as a progressive learning path, enabling developers, students, and researchers to understand the complexities of modern inference systems through hands-on experimentation with interactive notebooks, practical examples, and real-world implementation patterns.

>  **Educational Repository Notice**
> 
> This repository is designed exclusively for educational and demonstration purposes. The implementations showcase architectural concepts and techniques used in high-performance LLM inference systems but are not intended for production deployment on consumer hardware. 
> 
> Full functionality requires enterprise-grade GPU infrastructure (H100 to GB300 class) and may need modifications for execution on personal computing devices.
> 
>  **Important**: This project is intended for learning and research purposes only. It is not recommended for production use without significant modifications and proper security review.

## Table of Contents

- [Abstract](#abstract)
- [Repository Overview](#repository-overview)
- [Key Features](#key-features)
- [Architecture Diagram](#architecture-diagram)
- [Directory Structure](#directory-structure)
- [Learning Path](#learning-path)
- [Core Components](#core-components)
  - [CUDA Kernels](#cuda-kernels)
  - [Inference Engine](#inference-engine)
  - [Distributed Computing](#distributed-computing)
- [System Architecture](#system-architecture)
- [Development Workflow](#development-workflow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building the Project](#building-the-project)
  - [Running Examples](#running-examples)
- [Educational Value](#educational-value)
- [Project Status](#project-status)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Hardware Requirements](#hardware-requirements)

## Repository Overview

This repository serves as an educational resource for understanding how high-performance LLM inference engines work, from basic CUDA kernels to distributed inference across superclusters. It serves as a learning resource for developers, students, and researchers interested in understanding high-performance inference systems.

## Key Features

<div align="center">

| Feature | Description |
|--------:|:------------|
|  **High Performance** | Core implementations in **C++/CUDA** for maximum performance with optimized memory management and kernel execution |
|  **Python Integration** | **Python bindings** via PyBind11 for ease of use, enabling seamless integration with existing Python workflows |
|  **Interactive Learning** | **Interactive Jupyter notebooks** for hands-on learning with real examples and visualizations |
|  **Cluster Ready** | **SLURM scripts** for cluster job submission, enabling scalable distributed computing |
|  **Structured Curriculum** | Step-by-step tutorials from CUDA basics to supercluster scaling, designed for progressive learning |
|  **Distributed Computing** | **Enterprise-grade distributed computing** with NCCL integration for multi-GPU and multi-node operations |
|  **Security** | **Comprehensive security features** and performance monitoring for safe experimentation |
|  **Cross-Platform** | Support for Windows, Linux, and macOS environments with consistent build processes |

</div>

## Architecture Diagram

```
graph TB
    A[User Interface] --> B[Python Bindings]
    B --> C[Inference Engine]
    C --> D[CUDA Kernels]
    C --> E[Distributed Components]
    D --> F[GPU Acceleration]
    E --> G[NCCL Communication]
    G --> H[Multi-GPU/Node]
    
    subgraph "Core Components"
        C
        D
        E
    end
    
    subgraph "Hardware Layer"
        F
        G
        H
    end
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#fce4ec
    style E fill:#f3e5f5
    style F fill:#e0f2f1
    style G fill:#e8eaf6
    style H fill:#fff8e1
```

## Directory Structure

```
%%%
kernels/                        # Core CUDA kernel implementations
  %%% gemm_lt.cu                # cuBLASLt GEMM optimizations
  %%% layernorm.cu              # Layer normalization kernels
  %%% paged_attention.cu        # Paged attention mechanism
%%%
engine/                         # Core inference engine components
  %%% allocator.cu              # Memory allocation strategies
  %%% cuda_graphs.hpp           # CUDA graphs for reduced overhead
  %%% tensor.hpp                # Tensor operations
%%%
distributed/                    # Multi-GPU and multi-node components
  %%% inference_allreduce.hpp
  %%% inference_alltoall.hpp
  %%% nccl_env.hpp
%%%
bindings/                       # Python bindings
  %%% pybind_module.cpp
%%%
examples/                       # C++ example implementations
  %%% 01_single_gpu_infer.cpp
  %%% 02_multi_gpu_infer.cpp
  %%% 03_latency_bench.cpp
%%%
notebooks/                      # Interactive experiments and demos
  %%% 00_env_check.ipynb
  %%% 10_paged_attention.ipynb
  %%% 20_latency_vs_batch.ipynb
  %%% 30_multi_gpu_infer.ipynb
%%%
tutorials/                      # Educational notebooks
  %%% 01_intro_to_cuda.ipynb
  %%% 02_tensor_cores_basics.ipynb
  %%% 03_cublaslt_gemm.ipynb
  %%% 04_paged_attention.ipynb
  %%% 05_nccl_multi_gpu.ipynb
  %%% 06_cuda_graphs_inference.ipynb
  %%% 07_scaling_supercluster.ipynb
%%%
scripts/                        # Cluster job submission scripts
  %%% slurm/
      %%% sbatch_single_gpu.sh
      %%% sbatch_multi_gpu.sh
      %%% sbatch_multinode.sh
%%%
docs/                          # Documentation
  %%% usage.md                 # Detailed usage guide
  %%% project_summary.md       # Project completion summary
  %%% distributed_computing.md # Distributed computing documentation
%%%
CMakeLists.txt                  # Build configuration
Makefile                        # Makefile for Unix-like systems
build.sh                        # Build script for Unix-like systems
build.ps1                       # Build script for Windows
requirements.txt                # Python dependencies
README.md
```

## Learning Path

This repository is designed to be studied in sequence, with each tutorial building upon the previous concepts:

<div align="center">

###  **Beginner Level**
1. **Environment Check** (`notebooks/00_env_check.ipynb`)
2. **CUDA Fundamentals** (`tutorials/01_intro_to_cuda.ipynb`)
3. **Tensor Cores** (`tutorials/02_tensor_cores_basics.ipynb`)

###  **Intermediate Level**
4. **GEMM Optimizations** (`tutorials/03_cublaslt_gemm.ipynb`)
5. **Attention Mechanisms** (`tutorials/04_paged_attention.ipynb`)
6. **Multi-GPU Communication** (`tutorials/05_nccl_multi_gpu.ipynb`)

###  **Advanced Level**
7. **CUDA Graphs** (`tutorials/06_cuda_graphs_inference.ipynb`)
8. **Supercluster Scaling** (`tutorials/07_scaling_supercluster.ipynb`)

</div>

>  **Tip**: Start with the environment check notebook to ensure all dependencies are properly installed before proceeding with the tutorials.

## Core Components

###  CUDA Kernels

<details>
<summary>Click to expand</summary>

#### Paged Attention
Efficient attention computation with memory management

#### cuBLASLt GEMM
Optimized matrix multiplication operations

#### Layer Normalization
Performance-optimized normalization kernels

</details>

### ⚙ Inference Engine

<details>
<summary>Click to expand</summary>

#### Memory Allocator
Efficient GPU memory management

#### CUDA Graphs
Reduced kernel launch overhead for repeated operations

#### Tensor Operations
Core mathematical operations for inference

</details>

###  Distributed Computing

<details>
<summary>Click to expand</summary>

#### NCCL Integration
Multi-GPU and multi-node communication

#### AllReduce/AllToAll
Collective communication operations

#### Environment Management
Distributed computing setup with security features

</details>

## System Architecture

```
graph TD
    A[Application Layer] --> B[Python API]
    B --> C[Inference Engine]
    C --> D[Tensor Operations]
    C --> E[Memory Management]
    C --> F[CUDA Graphs]
    D --> G[CUDA Kernels]
    E --> H[GPU Memory]
    F --> I[Kernel Optimization]
    G --> J[LayerNorm]
    G --> K[GEMM]
    G --> L[Paged Attention]
    
    subgraph "Distributed Layer"
        M[NCCL Communication]
        N[AllReduce Operations]
        O[AllToAll Operations]
    end
    
    C --> M
    M --> N
    M --> O
    
    subgraph "Hardware Abstraction"
        P[GPU Cluster]
        Q[High-Speed Interconnect]
    end
    
    M --> P
    N --> P
    O --> P
    P --> Q
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#fce4ec
    style E fill:#f3e5f5
    style F fill:#e0f2f1
    style G fill:#e8eaf6
    style M fill:#fff8e1
    style P fill:#fafafa
```

## Development Workflow

```
graph LR
    A[Research & Design] --> B[Implementation]
    B --> C[Testing]
    C --> D[Documentation]
    D --> E[Review]
    E --> F[Deployment]
    F --> G[Monitoring]
    G --> H[Optimization]
    H --> A
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#fce4ec
    style E fill:#f3e5f5
    style F fill:#e0f2f1
    style G fill:#e8eaf6
    style H fill:#fff8e1
```

## End-to-End Workflow: From Start to Finish

This section details the complete workflow for using this educational repository, from initial setup to running distributed inference on a supercluster.

### 1. Initial Setup and Environment Preparation

1. **System Requirements Check**
   - Ensure you have the necessary hardware (H100 to GB300 class GPUs recommended)
   - Verify CUDA toolkit installation (11.0 or later)
   - Confirm C++ compiler with C++14 support is available
   - Check for NCCL library installation for distributed computing

2. **Dependency Installation**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # For Windows users, ensure Visual Studio Build Tools are installed
   # For Linux/Mac users, ensure build-essential is installed
   ```

3. **Repository Structure Familiarization**
   - Review the directory structure to understand component organization
   - Examine the learning path notebooks to identify your starting point
   - Check the [docs/usage.md](docs/usage.md) for platform-specific instructions

### 2. Learning Path Progression

Follow the structured learning path to build understanding progressively:

1. **Environment Verification**
   - Run `notebooks/00_env_check.ipynb` to verify your setup
   - Understand the project structure and components

2. **CUDA Fundamentals**
   - Complete `tutorials/01_intro_to_cuda.ipynb`
   - Understand thread hierarchy and memory management

3. **Core Kernel Implementation**
   - Study `tutorials/02_tensor_cores_basics.ipynb` for Tensor Cores
   - Learn cuBLASLt optimizations in `tutorials/03_cublaslt_gemm.ipynb`
   - Understand paged attention in `tutorials/04_paged_attention.ipynb`

4. **Distributed Computing**
   - Master NCCL in `tutorials/05_nccl_multi_gpu.ipynb`
   - Learn CUDA graphs in `tutorials/06_cuda_graphs_inference.ipynb`
   - Understand supercluster scaling in `tutorials/07_scaling_supercluster.ipynb`

### 3. Building the Project

Choose the appropriate build method for your platform:

**On Windows:**
```powershell
# Run the PowerShell build script
.\build.ps1
```

**On Linux/Mac with Make:**
```bash
# Use the Makefile
make all
```

**Using CMake (cross-platform):**
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j4
```

### 4. Testing and Validation

1. **Run Basic Examples**
   ```bash
   # Test single GPU inference
   ./build/single_gpu_infer
   
   # If multiple GPUs available, test multi-GPU inference
   ./build/multi_gpu_infer
   ```

2. **Execute Interactive Notebooks**
   ```bash
   # Start Jupyter notebook
   jupyter notebook
   
   # Navigate to notebooks/ and run examples
   ```

3. **Run Benchmarks**
   ```bash
   # Execute latency benchmark
   ./build/latency_bench
   ```

### 5. Python Integration

1. **Install Python Bindings**
   ```bash
   # Navigate to build directory
   cd build
   
   # Install the Python module
   python -m pip install .
   ```

2. **Use in Python Scripts**
   ```python
   import vllm_supercluster_demo as vllm
   
   # Use the functions
   result = vllm.layer_norm(input_tensor, weight, bias)
   ```

### 6. Cluster Deployment (HPC Environments)

For supercluster environments with SLURM:

1. **Single GPU Job**
   ```bash
   sbatch scripts/slurm/sbatch_single_gpu.sh
   ```

2. **Multi-GPU Job**
   ```bash
   sbatch scripts/slurm/sbatch_multi_gpu.sh
   ```

3. **Multi-Node Job**
   ```bash
   sbatch scripts/slurm/sbatch_multinode.sh
   ```

### 7. Performance Monitoring and Optimization

1. **Profile with Provided Tools**
   - Use `memory-profiler` for memory analysis
   - Apply `line-profiler` for line-by-line performance analysis
   - Monitor with `nvidia-smi` for GPU utilization

2. **Security and Compliance**
   - Run `bandit` for security scanning
   - Execute `safety check` for dependency vulnerability assessment

3. **Continuous Improvement**
   - Review performance statistics from examples
   - Optimize based on profiling results
   - Contribute improvements back to the repository

### 8. Troubleshooting Common Issues

1. **Build Failures**
   - Verify CUDA installation paths
   - Check for missing dependencies
   - Ensure compiler compatibility

2. **Runtime Errors**
   - Confirm GPU availability with `nvidia-smi`
   - Check for sufficient GPU memory
   - Validate NCCL environment variables

3. **Performance Bottlenecks**
   - Profile with CUDA profilers
   - Analyze memory usage patterns
   - Optimize data loading pipelines

This workflow ensures a comprehensive understanding of high-performance LLM inference engines, from fundamental concepts to production deployment in supercluster environments.

## Getting Started

This is an educational/demo repository designed for learning purposes. The code is structured to demonstrate concepts rather than for direct execution on consumer hardware.

###  Prerequisites

Before you begin, ensure you have the following installed:

- **Basic understanding** of CUDA programming
- **Familiarity** with C++ and Python
- **Understanding** of deep learning concepts
- **CUDA toolkit** (11.0 or later) installed
- **C++ compiler** with C++14 support
- **NCCL library** for distributed computing (optional but recommended)

### 🛠️ Building the Project

We provide multiple build options depending on your platform:

#### On Windows:
```powershell
# Run the PowerShell build script
.\build.ps1
```

#### On Linux/Mac with Make:
```bash
# Use the Makefile
make all
```

#### On Linux/Mac with Bash script:
```bash
# Make the build script executable and run it
chmod +x build.sh
./build.sh
```

#### Using CMake (cross-platform):
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j4
```

>  For detailed build instructions, see [docs/usage.md](docs/usage.md).

###  Running Examples

After building, you can run the examples:

```bash
# Run single GPU inference example (Windows)
.\build\single_gpu_infer.exe

# Run single GPU inference example (Linux/Mac)
./build/single_gpu_infer

# Run multi-GPU inference example (if NCCL is available)
./build/multi_gpu_infer

# Run latency benchmark
./build/latency_bench
```

## Educational Value

This repository bridges the gap between research papers and real-world implementations by providing:

<div align="center">

###  **Key Learning Outcomes**

| Benefit | Description |
|--------:|:------------|
|  **Clear Code Examples** | Well-documented implementations that explain complex concepts |
|  **Hands-On Experience** | Interactive notebooks for practical experimentation |
|  **Progressive Learning** | Structured curriculum from basics to advanced topics |
|  **Industry Insights** | Real-world techniques used in production inference systems |
|  **Security Best Practices** | Enterprise-grade security and reliability features |

</div>

>  **Perfect for**: Researchers, graduate students, ML engineers, and developers interested in high-performance LLM inference systems.

## Project Status

 **Completed**: This repository is now complete with all core components implemented:

<div align="center">

| Component | Status |
|----------:|:-------|
| CUDA kernel implementations (LayerNorm, GEMM, Paged Attention) | ✅ Completed |
| Inference engine components (Tensor, Allocator, CUDA Graphs) | ✅ Completed |
| Python bindings using PyBind11 | ✅ Completed |
| Example applications demonstrating usage | ✅ Completed |
| Educational tutorials covering all key concepts | ✅ Completed |
| Interactive Jupyter notebooks for hands-on learning | ✅ Completed |
| Cluster job submission scripts for HPC environments | ✅ Completed |
| Comprehensive documentation and build system | ✅ Completed |
| Cross-platform support (Windows, Linux, Mac) | ✅ Completed |
| Enterprise-grade distributed computing components | ✅ Completed |
| Security features and performance monitoring | ✅ Completed |

</div>

 For a detailed summary of what was accomplished, see [docs/project_summary.md](docs/project_summary.md).

## Contributing

We welcome contributions to the VLLM-With-Supercluster-DEMO project! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## License

This project is licensed under the Learning License v1.0 - see the [LICENSE](LICENSE) file for details.

## Hardware Requirements

This repository demonstrates concepts that typically require enterprise-grade hardware for full functionality:

###  **Recommended Configuration**
- **GPUs**: High-end GPUs (H100, A100, or similar class)
- **Multi-GPU**: 2-8 GPUs for distributed examples
- **HPC Cluster**: For supercluster scaling demonstrations

###  **Minimum Requirements**
- **GPU**: Modern CUDA-capable GPU with compute capability 7.0+
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free disk space

>  **Note**: The code is designed to be educational and may require modification for execution on consumer hardware. Some advanced features require enterprise-grade infrastructure.


For detailed usage instructions, see [docs/usage.md](docs/usage.md).

