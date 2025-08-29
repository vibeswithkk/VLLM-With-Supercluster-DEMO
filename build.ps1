# PowerShell build script for VLLM with Supercluster Demo
# This script helps Windows users build the project

Write-Host "VLLM with Supercluster Demo Build Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "kernels" -PathType Container)) {
    Write-Host "Error: kernels directory not found. Please run this script from the project root directory." -ForegroundColor Red
    exit 1
}

# Check if CUDA is installed
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = ${env:ProgramFiles} + "\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
}

if (-not (Test-Path $cudaPath -PathType Container)) {
    Write-Host "Warning: CUDA toolkit not found. Please install CUDA toolkit first." -ForegroundColor Yellow
    Write-Host "You can download it from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

# Check if nvcc is available
try {
    $nvccVersion = nvcc --version
    Write-Host "Found CUDA compiler:" -ForegroundColor Green
    $nvccVersion | Select-String "release"
} catch {
    Write-Host "Error: nvcc (CUDA compiler) not found. Please install CUDA toolkit first." -ForegroundColor Red
    exit 1
}

# Create build directory
Write-Host "Creating build directory..." -ForegroundColor Cyan
if (-not (Test-Path "build" -PathType Container)) {
    New-Item -ItemType Directory -Name "build" | Out-Null
}

# Set compiler paths
$nvcc = "nvcc"
if ($cudaPath) {
    $nvcc = Join-Path $cudaPath "bin\nvcc.exe"
}

# Include directories
$includes = "-Iengine -Ikernels -Idistributed"

# Libraries
$libs = "-lcublas -lcublasLt"

# Build single GPU inference example
Write-Host "Building single GPU inference example..." -ForegroundColor Cyan
$singleGpuCmd = "$nvcc -O3 --use_fast_math -std=c++14 $includes -o build\single_gpu_infer.exe examples\01_single_gpu_infer.cpp kernels\layernorm.cu kernels\gemm_lt.cu kernels\paged_attention.cu engine\allocator.cu $libs"
Invoke-Expression $singleGpuCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully built single_gpu_infer.exe" -ForegroundColor Green
} else {
    Write-Host "Failed to build single_gpu_infer.exe" -ForegroundColor Red
    exit 1
}

# Build latency benchmark
Write-Host "Building latency benchmark..." -ForegroundColor Cyan
$latencyBenchCmd = "$nvcc -O3 --use_fast_math -std=c++14 $includes -o build\latency_bench.exe examples\03_latency_bench.cpp kernels\layernorm.cu kernels\gemm_lt.cu kernels\paged_attention.cu engine\allocator.cu $libs"
Invoke-Expression $latencyBenchCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully built latency_bench.exe" -ForegroundColor Green
} else {
    Write-Host "Failed to build latency_bench.exe" -ForegroundColor Red
}

Write-Host ""
Write-Host "Build completed!" -ForegroundColor Green
Write-Host "Executables are located in the build directory:" -ForegroundColor Green
Get-ChildItem "build\*.exe" | ForEach-Object { Write-Host "  $($_.Name)" -ForegroundColor White }

Write-Host ""
Write-Host "To run the single GPU inference example:" -ForegroundColor Yellow
Write-Host "  .\build\single_gpu_infer.exe" -ForegroundColor White

Write-Host ""
Write-Host "To run the latency benchmark:" -ForegroundColor Yellow
Write-Host "  .\build\latency_bench.exe" -ForegroundColor White