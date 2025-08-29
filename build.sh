#!/bin/bash

# Build script for VLLM with Supercluster Demo
echo "VLLM with Supercluster Demo Build Script"
echo "========================================"

# Check if we're on Windows (Git Bash) or Linux/Mac
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows with Git Bash
    echo "Detected Windows environment"
    BUILD_CMD="build_win"
    IS_WINDOWS=1
else
    # Linux/Mac
    echo "Detected Unix-like environment"
    BUILD_CMD="build_unix"
    IS_WINDOWS=0
fi

build_win() {
    echo "Building on Windows..."
    
    # Check if cmake is available
    if ! command -v cmake &> /dev/null; then
        echo "Error: cmake not found. Please install CMake first."
        return 1
    fi
    
    # Check if nvcc is available
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc (CUDA compiler) not found. Please install CUDA toolkit first."
        return 1
    fi
    
    # Create build directory
    echo "Creating build directory..."
    mkdir -p build
    cd build
    
    # Configure with CMake
    echo "Configuring with CMake..."
    cmake .. -G "MinGW Makefiles"
    
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed."
        return 1
    fi
    
    # Build the project
    echo "Building project..."
    mingw32-make -j4
    
    if [ $? -ne 0 ]; then
        echo "Error: Build failed."
        return 1
    fi
    
    echo "Build completed successfully!"
    echo "Executables are in the build/ directory:"
    ls -la *.exe 2>/dev/null || echo "No executables found"
}

build_unix() {
    echo "Building on Unix-like system..."
    
    # Check if cmake is available
    if ! command -v cmake &> /dev/null; then
        echo "Error: cmake not found. Please install CMake first."
        return 1
    fi
    
    # Check if nvcc is available
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc (CUDA compiler) not found. Please install CUDA toolkit first."
        return 1
    fi
    
    # Create build directory
    echo "Creating build directory..."
    mkdir -p build
    cd build
    
    # Configure with CMake
    echo "Configuring with CMake..."
    cmake ..
    
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed."
        return 1
    fi
    
    # Build the project
    echo "Building project..."
    make -j$(nproc)
    
    if [ $? -ne 0 ]; then
        echo "Error: Build failed."
        return 1
    fi
    
    echo "Build completed successfully!"
    echo "Executables are in the build/ directory:"
    ls -la
}

# Run the appropriate build function
if [ $IS_WINDOWS -eq 1 ]; then
    build_win
else
    build_unix
fi