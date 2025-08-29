#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <memory>
#include <exception>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

// Forward declarations for our CUDA functions
extern "C" {
    void* gpu_malloc(size_t size);
    int gpu_free(void* ptr);
    size_t gpu_get_current_usage();
    size_t gpu_get_peak_usage();
    size_t gpu_get_total_allocated();
    void gpu_defragment();
    void* gpu_malloc_aligned(size_t size, size_t alignment);
    void* gpu_realloc(void* ptr, size_t new_size);
    void initialize_global_allocator(size_t initial_pool_size, size_t max_pool_size, bool enable_defragmentation);
}

// Enhanced error handling
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// Example CUDA kernel function declarations
cudaError_t launch_layernorm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int64_t rows,
    const int64_t cols,
    const float epsilon,
    cudaStream_t stream
);

cudaError_t launch_gemm_fp16_fp32(
    void* config,  // GemmConfig pointer
    float alpha,
    const void* A, int64_t m, int64_t k, int64_t lda,
    const void* B, int64_t k_, int64_t n, int64_t ldb,
    float beta,
    const float* C, int64_t m_, int64_t n_, int64_t ldc,
    float* D, int64_t m__, int64_t n__, int64_t ldd,
    const float* bias, int64_t biasSize,
    cudaStream_t stream
);

cudaError_t launch_paged_attention(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int* block_tables,
    const int* context_lens,
    const float scale,
    float* out,
    const int num_heads,
    const int head_size,
    const int num_seqs,
    const int max_seq_len,
    const int max_num_blocks_per_seq,
    const int block_size,
    cudaStream_t stream
);

// Enhanced Tensor class wrapper with comprehensive features
class PyTensor {
private:
    void* data_;
    std::vector<int64_t> shape_;
    size_t size_;
    size_t element_size_;
    bool is_valid_;
    mutable std::mutex tensor_mutex_;

public:
    // Constructor with shape
    PyTensor(const std::vector<int64_t>& shape) 
        : shape_(shape), size_(1), element_size_(sizeof(float)), is_valid_(false) {
        
        // Validate shape
        for (const auto& dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Tensor dimensions must be positive");
            }
            size_ *= static_cast<size_t>(dim);
        }
        
        // Allocate GPU memory
        if (size_ > 0) {
            data_ = gpu_malloc(size_ * element_size_);
            is_valid_ = (data_ != nullptr);
            if (!is_valid_) {
                throw std::runtime_error("Failed to allocate GPU memory for tensor");
            }
        } else {
            data_ = nullptr;
            is_valid_ = true;
        }
    }
    
    // Destructor
    ~PyTensor() {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        if (data_ && is_valid_) {
            gpu_free(data_);
        }
        data_ = nullptr;
        is_valid_ = false;
    }
    
    // Copy constructor
    PyTensor(const PyTensor& other) : shape_(other.shape_), size_(other.size_), 
                                     element_size_(other.element_size_), is_valid_(false) {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        if (other.is_valid_ && other.data_ && size_ > 0) {
            data_ = gpu_malloc(size_ * element_size_);
            is_valid_ = (data_ != nullptr);
            if (is_valid_) {
                CUDA_SAFE_CALL(cudaMemcpy(data_, other.data_, size_ * element_size_, cudaMemcpyDeviceToDevice));
            }
        }
    }
    
    // Assignment operator
    PyTensor& operator=(const PyTensor& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(tensor_mutex_);
            
            // Clean up existing resources
            if (data_ && is_valid_) {
                gpu_free(data_);
            }
            
            shape_ = other.shape_;
            size_ = other.size_;
            element_size_ = other.element_size_;
            is_valid_ = false;
            
            if (other.is_valid_ && other.data_ && size_ > 0) {
                data_ = gpu_malloc(size_ * element_size_);
                is_valid_ = (data_ != nullptr);
                if (is_valid_) {
                    CUDA_SAFE_CALL(cudaMemcpy(data_, other.data_, size_ * element_size_, cudaMemcpyDeviceToDevice));
                }
            }
        }
        return *this;
    }
    
    // Getters
    std::vector<int64_t> shape() const { 
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        return shape_; 
    }
    
    size_t size() const { 
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        return size_; 
    }
    
    size_t nbytes() const { 
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        return size_ * element_size_; 
    }
    
    bool is_valid() const { 
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        return is_valid_; 
    }
    
    // Copy data from numpy array with type checking
    void from_numpy(pybind11::array_t<float> input) {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy to invalid tensor");
        }
        
        pybind11::buffer_info buf_info = input.request();
        
        if (buf_info.size != static_cast<int64_t>(size_)) {
            throw std::runtime_error("Input array size does not match tensor size");
        }
        
        if (buf_info.ndim != static_cast<int>(shape_.size())) {
            throw std::runtime_error("Input array dimensions do not match tensor dimensions");
        }
        
        // Validate dimensions
        for (int i = 0; i < buf_info.ndim; i++) {
            if (buf_info.shape[i] != shape_[i]) {
                throw std::runtime_error("Input array shape does not match tensor shape");
            }
        }
        
        // Copy data to GPU
        CUDA_SAFE_CALL(cudaMemcpy(data_, buf_info.ptr, size_ * element_size_, cudaMemcpyHostToDevice));
    }
    
    // Copy data to numpy array
    pybind11::array_t<float> to_numpy() {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy from invalid tensor");
        }
        
        // Create numpy array with the same shape
        pybind11::array_t<float> result(shape_);
        pybind11::buffer_info buf_info = result.request();
        
        // Copy data from GPU
        CUDA_SAFE_CALL(cudaMemcpy(buf_info.ptr, data_, size_ * element_size_, cudaMemcpyDeviceToHost));
        
        return result;
    }
    
    // Async copy from numpy array
    void from_numpy_async(pybind11::array_t<float> input) {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy to invalid tensor");
        }
        
        pybind11::buffer_info buf_info = input.request();
        
        if (buf_info.size != static_cast<int64_t>(size_)) {
            throw std::runtime_error("Input array size does not match tensor size");
        }
        
        // Create CUDA stream for async operation
        cudaStream_t stream;
        CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        
        // Copy data to GPU asynchronously
        CUDA_SAFE_CALL(cudaMemcpyAsync(data_, buf_info.ptr, size_ * element_size_, cudaMemcpyHostToDevice, stream));
        
        // Synchronize stream
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        CUDA_SAFE_CALL(cudaStreamDestroy(stream));
    }
    
    // Async copy to numpy array
    pybind11::array_t<float> to_numpy_async() {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy from invalid tensor");
        }
        
        // Create numpy array with the same shape
        pybind11::array_t<float> result(shape_);
        pybind11::buffer_info buf_info = result.request();
        
        // Create CUDA stream for async operation
        cudaStream_t stream;
        CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        
        // Copy data from GPU asynchronously
        CUDA_SAFE_CALL(cudaMemcpyAsync(buf_info.ptr, data_, size_ * element_size_, cudaMemcpyDeviceToHost, stream));
        
        // Synchronize stream
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        CUDA_SAFE_CALL(cudaStreamDestroy(stream));
        
        return result;
    }
    
    // Fill tensor with a value
    void fill(float value) {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_ || !data_) {
            throw std::runtime_error("Cannot fill invalid tensor");
        }
        
        // For now, we'll use a simple approach - in a real implementation,
        // we'd use a CUDA kernel to fill the tensor efficiently
        std::vector<float> host_data(size_, value);
        CUDA_SAFE_CALL(cudaMemcpy(data_, host_data.data(), size_ * element_size_, cudaMemcpyHostToDevice));
    }
    
    // Zero out tensor
    void zero() {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_ || !data_) {
            throw std::runtime_error("Cannot zero invalid tensor");
        }
        
        CUDA_SAFE_CALL(cudaMemset(data_, 0, size_ * element_size_));
    }
    
    // Get raw data pointer (for advanced users)
    uintptr_t data_ptr() const {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        return reinterpret_cast<uintptr_t>(data_);
    }
    
    // Reshape tensor
    void reshape(const std::vector<int64_t>& new_shape) {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_) {
            throw std::runtime_error("Cannot reshape invalid tensor");
        }
        
        size_t new_size = 1;
        for (const auto& dim : new_shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Tensor dimensions must be positive");
            }
            new_size *= static_cast<size_t>(dim);
        }
        
        if (new_size != size_) {
            throw std::invalid_argument("Cannot reshape tensor to incompatible shape");
        }
        
        shape_ = new_shape;
    }
    
    // Slice tensor (returns indices for slicing)
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> slice_indices(int64_t start, int64_t end, int64_t dim_index = 0) const {
        std::lock_guard<std::mutex> lock(tensor_mutex_);
        
        if (!is_valid_) {
            throw std::runtime_error("Cannot slice invalid tensor");
        }
        
        if (dim_index < 0 || dim_index >= static_cast<int64_t>(shape_.size())) {
            throw std::invalid_argument("Invalid dimension index");
        }
        
        if (start < 0 || end > shape_[dim_index] || start >= end) {
            throw std::invalid_argument("Invalid slice range");
        }
        
        // Return start and end indices for each dimension
        std::vector<int64_t> start_indices(shape_.size(), 0);
        std::vector<int64_t> end_indices = shape_;
        
        start_indices[dim_index] = start;
        end_indices[dim_index] = end;
        
        return std::make_tuple(start_indices, end_indices);
    }
};

// Enhanced Memory allocator wrapper with comprehensive features
class PyAllocator {
private:
    static std::atomic<bool> is_initialized_;
    static std::mutex init_mutex_;

public:
    // Initialize allocator with custom parameters
    static void initialize(size_t initial_pool_size = 1024 * 1024 * 1024,  // 1GB
                          size_t max_pool_size = 8ULL * 1024 * 1024 * 1024,    // 8GB
                          bool enable_defragmentation = true) {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (!is_initialized_.load()) {
            initialize_global_allocator(initial_pool_size, max_pool_size, enable_defragmentation);
            is_initialized_.store(true);
        }
    }
    
    // Allocate memory
    static uintptr_t malloc(size_t size) {
        void* ptr = gpu_malloc(size);
        return reinterpret_cast<uintptr_t>(ptr);
    }
    
    // Allocate aligned memory
    static uintptr_t malloc_aligned(size_t size, size_t alignment) {
        void* ptr = gpu_malloc_aligned(size, alignment);
        return reinterpret_cast<uintptr_t>(ptr);
    }
    
    // Free memory
    static bool free(uintptr_t ptr) {
        return gpu_free(reinterpret_cast<void*>(ptr)) == 0;
    }
    
    // Reallocate memory
    static uintptr_t realloc(uintptr_t ptr, size_t new_size) {
        void* new_ptr = gpu_realloc(reinterpret_cast<void*>(ptr), new_size);
        return reinterpret_cast<uintptr_t>(new_ptr);
    }
    
    // Get current memory usage
    static size_t get_current_usage() {
        return gpu_get_current_usage();
    }
    
    // Get peak memory usage
    static size_t get_peak_usage() {
        return gpu_get_peak_usage();
    }
    
    // Get total allocated memory
    static size_t get_total_allocated() {
        return gpu_get_total_allocated();
    }
    
    // Defragment memory
    static void defragment() {
        gpu_defragment();
    }
    
    // Get detailed memory statistics
    static pybind11::dict get_memory_stats() {
        pybind11::dict stats;
        stats["current_usage"] = pybind11::cast(gpu_get_current_usage());
        stats["peak_usage"] = pybind11::cast(gpu_get_peak_usage());
        stats["total_allocated"] = pybind11::cast(gpu_get_total_allocated());
        return stats;
    }
};

// Initialize static members
std::atomic<bool> PyAllocator::is_initialized_{false};
std::mutex PyAllocator::init_mutex_;

// Enhanced Inference engine wrapper with comprehensive features
class PyInferenceEngine {
private:
    static std::atomic<uint64_t> operation_counter_;
    static std::mutex engine_mutex_;

public:
    // Layer normalization with comprehensive error checking
    static pybind11::array_t<float> layer_norm(
        pybind11::array_t<float> input,
        pybind11::array_t<float> weight,
        pybind11::array_t<float> bias,
        float epsilon = 1e-5f
    ) {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        
        // Get buffer info
        pybind11::buffer_info input_buf = input.request();
        pybind11::buffer_info weight_buf = weight.request();
        pybind11::buffer_info bias_buf = bias.request();
        
        // Validate input dimensions
        if (input_buf.ndim < 2) {
            throw std::runtime_error("Input must be at least 2D");
        }
        
        // Get dimensions
        int64_t rows = 1;
        for (int i = 0; i < input_buf.ndim - 1; i++) {
            rows *= input_buf.shape[i];
        }
        int64_t cols = input_buf.shape[input_buf.ndim - 1];
        
        // Validate weight and bias dimensions
        if (weight_buf.size != cols || bias_buf.size != cols) {
            throw std::runtime_error("Weight and bias must match last dimension of input");
        }
        
        // Validate data types
        if (input_buf.format != pybind11::format_descriptor<float>::format() ||
            weight_buf.format != pybind11::format_descriptor<float>::format() ||
            bias_buf.format != pybind11::format_descriptor<float>::format()) {
            throw std::runtime_error("All arrays must be of type float32");
        }
        
        // Create output array
        pybind11::array_t<float> output(input_buf.shape);
        pybind11::buffer_info output_buf = output.request();
        
        // Create CUDA stream
        cudaStream_t stream;
        CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        
        // Launch layer norm kernel
        cudaError_t err = launch_layernorm(
            static_cast<const float*>(input_buf.ptr),
            static_cast<const float*>(weight_buf.ptr),
            static_cast<const float*>(bias_buf.ptr),
            static_cast<float*>(output_buf.ptr),
            rows,
            cols,
            epsilon,
            stream
        );
        
        if (err != cudaSuccess) {
            cudaStreamDestroy(stream);
            throw std::runtime_error(std::string("Layer norm kernel failed: ") + cudaGetErrorString(err));
        }
        
        // Synchronize stream
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        CUDA_SAFE_CALL(cudaStreamDestroy(stream));
        
        operation_counter_.fetch_add(1);
        return output;
    }
    
    // Paged attention computation
    static pybind11::array_t<float> paged_attention(
        pybind11::array_t<float> query,
        pybind11::array_t<float> key_cache,
        pybind11::array_t<float> value_cache,
        pybind11::array_t<int> block_tables,
        pybind11::array_t<int> context_lens,
        float scale,
        int num_heads,
        int head_size,
        int num_seqs,
        int max_seq_len,
        int max_num_blocks_per_seq,
        int block_size
    ) {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        
        // Get buffer info
        pybind11::buffer_info query_buf = query.request();
        pybind11::buffer_info key_cache_buf = key_cache.request();
        pybind11::buffer_info value_cache_buf = value_cache.request();
        pybind11::buffer_info block_tables_buf = block_tables.request();
        pybind11::buffer_info context_lens_buf = context_lens.request();
        
        // Validate dimensions
        if (query_buf.ndim != 3) {
            throw std::runtime_error("Query must be 3D: [batch_size, num_heads, head_size]");
        }
        
        if (block_tables_buf.ndim != 2) {
            throw std::runtime_error("Block tables must be 2D: [num_seqs, max_num_blocks_per_seq]");
        }
        
        if (context_lens_buf.ndim != 1) {
            throw std::runtime_error("Context lengths must be 1D: [num_seqs]");
        }
        
        // Validate shapes
        if (query_buf.shape[1] != num_heads || query_buf.shape[2] != head_size) {
            throw std::runtime_error("Query shape does not match provided dimensions");
        }
        
        if (block_tables_buf.shape[0] != num_seqs || block_tables_buf.shape[1] != max_num_blocks_per_seq) {
            throw std::runtime_error("Block tables shape does not match provided dimensions");
        }
        
        if (context_lens_buf.shape[0] != num_seqs) {
            throw std::runtime_error("Context lengths shape does not match provided dimensions");
        }
        
        // Create output array
        std::vector<int64_t> output_shape = {query_buf.shape[0], query_buf.shape[1], query_buf.shape[2]};
        pybind11::array_t<float> output(output_shape);
        pybind11::buffer_info output_buf = output.request();
        
        // Create CUDA stream
        cudaStream_t stream;
        CUDA_SAFE_CALL(cudaStreamCreate(&stream));
        
        // Launch paged attention kernel
        cudaError_t err = launch_paged_attention(
            static_cast<const float*>(query_buf.ptr),
            static_cast<const float*>(key_cache_buf.ptr),
            static_cast<const float*>(value_cache_buf.ptr),
            static_cast<const int*>(block_tables_buf.ptr),
            static_cast<const int*>(context_lens_buf.ptr),
            scale,
            static_cast<float*>(output_buf.ptr),
            num_heads,
            head_size,
            num_seqs,
            max_seq_len,
            max_num_blocks_per_seq,
            block_size,
            stream
        );
        
        if (err != cudaSuccess) {
            cudaStreamDestroy(stream);
            throw std::runtime_error(std::string("Paged attention kernel failed: ") + cudaGetErrorString(err));
        }
        
        // Synchronize stream
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        CUDA_SAFE_CALL(cudaStreamDestroy(stream));
        
        operation_counter_.fetch_add(1);
        return output;
    }
    
    // Get engine statistics
    static pybind11::dict get_stats() {
        pybind11::dict stats;
        stats["operations_completed"] = pybind11::cast(operation_counter_.load());
        return stats;
    }
    
    // Reset engine statistics
    static void reset_stats() {
        operation_counter_.store(0);
    }
};

// Initialize static members
std::atomic<uint64_t> PyInferenceEngine::operation_counter_{0};
std::mutex PyInferenceEngine::engine_mutex_;

// Python module definition with comprehensive features
PYBIND11_MODULE(vllm_supercluster_demo, m) {
    m.doc() = "VLLM with Supercluster Demo - Enterprise-grade Python bindings for CUDA accelerated inference";
    
    // Tensor class
    pybind11::class_<PyTensor>(m, "Tensor")
        .def(pybind11::init<const std::vector<int64_t>&>(), "Create a tensor with given shape")
        .def("shape", &PyTensor::shape, "Get tensor shape")
        .def("size", &PyTensor::size, "Get tensor size")
        .def("nbytes", &PyTensor::nbytes, "Get tensor size in bytes")
        .def("is_valid", &PyTensor::is_valid, "Check if tensor is valid")
        .def("from_numpy", &PyTensor::from_numpy, "Copy data from numpy array")
        .def("to_numpy", &PyTensor::to_numpy, "Copy data to numpy array")
        .def("from_numpy_async", &PyTensor::from_numpy_async, "Copy data from numpy array asynchronously")
        .def("to_numpy_async", &PyTensor::to_numpy_async, "Copy data to numpy array asynchronously")
        .def("fill", &PyTensor::fill, "Fill tensor with a value")
        .def("zero", &PyTensor::zero, "Zero out tensor")
        .def("data_ptr", &PyTensor::data_ptr, "Get raw data pointer")
        .def("reshape", &PyTensor::reshape, "Reshape tensor")
        .def("slice_indices", &PyTensor::slice_indices, "Get slice indices");
    
    // Allocator class
    pybind11::class_<PyAllocator>(m, "Allocator")
        .def_static("initialize", &PyAllocator::initialize, 
                   pybind11::arg("initial_pool_size") = 1024 * 1024 * 1024,
                   pybind11::arg("max_pool_size") = 8ULL * 1024 * 1024 * 1024,
                   pybind11::arg("enable_defragmentation") = true,
                   "Initialize memory allocator with custom parameters")
        .def_static("malloc", &PyAllocator::malloc, "Allocate GPU memory")
        .def_static("malloc_aligned", &PyAllocator::malloc_aligned, "Allocate aligned GPU memory")
        .def_static("free", &PyAllocator::free, "Free GPU memory")
        .def_static("realloc", &PyAllocator::realloc, "Reallocate GPU memory")
        .def_static("get_current_usage", &PyAllocator::get_current_usage, "Get current GPU memory usage")
        .def_static("get_peak_usage", &PyAllocator::get_peak_usage, "Get peak GPU memory usage")
        .def_static("get_total_allocated", &PyAllocator::get_total_allocated, "Get total GPU memory allocated")
        .def_static("defragment", &PyAllocator::defragment, "Defragment GPU memory")
        .def_static("get_memory_stats", &PyAllocator::get_memory_stats, "Get detailed memory statistics");
    
    // Inference engine class
    pybind11::class_<PyInferenceEngine>(m, "InferenceEngine")
        .def_static("layer_norm", &PyInferenceEngine::layer_norm, 
                   pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"), 
                   pybind11::arg("epsilon") = 1e-5f,
                   "Apply layer normalization to input tensor")
        .def_static("paged_attention", &PyInferenceEngine::paged_attention,
                   pybind11::arg("query"), pybind11::arg("key_cache"), pybind11::arg("value_cache"),
                   pybind11::arg("block_tables"), pybind11::arg("context_lens"), pybind11::arg("scale"),
                   pybind11::arg("num_heads"), pybind11::arg("head_size"), pybind11::arg("num_seqs"),
                   pybind11::arg("max_seq_len"), pybind11::arg("max_num_blocks_per_seq"), pybind11::arg("block_size"),
                   "Compute paged attention")
        .def_static("get_stats", &PyInferenceEngine::get_stats, "Get engine statistics")
        .def_static("reset_stats", &PyInferenceEngine::reset_stats, "Reset engine statistics");
    
    // Direct function bindings
    m.def("gpu_malloc", &PyAllocator::malloc, "Allocate GPU memory");
    m.def("gpu_free", &PyAllocator::free, "Free GPU memory");
    m.def("get_gpu_memory_usage", &PyAllocator::get_current_usage, "Get current GPU memory usage");
    m.def("get_gpu_peak_usage", &PyAllocator::get_peak_usage, "Get peak GPU memory usage");
    m.def("get_gpu_total_allocated", &PyAllocator::get_total_allocated, "Get total GPU memory allocated");
    m.def("defragment_gpu_memory", &PyAllocator::defragment, "Defragment GPU memory");
    
    m.def("layer_norm", &PyInferenceEngine::layer_norm, 
          "Apply layer normalization to input tensor",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"), 
          pybind11::arg("epsilon") = 1e-5f);
    
    m.def("paged_attention", &PyInferenceEngine::paged_attention,
          "Compute paged attention",
          pybind11::arg("query"), pybind11::arg("key_cache"), pybind11::arg("value_cache"),
          pybind11::arg("block_tables"), pybind11::arg("context_lens"), pybind11::arg("scale"),
          pybind11::arg("num_heads"), pybind11::arg("head_size"), pybind11::arg("num_seqs"),
          pybind11::arg("max_seq_len"), pybind11::arg("max_num_blocks_per_seq"), pybind11::arg("block_size"));
    
    // Constants
    m.attr("VERSION") = "1.0.0";
    m.attr("CUDA_ENABLED") = true;
    m.attr("MAX_THREADS_PER_BLOCK") = 1024;
    m.attr("WARP_SIZE") = 32;
    
    // Initialize allocator by default
    try {
        PyAllocator::initialize();
    } catch (const std::exception& e) {
        // Log error but don't fail module import
        std::cerr << "Warning: Failed to initialize GPU allocator: " << e.what() << std::endl;
    }
}