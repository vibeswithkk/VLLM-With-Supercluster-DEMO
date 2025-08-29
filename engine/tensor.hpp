#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <cstdint>

// Enterprise-grade error handling
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// Enum for data types with size information
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64,
    BOOL
};

// Enum for memory layout
enum class MemoryLayout {
    ROW_MAJOR,
    COLUMN_MAJOR
};

// Device memory wrapper with enhanced error handling and resource management
class DeviceMemory {
private:
    void* ptr_;
    size_t size_;
    bool own_memory_;
    bool is_valid_;

public:
    // Default constructor
    DeviceMemory() : ptr_(nullptr), size_(0), own_memory_(false), is_valid_(false) {}
    
    // Constructor with size allocation
    explicit DeviceMemory(size_t size) : ptr_(nullptr), size_(size), own_memory_(true), is_valid_(false) {
        if (size > 0) {
            cudaError_t err = cudaMalloc(&ptr_, size);
            if (err == cudaSuccess) {
                is_valid_ = true;
            } else {
                throw std::runtime_error(std::string("Failed to allocate device memory: ") + cudaGetErrorString(err));
            }
        }
    }
    
    // Constructor with existing pointer
    DeviceMemory(void* ptr, size_t size) : ptr_(ptr), size_(size), own_memory_(false), is_valid_(ptr != nullptr) {
        if (ptr == nullptr && size > 0) {
            throw std::invalid_argument("Null pointer with non-zero size");
        }
    }
    
    // Destructor with proper cleanup
    ~DeviceMemory() {
        if (own_memory_ && ptr_ && is_valid_) {
            cudaFree(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
        is_valid_ = false;
    }
    
    // Copy constructor
    DeviceMemory(const DeviceMemory& other) : ptr_(nullptr), size_(other.size_), own_memory_(true), is_valid_(false) {
        if (other.is_valid_ && other.ptr_ && other.size_ > 0) {
            CUDA_SAFE_CALL(cudaMalloc(&ptr_, size_));
            CUDA_SAFE_CALL(cudaMemcpy(ptr_, other.ptr_, size_, cudaMemcpyDeviceToDevice));
            is_valid_ = true;
        }
    }
    
    // Move constructor
    DeviceMemory(DeviceMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_), 
                                                  own_memory_(other.own_memory_), is_valid_(other.is_valid_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.own_memory_ = false;
        other.is_valid_ = false;
    }
    
    // Copy assignment operator
    DeviceMemory& operator=(const DeviceMemory& other) {
        if (this != &other) {
            // Clean up existing resources
            if (own_memory_ && ptr_ && is_valid_) {
                cudaFree(ptr_);
            }
            
            ptr_ = nullptr;
            size_ = other.size_;
            own_memory_ = true;
            is_valid_ = false;
            
            if (other.is_valid_ && other.ptr_ && other.size_ > 0) {
                CUDA_SAFE_CALL(cudaMalloc(&ptr_, size_));
                CUDA_SAFE_CALL(cudaMemcpy(ptr_, other.ptr_, size_, cudaMemcpyDeviceToDevice));
                is_valid_ = true;
            }
        }
        return *this;
    }
    
    // Move assignment operator
    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            if (own_memory_ && ptr_ && is_valid_) {
                cudaFree(ptr_);
            }
            
            ptr_ = other.ptr_;
            size_ = other.size_;
            own_memory_ = other.own_memory_;
            is_valid_ = other.is_valid_;
            
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.own_memory_ = false;
            other.is_valid_ = false;
        }
        return *this;
    }
    
    // Getters
    void* get() const { return is_valid_ ? ptr_ : nullptr; }
    size_t size() const { return size_; }
    bool is_valid() const { return is_valid_; }
    
    // Copy data from host to device with validation
    void copy_from_host(const void* host_ptr, size_t copy_size) {
        if (!is_valid_ || !host_ptr) {
            throw std::invalid_argument("Invalid memory or null pointer");
        }
        
        if (copy_size > size_) {
            throw std::invalid_argument("Copy size exceeds allocated memory");
        }
        
        CUDA_SAFE_CALL(cudaMemcpy(ptr_, host_ptr, copy_size, cudaMemcpyHostToDevice));
    }
    
    // Copy data from device to host with validation
    void copy_to_host(void* host_ptr, size_t copy_size) const {
        if (!is_valid_ || !host_ptr) {
            throw std::invalid_argument("Invalid memory or null pointer");
        }
        
        if (copy_size > size_) {
            throw std::invalid_argument("Copy size exceeds allocated memory");
        }
        
        CUDA_SAFE_CALL(cudaMemcpy(host_ptr, ptr_, copy_size, cudaMemcpyDeviceToHost));
    }
    
    // Copy data from device to device with validation
    void copy_from_device(const void* device_ptr, size_t copy_size) {
        if (!is_valid_ || !device_ptr) {
            throw std::invalid_argument("Invalid memory or null pointer");
        }
        
        if (copy_size > size_) {
            throw std::invalid_argument("Copy size exceeds allocated memory");
        }
        
        CUDA_SAFE_CALL(cudaMemcpy(ptr_, device_ptr, copy_size, cudaMemcpyDeviceToDevice));
    }
    
    // Async copy from host to device
    void copy_from_host_async(const void* host_ptr, size_t copy_size, cudaStream_t stream) {
        if (!is_valid_ || !host_ptr) {
            throw std::invalid_argument("Invalid memory or null pointer");
        }
        
        if (copy_size > size_) {
            throw std::invalid_argument("Copy size exceeds allocated memory");
        }
        
        CUDA_SAFE_CALL(cudaMemcpyAsync(ptr_, host_ptr, copy_size, cudaMemcpyHostToDevice, stream));
    }
    
    // Async copy from device to host
    void copy_to_host_async(void* host_ptr, size_t copy_size, cudaStream_t stream) const {
        if (!is_valid_ || !host_ptr) {
            throw std::invalid_argument("Invalid memory or null pointer");
        }
        
        if (copy_size > size_) {
            throw std::invalid_argument("Copy size exceeds allocated memory");
        }
        
        CUDA_SAFE_CALL(cudaMemcpyAsync(host_ptr, ptr_, copy_size, cudaMemcpyDeviceToHost, stream));
    }
    
    // Fill memory with a value
    void fill(unsigned char value) {
        if (is_valid_ && ptr_) {
            CUDA_SAFE_CALL(cudaMemset(ptr_, value, size_));
        }
    }
    
    // Zero out memory
    void zero() {
        fill(0);
    }
};

// Enhanced Tensor class with comprehensive features
class Tensor {
private:
    std::shared_ptr<DeviceMemory> data_;
    std::vector<int64_t> shape_;
    DataType dtype_;
    MemoryLayout layout_;
    size_t size_;
    size_t element_size_;
    bool is_valid_;

public:
    // Default constructor
    Tensor() : data_(nullptr), dtype_(DataType::FLOAT32), layout_(MemoryLayout::ROW_MAJOR), 
               size_(0), element_size_(4), is_valid_(false) {}
    
    // Constructor with shape
    explicit Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32, 
                   MemoryLayout layout = MemoryLayout::ROW_MAJOR) 
        : shape_(shape), dtype_(dtype), layout_(layout), size_(1), element_size_(0), is_valid_(false) {
        
        // Validate shape
        for (const auto& dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Tensor dimensions must be positive");
            }
            size_ *= static_cast<size_t>(dim);
        }
        
        // Set element size based on data type
        element_size_ = get_dtype_size(dtype);
        
        // Allocate device memory
        if (size_ > 0) {
            data_ = std::make_shared<DeviceMemory>(size_ * element_size_);
            is_valid_ = data_->is_valid();
        }
    }
    
    // Constructor with existing data
    Tensor(const std::vector<int64_t>& shape, void* data_ptr, DataType dtype = DataType::FLOAT32, 
           MemoryLayout layout = MemoryLayout::ROW_MAJOR)
        : shape_(shape), dtype_(dtype), layout_(layout), size_(1), element_size_(0), is_valid_(data_ptr != nullptr) {
        
        // Validate shape
        for (const auto& dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("Tensor dimensions must be positive");
            }
            size_ *= static_cast<size_t>(dim);
        }
        
        // Set element size based on data type
        element_size_ = get_dtype_size(dtype);
        
        // Wrap existing data
        if (data_ptr && size_ > 0) {
            data_ = std::make_shared<DeviceMemory>(data_ptr, size_ * element_size_);
            is_valid_ = data_->is_valid();
        }
    }
    
    // Copy constructor
    Tensor(const Tensor& other) : shape_(other.shape_), dtype_(other.dtype_), layout_(other.layout_),
                                  size_(other.size_), element_size_(other.element_size_), is_valid_(other.is_valid_) {
        if (other.data_ && other.is_valid_) {
            data_ = std::make_shared<DeviceMemory>(*other.data_);
            is_valid_ = data_->is_valid();
        }
    }
    
    // Move constructor
    Tensor(Tensor&& other) noexcept : shape_(std::move(other.shape_)), dtype_(other.dtype_), layout_(other.layout_),
                                      size_(other.size_), element_size_(other.element_size_), is_valid_(other.is_valid_) {
        data_ = std::move(other.data_);
        other.is_valid_ = false;
    }
    
    // Copy assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            dtype_ = other.dtype_;
            layout_ = other.layout_;
            size_ = other.size_;
            element_size_ = other.element_size_;
            is_valid_ = other.is_valid_;
            
            if (other.data_ && other.is_valid_) {
                data_ = std::make_shared<DeviceMemory>(*other.data_);
                is_valid_ = data_->is_valid();
            } else {
                data_.reset();
                is_valid_ = false;
            }
        }
        return *this;
    }
    
    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            layout_ = other.layout_;
            size_ = other.size_;
            element_size_ = other.element_size_;
            is_valid_ = other.is_valid_;
            
            data_ = std::move(other.data_);
            other.is_valid_ = false;
        }
        return *this;
    }
    
    // Destructor
    ~Tensor() = default;
    
    // Getters
    void* data() const { return (data_ && is_valid_) ? data_->get() : nullptr; }
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    MemoryLayout layout() const { return layout_; }
    size_t size() const { return size_; }
    size_t element_size() const { return element_size_; }
    size_t nbytes() const { return size_ * element_size_; }
    bool is_valid() const { return is_valid_; }
    
    // Get shape dimension
    int64_t dim(int64_t index) const {
        if (index >= 0 && index < static_cast<int64_t>(shape_.size())) {
            return shape_[index];
        }
        return 1;
    }
    
    // Get number of dimensions
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    
    // Reshape tensor with validation
    void reshape(const std::vector<int64_t>& new_shape) {
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
    
    // Copy data from host with type safety
    template<typename T>
    void from_host(const std::vector<T>& host_data) {
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy to invalid tensor");
        }
        
        if (host_data.size() != size_) {
            throw std::invalid_argument("Host data size does not match tensor size");
        }
        
        if (sizeof(T) != element_size_) {
            throw std::invalid_argument("Host data type size does not match tensor element size");
        }
        
        if (data_) {
            data_->copy_from_host(host_data.data(), host_data.size() * sizeof(T));
        }
    }
    
    // Copy data to host with type safety
    template<typename T>
    std::vector<T> to_host() const {
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy from invalid tensor");
        }
        
        std::vector<T> host_data(size_);
        if (sizeof(T) != element_size_) {
            throw std::invalid_argument("Host data type size does not match tensor element size");
        }
        
        if (data_) {
            data_->copy_to_host(host_data.data(), size_ * sizeof(T));
        }
        
        return host_data;
    }
    
    // Async copy from host
    template<typename T>
    void from_host_async(const std::vector<T>& host_data, cudaStream_t stream) {
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy to invalid tensor");
        }
        
        if (host_data.size() != size_) {
            throw std::invalid_argument("Host data size does not match tensor size");
        }
        
        if (sizeof(T) != element_size_) {
            throw std::invalid_argument("Host data type size does not match tensor element size");
        }
        
        if (data_) {
            data_->copy_from_host_async(host_data.data(), host_data.size() * sizeof(T), stream);
        }
    }
    
    // Async copy to host
    template<typename T>
    void to_host_async(std::vector<T>& host_data, cudaStream_t stream) const {
        if (!is_valid_) {
            throw std::runtime_error("Cannot copy from invalid tensor");
        }
        
        if (host_data.size() != size_) {
            throw std::invalid_argument("Host data size does not match tensor size");
        }
        
        if (sizeof(T) != element_size_) {
            throw std::invalid_argument("Host data type size does not match tensor element size");
        }
        
        if (data_) {
            data_->copy_to_host_async(host_data.data(), size_ * sizeof(T), stream);
        }
    }
    
    // Slice tensor (returns a view)
    Tensor slice(int64_t start, int64_t end, int64_t dim_index = 0) const {
        if (!is_valid_) {
            throw std::runtime_error("Cannot slice invalid tensor");
        }
        
        if (dim_index < 0 || dim_index >= static_cast<int64_t>(shape_.size())) {
            throw std::invalid_argument("Invalid dimension index");
        }
        
        if (start < 0 || end > shape_[dim_index] || start >= end) {
            throw std::invalid_argument("Invalid slice range");
        }
        
        Tensor sliced = *this;
        sliced.shape_[dim_index] = end - start;
        
        // Calculate offset
        size_t offset = static_cast<size_t>(start);
        for (int64_t i = dim_index + 1; i < static_cast<int64_t>(shape_.size()); i++) {
            offset *= static_cast<size_t>(shape_[i]);
        }
        offset *= element_size_;
        
        // Create new memory view
        auto new_data = std::make_shared<DeviceMemory>(
            static_cast<char*>(data_->get()) + offset,
            sliced.size() * element_size_
        );
        sliced.data_ = new_data;
        sliced.is_valid_ = new_data->is_valid();
        
        return sliced;
    }
    
    // Fill tensor with a value
    void fill(unsigned char value) {
        if (is_valid_ && data_) {
            data_->fill(value);
        }
    }
    
    // Zero out tensor
    void zero() {
        fill(0);
    }
    
    // Clone tensor
    Tensor clone() const {
        if (!is_valid_) {
            throw std::runtime_error("Cannot clone invalid tensor");
        }
        
        Tensor cloned(shape_, dtype_, layout_);
        if (cloned.is_valid_ && data_) {
            cloned.data_->copy_from_device(data_->get(), nbytes());
        }
        return cloned;
    }
    
    // Check if tensor is contiguous in memory
    bool is_contiguous() const {
        // For now, assume row-major tensors are contiguous
        return layout_ == MemoryLayout::ROW_MAJOR;
    }
};

// Utility functions
inline size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        case DataType::BOOL: return 1;
        default: return 4;
    }
}

inline std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT16: return "float16";
        case DataType::INT32: return "int32";
        case DataType::INT64: return "int64";
        case DataType::BOOL: return "bool";
        default: return "unknown";
    }
}

inline DataType string_to_dtype(const std::string& str) {
    if (str == "float32") return DataType::FLOAT32;
    if (str == "float16") return DataType::FLOAT16;
    if (str == "int32") return DataType::INT32;
    if (str == "int64") return DataType::INT64;
    if (str == "bool") return DataType::BOOL;
    throw std::invalid_argument("Unknown data type string: " + str);
}

// Tensor factory functions
inline Tensor make_tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32) {
    return Tensor(shape, dtype);
}

inline Tensor make_tensor_like(const Tensor& other) {
    return Tensor(other.shape(), other.dtype(), other.layout());
}

// Tensor math operations (declarations)
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

// Tensor reduction operations (declarations)
Tensor sum(const Tensor& tensor, int64_t dim = -1);
Tensor mean(const Tensor& tensor, int64_t dim = -1);
Tensor max(const Tensor& tensor, int64_t dim = -1);
Tensor min(const Tensor& tensor, int64_t dim = -1);

#endif // TENSOR_HPP