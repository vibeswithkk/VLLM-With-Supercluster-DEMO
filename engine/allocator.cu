#include <cuda_runtime.h>
#include <map>
#include <list>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <memory>
#include <cassert>
#include <cstdint>

// Enterprise-grade error handling
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return nullptr; \
    } \
} while(0)

// Memory block structure with enhanced metadata
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool is_free;
    size_t alignment;
    uint64_t allocation_id;
    uint64_t timestamp;
    
    MemoryBlock(void* p, size_t s, size_t align = 256) 
        : ptr(p), size(s), is_free(true), alignment(align), allocation_id(0), timestamp(0) {}
};

// Enhanced GPU memory allocator with improved performance and security
class GPUMemoryAllocator {
private:
    std::list<MemoryBlock> blocks_;
    mutable std::mutex mutex_;
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> peak_usage_;
    std::atomic<size_t> current_usage_;
    std::atomic<uint64_t> allocation_counter_;
    std::atomic<uint64_t> free_counter_;
    size_t initial_pool_size_;
    size_t max_pool_size_;
    bool enable_defragmentation_;
    
    // Statistics tracking
    struct AllocatorStats {
        std::atomic<uint64_t> total_allocations;
        std::atomic<uint64_t> total_frees;
        std::atomic<uint64_t> allocation_failures;
        std::atomic<uint64_t> fragmentation_events;
        std::atomic<size_t> max_block_size;
        std::atomic<size_t> min_block_size;
        
        AllocatorStats() : total_allocations(0), total_frees(0), allocation_failures(0),
                          fragmentation_events(0), max_block_size(0), min_block_size(SIZE_MAX) {}
    };
    
    AllocatorStats stats_;

public:
    // Constructor with configuration
    explicit GPUMemoryAllocator(size_t initial_pool_size = 1024 * 1024 * 1024,  // 1GB
                               size_t max_pool_size = 8ULL * 1024 * 1024 * 1024,  // 8GB
                               bool enable_defragmentation = true)
        : total_allocated_(0), peak_usage_(0), current_usage_(0), 
          allocation_counter_(0), free_counter_(0),
          initial_pool_size_(initial_pool_size), max_pool_size_(max_pool_size),
          enable_defragmentation_(enable_defragmentation) {
        
        // Initialize with a large block
        void* initial_ptr = nullptr;
        if (initial_pool_size_ > 0) {
            cudaError_t err = cudaMalloc(&initial_ptr, initial_pool_size_);
            if (err == cudaSuccess && initial_ptr) {
                blocks_.emplace_back(initial_ptr, initial_pool_size_);
                total_allocated_.store(initial_pool_size_);
                stats_.max_block_size.store(initial_pool_size_);
                stats_.min_block_size.store(initial_pool_size_);
            }
        }
    }
    
    // Destructor with proper cleanup
    ~GPUMemoryAllocator() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Free all allocated memory
        for (auto& block : blocks_) {
            if (block.ptr) {
                cudaFree(block.ptr);
            }
        }
        blocks_.clear();
    }
    
    // Allocate memory with enhanced error handling and alignment
    void* allocate(size_t size, size_t alignment = 256) {
        if (size == 0) {
            return nullptr;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Align size to specified boundary for better performance
        size = (size + alignment - 1) & ~(alignment - 1);
        
        // Check if we would exceed maximum pool size
        size_t new_total = total_allocated_.load() + size;
        if (new_total > max_pool_size_) {
            stats_.allocation_failures.fetch_add(1);
            return nullptr;
        }
        
        // Find a free block that's large enough
        for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
            if (it->is_free && it->size >= size) {
                // Split the block if it's significantly larger
                if (it->size >= size * 2) {
                    // Create a new block for the remaining space
                    void* remaining_ptr = static_cast<char*>(it->ptr) + size;
                    size_t remaining_size = it->size - size;
                    
                    blocks_.emplace(++it, remaining_ptr, remaining_size, alignment);
                    --it; // Move back to the original block
                }
                
                // Mark the block as used
                it->is_free = false;
                it->size = size;
                it->alignment = alignment;
                it->allocation_id = allocation_counter_.fetch_add(1);
                it->timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count());
                
                // Update usage statistics
                size_t current_usage = current_usage_.fetch_add(size) + size;
                size_t peak = peak_usage_.load();
                while (peak < current_usage && !peak_usage_.compare_exchange_weak(peak, current_usage));
                
                stats_.total_allocations.fetch_add(1);
                
                return it->ptr;
            }
        }
        
        // If no suitable block found, allocate a new one
        void* new_ptr = nullptr;
        cudaError_t err = cudaMalloc(&new_ptr, size);
        if (err == cudaSuccess && new_ptr) {
            blocks_.emplace_back(new_ptr, size, alignment);
            auto it = blocks_.end();
            --it;
            it->is_free = false;
            it->allocation_id = allocation_counter_.fetch_add(1);
            it->timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count());
            
            total_allocated_.fetch_add(size);
            size_t current_usage = current_usage_.fetch_add(size) + size;
            size_t peak = peak_usage_.load();
            while (peak < current_usage && !peak_usage_.compare_exchange_weak(peak, current_usage));
            
            // Update block size statistics
            size_t current_max = stats_.max_block_size.load();
            while (size > current_max && !stats_.max_block_size.compare_exchange_weak(current_max, size));
            
            size_t current_min = stats_.min_block_size.load();
            while (size < current_min && !stats_.min_block_size.compare_exchange_weak(current_min, size));
            
            stats_.total_allocations.fetch_add(1);
            
            return new_ptr;
        }
        
        stats_.allocation_failures.fetch_add(1);
        return nullptr;
    }
    
    // Free memory with enhanced error handling
    bool free(void* ptr) {
        if (!ptr) return true; // Freeing null pointer is valid
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find the block containing this pointer
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                if (block.is_free) {
                    // Double free detected
                    fprintf(stderr, "Warning: Attempt to free already freed memory at %p\n", ptr);
                    return false;
                }
                
                block.is_free = true;
                current_usage_.fetch_sub(block.size);
                free_counter_.fetch_add(1);
                stats_.total_frees.fetch_add(1);
                
                // Optionally defragment memory
                if (enable_defragmentation_ && (free_counter_.load() % 100 == 0)) {
                    defragment_internal();
                }
                
                return true;
            }
        }
        
        // If not found in our blocks, it might be a direct CUDA allocation
        // This is potentially dangerous but we'll allow it for compatibility
        cudaFree(ptr);
        return true;
    }
    
    // Get current memory usage
    size_t get_current_usage() const {
        return current_usage_.load();
    }
    
    // Get peak memory usage
    size_t get_peak_usage() const {
        return peak_usage_.load();
    }
    
    // Get total allocated memory
    size_t get_total_allocated() const {
        return total_allocated_.load();
    }
    
    // Get allocation statistics
    struct AllocationStats {
        uint64_t total_allocations;
        uint64_t total_frees;
        uint64_t allocation_failures;
        uint64_t fragmentation_events;
        size_t current_usage;
        size_t peak_usage;
        size_t total_allocated;
        size_t max_block_size;
        size_t min_block_size;
    };
    
    AllocationStats get_stats() const {
        AllocationStats stats;
        stats.total_allocations = stats_.total_allocations.load();
        stats.total_frees = stats_.total_frees.load();
        stats.allocation_failures = stats_.allocation_failures.load();
        stats.fragmentation_events = stats_.fragmentation_events.load();
        stats.current_usage = current_usage_.load();
        stats.peak_usage = peak_usage_.load();
        stats.total_allocated = total_allocated_.load();
        stats.max_block_size = stats_.max_block_size.load();
        stats.min_block_size = stats_.min_block_size.load();
        return stats;
    }
    
    // Reset statistics
    void reset_stats() {
        stats_.total_allocations.store(0);
        stats_.total_frees.store(0);
        stats_.allocation_failures.store(0);
        stats_.fragmentation_events.store(0);
        // Don't reset usage statistics as they reflect actual memory state
    }
    
    // Defragment memory (merge adjacent free blocks)
    void defragment() {
        std::lock_guard<std::mutex> lock(mutex_);
        defragment_internal();
    }
    
private:
    // Internal defragmentation implementation
    void defragment_internal() {
        // Sort blocks by pointer address
        blocks_.sort([](const MemoryBlock& a, const MemoryBlock& b) {
            return a.ptr < b.ptr;
        });
        
        // Merge adjacent free blocks
        auto it = blocks_.begin();
        while (it != blocks_.end()) {
            auto next = std::next(it);
            if (next != blocks_.end() && 
                it->is_free && next->is_free &&
                static_cast<char*>(it->ptr) + it->size == next->ptr) {
                // Merge blocks
                it->size += next->size;
                blocks_.erase(next);
                stats_.fragmentation_events.fetch_add(1);
            } else {
                ++it;
            }
        }
    }
    
public:
    // Allocate aligned memory
    void* allocate_aligned(size_t size, size_t alignment) {
        return allocate(size, alignment);
    }
    
    // Reallocate memory (similar to realloc)
    void* reallocate(void* ptr, size_t new_size) {
        if (!ptr) {
            return allocate(new_size);
        }
        
        if (new_size == 0) {
            free(ptr);
            return nullptr;
        }
        
        // Find the existing block
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = blocks_.end();
        for (auto block_it = blocks_.begin(); block_it != blocks_.end(); ++block_it) {
            if (block_it->ptr == ptr) {
                it = block_it;
                break;
            }
        }
        
        if (it == blocks_.end()) {
            // Pointer not found in our blocks
            return nullptr;
        }
        
        // If new size is smaller or equal and within same block, just return the same pointer
        if (new_size <= it->size) {
            return ptr;
        }
        
        // Need to allocate new memory and copy
        void* new_ptr = allocate(new_size, it->alignment);
        if (new_ptr) {
            // Copy existing data
            CUDA_SAFE_CALL(cudaMemcpy(new_ptr, ptr, it->size, cudaMemcpyDeviceToDevice));
            // Free old memory
            free(ptr);
        }
        
        return new_ptr;
    }
    
    // Get memory block information for debugging
    struct BlockInfo {
        void* ptr;
        size_t size;
        bool is_free;
        size_t alignment;
        uint64_t allocation_id;
        uint64_t timestamp;
    };
    
    std::vector<BlockInfo> get_block_info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<BlockInfo> info;
        info.reserve(blocks_.size());
        
        for (const auto& block : blocks_) {
            BlockInfo bi;
            bi.ptr = block.ptr;
            bi.size = block.size;
            bi.is_free = block.is_free;
            bi.alignment = block.alignment;
            bi.allocation_id = block.allocation_id;
            bi.timestamp = block.timestamp;
            info.push_back(bi);
        }
        
        return info;
    }
};

// Global allocator instance with thread-safe access
static std::unique_ptr<GPUMemoryAllocator> g_allocator_instance;
static std::once_flag g_allocator_init_flag;

// Initialize global allocator
void initialize_global_allocator(size_t initial_pool_size = 1024 * 1024 * 1024,
                                size_t max_pool_size = 8ULL * 1024 * 1024 * 1024,
                                bool enable_defragmentation = true) {
    std::call_once(g_allocator_init_flag, [initial_pool_size, max_pool_size, enable_defragmentation]() {
        g_allocator_instance = std::make_unique<GPUMemoryAllocator>(
            initial_pool_size, max_pool_size, enable_defragmentation);
    });
}

// Get global allocator instance
GPUMemoryAllocator& get_global_allocator() {
    if (!g_allocator_instance) {
        initialize_global_allocator();
    }
    return *g_allocator_instance;
}

// C-style interface functions with enhanced error handling
extern "C" {
    void* gpu_malloc(size_t size) {
        try {
            return get_global_allocator().allocate(size);
        } catch (...) {
            return nullptr;
        }
    }
    
    void* gpu_malloc_aligned(size_t size, size_t alignment) {
        try {
            return get_global_allocator().allocate_aligned(size, alignment);
        } catch (...) {
            return nullptr;
        }
    }
    
    int gpu_free(void* ptr) {
        try {
            return get_global_allocator().free(ptr) ? 0 : -1;
        } catch (...) {
            return -1;
        }
    }
    
    size_t gpu_get_current_usage() {
        try {
            return get_global_allocator().get_current_usage();
        } catch (...) {
            return 0;
        }
    }
    
    size_t gpu_get_peak_usage() {
        try {
            return get_global_allocator().get_peak_usage();
        } catch (...) {
            return 0;
        }
    }
    
    size_t gpu_get_total_allocated() {
        try {
            return get_global_allocator().get_total_allocated();
        } catch (...) {
            return 0;
        }
    }
    
    void gpu_defragment() {
        try {
            get_global_allocator().defragment();
        } catch (...) {
            // Silently ignore errors in C interface
        }
    }
    
    void* gpu_realloc(void* ptr, size_t new_size) {
        try {
            return get_global_allocator().reallocate(ptr, new_size);
        } catch (...) {
            return nullptr;
        }
    }
}

// Allocator statistics
void print_allocator_stats() {
    try {
        auto stats = get_global_allocator().get_stats();
        std::cout << "GPU Memory Allocator Statistics:" << std::endl;
        std::cout << "  Current Usage: " << stats.current_usage << " bytes" << std::endl;
        std::cout << "  Peak Usage: " << stats.peak_usage << " bytes" << std::endl;
        std::cout << "  Total Allocated: " << stats.total_allocated << " bytes" << std::endl;
        std::cout << "  Total Allocations: " << stats.total_allocations << std::endl;
        std::cout << "  Total Frees: " << stats.total_frees << std::endl;
        std::cout << "  Allocation Failures: " << stats.allocation_failures << std::endl;
        std::cout << "  Fragmentation Events: " << stats.fragmentation_events << std::endl;
        std::cout << "  Max Block Size: " << stats.max_block_size << " bytes" << std::endl;
        std::cout << "  Min Block Size: " << stats.min_block_size << " bytes" << std::endl;
    } catch (...) {
        std::cout << "Error retrieving allocator statistics" << std::endl;
    }
}

// Memory pool management
class MemoryPool {
private:
    GPUMemoryAllocator allocator_;
    size_t reserved_size_;
    
public:
    explicit MemoryPool(size_t pool_size, size_t reserved_size = 0)
        : allocator_(pool_size, pool_size, true), reserved_size_(reserved_size) {}
    
    void* allocate(size_t size) {
        return allocator_.allocate(size);
    }
    
    bool free(void* ptr) {
        return allocator_.free(ptr);
    }
    
    size_t get_available_memory() const {
        return allocator_.get_total_allocated() - allocator_.get_current_usage() - reserved_size_;
    }
    
    GPUMemoryAllocator& get_allocator() {
        return allocator_;
    }
};