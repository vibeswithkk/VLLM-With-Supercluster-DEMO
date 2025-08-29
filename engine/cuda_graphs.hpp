#ifndef CUDA_GRAPHS_HPP
#define CUDA_GRAPHS_HPP

#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cassert>
#include <cstdint>

// Enterprise-grade error handling
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
} while(0)

// CUDA Graph wrapper class with enhanced features
class CudaGraph {
private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool is_captured_;
    bool is_executable_;
    bool is_valid_;
    cudaStream_t capture_stream_;
    uint64_t graph_id_;
    std::chrono::high_resolution_clock::time_point creation_time_;
    
    // Statistics
    struct GraphStats {
        std::atomic<uint64_t> execution_count;
        std::atomic<uint64_t> total_execution_time_ns;
        std::atomic<uint64_t> min_execution_time_ns;
        std::atomic<uint64_t> max_execution_time_ns;
        
        GraphStats() : execution_count(0), total_execution_time_ns(0), 
                      min_execution_time_ns(UINT64_MAX), max_execution_time_ns(0) {}
    };
    
    std::unique_ptr<GraphStats> stats_;

public:
    // Constructor
    CudaGraph(uint64_t graph_id = 0) 
        : graph_(nullptr), graph_exec_(nullptr), is_captured_(false), is_executable_(false),
          is_valid_(false), capture_stream_(nullptr), graph_id_(graph_id) {
        stats_ = std::make_unique<GraphStats>();
        creation_time_ = std::chrono::high_resolution_clock::now();
    }
    
    // Destructor with proper cleanup
    ~CudaGraph() {
        cleanup();
    }
    
    // Cleanup resources
    void cleanup() {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        is_captured_ = false;
        is_executable_ = false;
        is_valid_ = false;
    }
    
    // Begin capturing a graph
    cudaError_t begin_capture(cudaStream_t stream, cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {
        if (is_captured_ || is_executable_) {
            return cudaErrorInvalidValue;
        }
        
        CUDA_SAFE_CALL(cudaStreamBeginCapture(stream, mode));
        
        capture_stream_ = stream;
        is_valid_ = true;
        return cudaSuccess;
    }
    
    // End capturing a graph
    cudaError_t end_capture(cudaStream_t stream) {
        if (!is_valid_ || is_captured_ || !capture_stream_ || capture_stream_ != stream) {
            return cudaErrorInvalidValue;
        }
        
        CUDA_SAFE_CALL(cudaStreamEndCapture(stream, &graph_));
        
        is_captured_ = true;
        capture_stream_ = nullptr;
        return cudaSuccess;
    }
    
    // Instantiate the graph for execution with error handling
    cudaError_t instantiate() {
        if (!is_captured_ || is_executable_ || !graph_) {
            return cudaErrorInvalidValue;
        }
        
        // Try to instantiate with error node reporting
        cudaGraphExec_t exec = nullptr;
        cudaError_t err = cudaGraphInstantiate(&exec, graph_, nullptr, nullptr, 0);
        
        if (err != cudaSuccess) {
            // Try to get more detailed error information
            char* error_node_log = nullptr;
            size_t error_log_size = 0;
            
            cudaGraphInstantiate(&exec, graph_, &error_node_log, &error_log_size, 1024);
            if (error_node_log && error_log_size > 0) {
                fprintf(stderr, "Graph instantiation error: %s\n", error_node_log);
            }
            
            return err;
        }
        
        graph_exec_ = exec;
        is_executable_ = true;
        return cudaSuccess;
    }
    
    // Execute the graph with performance monitoring
    cudaError_t execute(cudaStream_t stream) {
        if (!is_executable_ || !graph_exec_) {
            return cudaErrorInvalidValue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec_, stream));
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Update statistics
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        stats_->execution_count.fetch_add(1);
        stats_->total_execution_time_ns.fetch_add(duration_ns);
        
        uint64_t current_min = stats_->min_execution_time_ns.load();
        while (duration_ns < current_min && 
               !stats_->min_execution_time_ns.compare_exchange_weak(current_min, duration_ns));
        
        uint64_t current_max = stats_->max_execution_time_ns.load();
        while (duration_ns > current_max && 
               !stats_->max_execution_time_ns.compare_exchange_weak(current_max, duration_ns));
        
        return cudaSuccess;
    }
    
    // Execute the graph and synchronize
    cudaError_t execute_and_sync(cudaStream_t stream) {
        cudaError_t err = execute(stream);
        if (err != cudaSuccess) {
            return err;
        }
        return cudaStreamSynchronize(stream);
    }
    
    // Getters
    bool is_captured() const { return is_captured_; }
    bool is_executable() const { return is_executable_; }
    bool is_valid() const { return is_valid_; }
    uint64_t get_graph_id() const { return graph_id_; }
    
    // Get graph statistics
    struct ExecutionStats {
        uint64_t execution_count;
        double avg_execution_time_ms;
        double min_execution_time_ms;
        double max_execution_time_ms;
        double total_execution_time_ms;
    };
    
    ExecutionStats get_stats() const {
        ExecutionStats stats;
        stats.execution_count = stats_->execution_count.load();
        
        uint64_t total_ns = stats_->total_execution_time_ns.load();
        stats.total_execution_time_ms = total_ns / 1000000.0;
        
        if (stats.execution_count > 0) {
            stats.avg_execution_time_ms = stats.total_execution_time_ms / stats.execution_count;
        } else {
            stats.avg_execution_time_ms = 0.0;
        }
        
        uint64_t min_ns = stats_->min_execution_time_ns.load();
        stats.min_execution_time_ms = (min_ns != UINT64_MAX) ? min_ns / 1000000.0 : 0.0;
        
        uint64_t max_ns = stats_->max_execution_time_ns.load();
        stats.max_execution_time_ms = max_ns / 1000000.0;
        
        return stats;
    }
    
    // Reset statistics
    void reset_stats() {
        stats_->execution_count.store(0);
        stats_->total_execution_time_ns.store(0);
        stats_->min_execution_time_ns.store(UINT64_MAX);
        stats_->max_execution_time_ns.store(0);
    }
    
    // Update graph (for parameter changes)
    cudaError_t update_graph(const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t num_nodes) {
        if (!is_executable_ || !graph_exec_) {
            return cudaErrorInvalidValue;
        }
        
        return cudaGraphExecUpdate(graph_exec_, graph_, from, to, num_nodes);
    }
    
    // Upload parameters to graph
    cudaError_t upload_params(const void* symbol, const void* src, size_t size) {
        if (!is_executable_ || !graph_exec_) {
            return cudaErrorInvalidValue;
        }
        
        return cudaGraphExecMemcpyNodeSetParams(graph_exec_, nullptr, src, symbol, size, cudaMemcpyDefault);
    }
};

// Enhanced Graph execution cache for reusing graphs with similar parameters
class GraphCache {
private:
    struct GraphKey {
        size_t batch_size;
        size_t seq_len;
        size_t hidden_size;
        size_t num_heads;
        uint32_t hash;
        
        GraphKey(size_t bs, size_t sl, size_t hs, size_t nh) 
            : batch_size(bs), seq_len(sl), hidden_size(hs), num_heads(nh) {
            // Simple hash function
            hash = static_cast<uint32_t>((batch_size * 1000000 + seq_len * 1000 + hidden_size + num_heads) & 0xFFFFFFFF);
        }
        
        bool operator<(const GraphKey& other) const {
            if (batch_size != other.batch_size) return batch_size < other.batch_size;
            if (seq_len != other.seq_len) return seq_len < other.seq_len;
            if (hidden_size != other.hidden_size) return hidden_size < other.hidden_size;
            return num_heads < other.num_heads;
        }
        
        bool operator==(const GraphKey& other) const {
            return batch_size == other.batch_size && seq_len == other.seq_len && 
                   hidden_size == other.hidden_size && num_heads == other.num_heads;
        }
    };
    
    struct GraphEntry {
        std::unique_ptr<CudaGraph> graph;
        std::chrono::high_resolution_clock::time_point last_used;
        size_t hit_count;
        
        GraphEntry(std::unique_ptr<CudaGraph> g) 
            : graph(std::move(g)), last_used(std::chrono::high_resolution_clock::now()), hit_count(0) {}
    };
    
    std::map<GraphKey, GraphEntry> cache_;
    mutable std::mutex cache_mutex_;
    std::atomic<uint64_t> graph_id_counter_;
    std::atomic<size_t> cache_hits_;
    std::atomic<size_t> cache_misses_;
    size_t max_cache_size_;
    static const size_t kDefaultMaxCacheSize = 64;
    
    // Cache statistics
    struct CacheStats {
        std::atomic<uint64_t> total_requests;
        std::atomic<uint64_t> evictions;
        std::atomic<uint64_t> insertions;
        
        CacheStats() : total_requests(0), evictions(0), insertions(0) {}
    };
    
    std::unique_ptr<CacheStats> stats_;

public:
    // Constructor
    explicit GraphCache(size_t max_cache_size = kDefaultMaxCacheSize) 
        : max_cache_size_(max_cache_size), graph_id_counter_(0) {
        stats_ = std::make_unique<CacheStats>();
    }
    
    // Get or create a graph for specific parameters
    CudaGraph* get_graph(size_t batch_size, size_t seq_len, size_t hidden_size, size_t num_heads) {
        GraphKey key(batch_size, seq_len, hidden_size, num_heads);
        
        std::lock_guard<std::mutex> lock(cache_mutex_);
        stats_->total_requests.fetch_add(1);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Found in cache, update timestamp and hit count
            it->second.last_used = std::chrono::high_resolution_clock::now();
            it->second.hit_count++;
            cache_hits_.fetch_add(1);
            return it->second.graph.get();
        }
        
        // Not in cache
        cache_misses_.fetch_add(1);
        
        // Check if we need to evict
        if (cache_.size() >= max_cache_size_) {
            // Remove least recently used
            auto lru_it = cache_.begin();
            auto oldest_time = lru_it->second.last_used;
            
            for (auto cit = cache_.begin(); cit != cache_.end(); ++cit) {
                if (cit->second.last_used < oldest_time) {
                    oldest_time = cit->second.last_used;
                    lru_it = cit;
                }
            }
            
            cache_.erase(lru_it);
            stats_->evictions.fetch_add(1);
        }
        
        // Create new graph entry
        uint64_t graph_id = graph_id_counter_.fetch_add(1);
        auto new_graph = std::make_unique<CudaGraph>(graph_id);
        auto result = cache_.emplace(std::make_pair(key, GraphEntry(std::move(new_graph))));
        stats_->insertions.fetch_add(1);
        
        return result.first->second.graph.get();
    }
    
    // Clear the cache
    void clear() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_.clear();
    }
    
    // Get cache statistics
    struct CacheStatistics {
        size_t cache_size;
        size_t cache_hits;
        size_t cache_misses;
        double hit_rate;
        uint64_t total_requests;
        uint64_t evictions;
        uint64_t insertions;
    };
    
    CacheStatistics get_stats() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        CacheStatistics stats;
        stats.cache_size = cache_.size();
        stats.cache_hits = cache_hits_.load();
        stats.cache_misses = cache_misses_.load();
        stats.total_requests = stats_->total_requests.load();
        stats.evictions = stats_->evictions.load();
        stats.insertions = stats_->insertions.load();
        
        if (stats.total_requests > 0) {
            stats.hit_rate = static_cast<double>(stats.cache_hits) / stats.total_requests;
        } else {
            stats.hit_rate = 0.0;
        }
        
        return stats;
    }
    
    // Reset cache statistics
    void reset_stats() {
        cache_hits_.store(0);
        cache_misses_.store(0);
        stats_->total_requests.store(0);
        stats_->evictions.store(0);
        stats_->insertions.store(0);
    }
    
    // Get detailed cache contents for debugging
    struct CacheEntryInfo {
        size_t batch_size;
        size_t seq_len;
        size_t hidden_size;
        size_t num_heads;
        uint64_t graph_id;
        size_t hit_count;
        std::chrono::milliseconds time_since_last_use;
    };
    
    std::vector<CacheEntryInfo> get_cache_contents() const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        std::vector<CacheEntryInfo> contents;
        contents.reserve(cache_.size());
        
        auto now = std::chrono::high_resolution_clock::now();
        
        for (const auto& entry : cache_) {
            CacheEntryInfo info;
            info.batch_size = entry.first.batch_size;
            info.seq_len = entry.first.seq_len;
            info.hidden_size = entry.first.hidden_size;
            info.num_heads = entry.first.num_heads;
            info.graph_id = entry.second.graph->get_graph_id();
            info.hit_count = entry.second.hit_count;
            info.time_since_last_use = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - entry.second.last_used);
            contents.push_back(info);
        }
        
        return contents;
    }
};

// Enhanced Context for managing CUDA graphs in the inference engine
class GraphContext {
private:
    cudaStream_t stream_;
    std::unique_ptr<GraphCache> cache_;
    bool capture_mode_;
    std::unique_ptr<CudaGraph> current_graph_;
    uint64_t context_id_;
    std::chrono::high_resolution_clock::time_point creation_time_;
    
    // Context statistics
    struct ContextStats {
        std::atomic<uint64_t> graphs_created;
        std::atomic<uint64_t> graphs_executed;
        std::atomic<uint64_t> capture_operations;
        
        ContextStats() : graphs_created(0), graphs_executed(0), capture_operations(0) {}
    };
    
    std::unique_ptr<ContextStats> stats_;

public:
    // Constructor
    explicit GraphContext(uint64_t context_id = 0) 
        : capture_mode_(false), context_id_(context_id) {
        CUDA_SAFE_CALL(cudaStreamCreate(&stream_));
        cache_ = std::make_unique<GraphCache>();
        stats_ = std::make_unique<ContextStats>();
        creation_time_ = std::chrono::high_resolution_clock::now();
    }
    
    // Destructor
    ~GraphContext() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Get the CUDA stream
    cudaStream_t stream() const { return stream_; }
    
    // Begin graph capture mode
    cudaError_t begin_capture(cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {
        if (capture_mode_) {
            return cudaErrorInvalidValue;
        }
        
        capture_mode_ = true;
        current_graph_ = std::make_unique<CudaGraph>(context_id_);
        stats_->capture_operations.fetch_add(1);
        
        return current_graph_->begin_capture(stream_, mode);
    }
    
    // End graph capture mode
    cudaError_t end_capture() {
        if (!capture_mode_ || !current_graph_) {
            return cudaErrorInvalidValue;
        }
        
        cudaError_t err = current_graph_->end_capture(stream_);
        if (err != cudaSuccess) {
            capture_mode_ = false;
            current_graph_.reset();
            return err;
        }
        
        err = current_graph_->instantiate();
        if (err != cudaSuccess) {
            capture_mode_ = false;
            current_graph_.reset();
            return err;
        }
        
        capture_mode_ = false;
        stats_->graphs_created.fetch_add(1);
        return cudaSuccess;
    }
    
    // Execute a graph with caching
    cudaError_t execute_graph(size_t batch_size, size_t seq_len, size_t hidden_size, size_t num_heads) {
        if (capture_mode_) {
            return cudaErrorInvalidValue;
        }
        
        CudaGraph* graph = cache_->get_graph(batch_size, seq_len, hidden_size, num_heads);
        if (graph && graph->is_executable()) {
            cudaError_t err = graph->execute(stream_);
            if (err == cudaSuccess) {
                stats_->graphs_executed.fetch_add(1);
            }
            return err;
        }
        
        return cudaErrorInvalidValue;
    }
    
    // Execute a graph and synchronize
    cudaError_t execute_graph_and_sync(size_t batch_size, size_t seq_len, size_t hidden_size, size_t num_heads) {
        cudaError_t err = execute_graph(batch_size, seq_len, hidden_size, num_heads);
        if (err != cudaSuccess) {
            return err;
        }
        return cudaStreamSynchronize(stream_);
    }
    
    // Get context ID
    uint64_t get_context_id() const { return context_id_; }
    
    // Get cache
    GraphCache& get_cache() { return *cache_; }
    const GraphCache& get_cache() const { return *cache_; }
    
    // Check if in capture mode
    bool is_capture_mode() const { return capture_mode_; }
    
    // Get context statistics
    struct ContextStatistics {
        uint64_t context_id;
        uint64_t graphs_created;
        uint64_t graphs_executed;
        uint64_t capture_operations;
        std::chrono::seconds uptime;
        GraphCache::CacheStatistics cache_stats;
    };
    
    ContextStatistics get_stats() const {
        ContextStatistics stats;
        stats.context_id = context_id_;
        stats.graphs_created = stats_->graphs_created.load();
        stats.graphs_executed = stats_->graphs_executed.load();
        stats.capture_operations = stats_->capture_operations.load();
        
        auto now = std::chrono::high_resolution_clock::now();
        stats.uptime = std::chrono::duration_cast<std::chrono::seconds>(now - creation_time_);
        
        stats.cache_stats = cache_->get_stats();
        
        return stats;
    }
    
    // Reset statistics
    void reset_stats() {
        stats_->graphs_created.store(0);
        stats_->graphs_executed.store(0);
        stats_->capture_operations.store(0);
        cache_->reset_stats();
    }
};

// Utility functions for common graph operations
namespace graph_utils {
    
    // Record an event in the graph
    inline cudaError_t record_event(cudaStream_t stream, cudaEvent_t event) {
        return cudaEventRecord(event, stream);
    }
    
    // Wait for an event in the graph
    inline cudaError_t wait_event(cudaStream_t stream, cudaEvent_t event) {
        return cudaStreamWaitEvent(stream, event, 0);
    }
    
    // Synchronize the stream
    inline cudaError_t synchronize_stream(cudaStream_t stream) {
        return cudaStreamSynchronize(stream);
    }
    
    // Create a memory copy node in a graph
    inline cudaError_t add_memcpy_node(cudaGraph_t graph, cudaGraphNode_t* node_out,
                                      const cudaGraphNode_t* dependencies, size_t num_dependencies,
                                      const cudaMemcpy3DParms* copy_params) {
        return cudaGraphAddMemcpyNode(node_out, graph, dependencies, num_dependencies, copy_params);
    }
    
    // Create a kernel node in a graph
    inline cudaError_t add_kernel_node(cudaGraph_t graph, cudaGraphNode_t* node_out,
                                      const cudaGraphNode_t* dependencies, size_t num_dependencies,
                                      const cudaKernelNodeParams* node_params) {
        return cudaGraphAddKernelNode(node_out, graph, dependencies, num_dependencies, node_params);
    }
    
    // Create a memset node in a graph
    inline cudaError_t add_memset_node(cudaGraph_t graph, cudaGraphNode_t* node_out,
                                      const cudaGraphNode_t* dependencies, size_t num_dependencies,
                                      const cudaMemsetParams* memset_params) {
        return cudaGraphAddMemsetNode(node_out, graph, dependencies, num_dependencies, memset_params);
    }
}

// Graph manager for managing multiple graph contexts
class GraphManager {
private:
    std::map<uint64_t, std::unique_ptr<GraphContext>> contexts_;
    mutable std::mutex manager_mutex_;
    std::atomic<uint64_t> context_id_counter_;
    
public:
    // Constructor
    GraphManager() : context_id_counter_(0) {}
    
    // Create a new graph context
    uint64_t create_context() {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        uint64_t context_id = context_id_counter_.fetch_add(1);
        contexts_[context_id] = std::make_unique<GraphContext>(context_id);
        return context_id;
    }
    
    // Get a graph context
    GraphContext* get_context(uint64_t context_id) {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        auto it = contexts_.find(context_id);
        if (it != contexts_.end()) {
            return it->second.get();
        }
        return nullptr;
    }
    
    // Destroy a graph context
    bool destroy_context(uint64_t context_id) {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        auto it = contexts_.find(context_id);
        if (it != contexts_.end()) {
            contexts_.erase(it);
            return true;
        }
        return false;
    }
    
    // Get all context IDs
    std::vector<uint64_t> get_context_ids() const {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        std::vector<uint64_t> ids;
        ids.reserve(contexts_.size());
        for (const auto& pair : contexts_) {
            ids.push_back(pair.first);
        }
        return ids;
    }
};

#endif // CUDA_GRAPHS_HPP