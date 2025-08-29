#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

// Include engine components
#include "../engine/tensor.hpp"
#include "../engine/allocator.hpp"
#include "../engine/cuda_graphs.hpp"

// Include kernel components
#include "../kernels/layernorm.hpp"
#include "../kernels/gemm_lt.hpp"
#include "../kernels/paged_attention.hpp"

/**
 * @brief Enterprise-grade latency benchmark for inference operations
 * 
 * This benchmark measures the performance of key inference operations
 * with comprehensive statistics, error handling, and enterprise-grade
 * reporting capabilities.
 */
class LatencyBenchmark {
private:
    // Allocator
    std::unique_ptr<GPUAllocator> allocator_;
    
    // CUDA graphs for optimized execution
    std::unique_ptr<CudaGraphManager> graph_manager_;
    
    // Random number generator for test data
    std::mt19937 rng_;
    
    // Benchmark configuration
    size_t num_warmup_iterations_;
    size_t num_benchmark_iterations_;
    
public:
    /**
     * @brief Constructor
     * @param num_warmup_iterations Number of warmup iterations
     * @param num_benchmark_iterations Number of benchmark iterations
     */
    LatencyBenchmark(size_t num_warmup_iterations = 10, 
                    size_t num_benchmark_iterations = 100)
        : num_warmup_iterations_(num_warmup_iterations),
          num_benchmark_iterations_(num_benchmark_iterations),
          rng_(std::random_device{}()) {
        
        // Initialize allocator
        allocator_ = std::make_unique<GPUAllocator>();
        allocator_->initialize(2ULL * 1024 * 1024 * 1024,  // 2GB initial pool
                              16ULL * 1024 * 1024 * 1024, // 16GB max pool
                              true);  // Enable defragmentation
        
        // Initialize CUDA graphs
        graph_manager_ = std::make_unique<CudaGraphManager>();
    }
    
    /**
     * @brief Statistics structure for benchmark results
     */
    struct BenchmarkStats {
        double mean_latency_ms;
        double median_latency_ms;
        double min_latency_ms;
        double max_latency_ms;
        double std_deviation_ms;
        double p90_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        size_t throughput_ops_per_sec;
        size_t memory_usage_bytes;
        
        void print() const {
            std::cout << "  Mean Latency: " << mean_latency_ms << " ms" << std::endl;
            std::cout << "  Median Latency: " << median_latency_ms << " ms" << std::endl;
            std::cout << "  Min Latency: " << min_latency_ms << " ms" << std::endl;
            std::cout << "  Max Latency: " << max_latency_ms << " ms" << std::endl;
            std::cout << "  Std Deviation: " << std_deviation_ms << " ms" << std::endl;
            std::cout << "  90th Percentile: " << p90_latency_ms << " ms" << std::endl;
            std::cout << "  95th Percentile: " << p95_latency_ms << " ms" << std::endl;
            std::cout << "  99th Percentile: " << p99_latency_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << throughput_ops_per_sec << " ops/sec" << std::endl;
            std::cout << "  Memory Usage: " << memory_usage_bytes << " bytes" << std::endl;
        }
    };
    
    /**
     * @brief Run benchmark for layer normalization
     * @param batch_size Batch size
     * @param seq_length Sequence length
     * @param hidden_size Hidden size
     * @return Benchmark statistics
     */
    BenchmarkStats benchmarkLayerNorm(size_t batch_size, 
                                    size_t seq_length, 
                                    size_t hidden_size) {
        std::cout << "Benchmarking LayerNorm: [" << batch_size << ", " << seq_length 
                  << ", " << hidden_size << "]" << std::endl;
        
        // Create tensors
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size * seq_length), 
                                           static_cast<int64_t>(hidden_size)};
        auto input = std::make_shared<Tensor>(input_shape, Tensor::DataType::FLOAT32);
        auto weight = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(hidden_size)}, 
                                              Tensor::DataType::FLOAT32);
        auto bias = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(hidden_size)}, 
                                            Tensor::DataType::FLOAT32);
        
        // Initialize with random data
        input->fill_random();
        weight->fill_random();
        bias->fill_random();
        
        // Warmup iterations
        for (size_t i = 0; i < num_warmup_iterations_; ++i) {
            auto result = layerNorm(input, weight, bias);
            // Synchronize to ensure completion
            cudaDeviceSynchronize();
        }
        
        // Benchmark iterations
        std::vector<double> latencies;
        latencies.reserve(num_benchmark_iterations_);
        
        for (size_t i = 0; i < num_benchmark_iterations_; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = layerNorm(input, weight, bias);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            latencies.push_back(duration.count() / 1000.0); // Convert to milliseconds
        }
        
        return computeStats(latencies);
    }
    
    /**
     * @brief Run benchmark for GEMM operation
     * @param m M dimension
     * @param n N dimension
     * @param k K dimension
     * @return Benchmark statistics
     */
    BenchmarkStats benchmarkGEMM(size_t m, size_t n, size_t k) {
        std::cout << "Benchmarking GEMM: [" << m << ", " << n << ", " << k << "]" << std::endl;
        
        // Create tensors
        auto a = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(m), 
                                                              static_cast<int64_t>(k)}, 
                                         Tensor::DataType::FLOAT32);
        auto b = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(k), 
                                                              static_cast<int64_t>(n)}, 
                                         Tensor::DataType::FLOAT32);
        auto bias = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(n)}, 
                                            Tensor::DataType::FLOAT32);
        
        // Initialize with random data
        a->fill_random();
        b->fill_random();
        bias->fill_random();
        
        // Warmup iterations
        for (size_t i = 0; i < num_warmup_iterations_; ++i) {
            auto result = gemm(a, b, bias);
            // Synchronize to ensure completion
            cudaDeviceSynchronize();
        }
        
        // Benchmark iterations
        std::vector<double> latencies;
        latencies.reserve(num_benchmark_iterations_);
        
        for (size_t i = 0; i < num_benchmark_iterations_; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = gemm(a, b, bias);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            latencies.push_back(duration.count() / 1000.0); // Convert to milliseconds
        }
        
        return computeStats(latencies);
    }
    
    /**
     * @brief Run benchmark for paged attention
     * @param batch_size Batch size
     * @param num_heads Number of attention heads
     * @param head_size Head size
     * @param seq_length Sequence length
     * @return Benchmark statistics
     */
    BenchmarkStats benchmarkPagedAttention(size_t batch_size,
                                         size_t num_heads,
                                         size_t head_size,
                                         size_t seq_length) {
        std::cout << "Benchmarking PagedAttention: [" << batch_size << ", " << num_heads 
                  << ", " << head_size << ", " << seq_length << "]" << std::endl;
        
        // Create tensors
        auto query = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(batch_size), 
                                                                  static_cast<int64_t>(num_heads), 
                                                                  static_cast<int64_t>(head_size)}, 
                                             Tensor::DataType::FLOAT32);
        
        // For paged attention, we need key and value caches
        size_t block_size = 16; // Typical block size
        size_t num_blocks = (seq_length + block_size - 1) / block_size;
        
        auto key_cache = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(num_blocks), 
                                                                      static_cast<int64_t>(num_heads), 
                                                                      static_cast<int64_t>(head_size), 
                                                                      static_cast<int64_t>(block_size)}, 
                                                 Tensor::DataType::FLOAT32);
        
        auto value_cache = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(num_blocks), 
                                                                        static_cast<int64_t>(num_heads), 
                                                                        static_cast<int64_t>(head_size), 
                                                                        static_cast<int64_t>(block_size)}, 
                                                   Tensor::DataType::FLOAT32);
        
        // Block tables and context lengths
        auto block_tables = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(batch_size), 
                                                                         static_cast<int64_t>(num_blocks)}, 
                                                    Tensor::DataType::INT32);
        
        auto context_lens = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(batch_size)}, 
                                                    Tensor::DataType::INT32);
        
        // Initialize with random data
        query->fill_random();
        key_cache->fill_random();
        value_cache->fill_random();
        block_tables->fill_random();
        context_lens->fill_random();
        
        // Warmup iterations
        for (size_t i = 0; i < num_warmup_iterations_; ++i) {
            auto result = pagedAttention(query, key_cache, value_cache, block_tables, context_lens, 1.0f);
            // Synchronize to ensure completion
            cudaDeviceSynchronize();
        }
        
        // Benchmark iterations
        std::vector<double> latencies;
        latencies.reserve(num_benchmark_iterations_);
        
        for (size_t i = 0; i < num_benchmark_iterations_; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = pagedAttention(query, key_cache, value_cache, block_tables, context_lens, 1.0f);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            latencies.push_back(duration.count() / 1000.0); // Convert to milliseconds
        }
        
        return computeStats(latencies);
    }
    
private:
    /**
     * @brief Compute statistics from latency measurements
     * @param latencies Vector of latency measurements in milliseconds
     * @return Benchmark statistics
     */
    BenchmarkStats computeStats(const std::vector<double>& latencies) {
        if (latencies.empty()) {
            return BenchmarkStats{};
        }
        
        // Sort latencies for percentile calculations
        std::vector<double> sorted_latencies = latencies;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());
        
        // Calculate mean
        double sum = 0.0;
        for (double latency : sorted_latencies) {
            sum += latency;
        }
        double mean = sum / sorted_latencies.size();
        
        // Calculate standard deviation
        double sum_sq_diff = 0.0;
        for (double latency : sorted_latencies) {
            double diff = latency - mean;
            sum_sq_diff += diff * diff;
        }
        double std_dev = std::sqrt(sum_sq_diff / sorted_latencies.size());
        
        // Calculate percentiles
        size_t size = sorted_latencies.size();
        double p90 = sorted_latencies[static_cast<size_t>(0.90 * (size - 1))];
        double p95 = sorted_latencies[static_cast<size_t>(0.95 * (size - 1))];
        double p99 = sorted_latencies[static_cast<size_t>(0.99 * (size - 1))];
        
        // Calculate throughput (operations per second)
        double avg_latency_sec = mean / 1000.0; // Convert to seconds
        size_t throughput = static_cast<size_t>(1.0 / avg_latency_sec);
        
        BenchmarkStats stats;
        stats.mean_latency_ms = mean;
        stats.median_latency_ms = sorted_latencies[size / 2];
        stats.min_latency_ms = sorted_latencies.front();
        stats.max_latency_ms = sorted_latencies.back();
        stats.std_deviation_ms = std_dev;
        stats.p90_latency_ms = p90;
        stats.p95_latency_ms = p95;
        stats.p99_latency_ms = p99;
        stats.throughput_ops_per_sec = throughput;
        stats.memory_usage_bytes = allocator_->get_current_usage();
        
        return stats;
    }
};

/**
 * @brief Main function for latency benchmark example
 */
int main(int argc, char* argv[]) {
    try {
        std::cout << "Enterprise-Grade Latency Benchmark for Inference Operations" << std::endl;
        std::cout << "=========================================================" << std::endl;
        
        // Create benchmark instance
        LatencyBenchmark benchmark(5, 50); // 5 warmup, 50 benchmark iterations
        
        // Run benchmarks for different operations
        std::cout << "\n1. Layer Normalization Benchmarks:" << std::endl;
        std::cout << "----------------------------------" << std::endl;
        
        // Small model
        auto stats1 = benchmark.benchmarkLayerNorm(1, 512, 768);
        std::cout << "Small Model (1x512x768):" << std::endl;
        stats1.print();
        
        // Medium model
        auto stats2 = benchmark.benchmarkLayerNorm(8, 512, 1024);
        std::cout << "\nMedium Model (8x512x1024):" << std::endl;
        stats2.print();
        
        // Large model
        auto stats3 = benchmark.benchmarkLayerNorm(32, 1024, 4096);
        std::cout << "\nLarge Model (32x1024x4096):" << std::endl;
        stats3.print();
        
        std::cout << "\n2. GEMM Operation Benchmarks:" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        
        // Small GEMM
        auto stats4 = benchmark.benchmarkGEMM(512, 768, 768);
        std::cout << "Small GEMM (512x768x768):" << std::endl;
        stats4.print();
        
        // Medium GEMM
        auto stats5 = benchmark.benchmarkGEMM(1024, 1024, 4096);
        std::cout << "\nMedium GEMM (1024x1024x4096):" << std::endl;
        stats5.print();
        
        // Large GEMM
        auto stats6 = benchmark.benchmarkGEMM(4096, 4096, 4096);
        std::cout << "\nLarge GEMM (4096x4096x4096):" << std::endl;
        stats6.print();
        
        std::cout << "\n3. Paged Attention Benchmarks:" << std::endl;
        std::cout << "------------------------------" << std::endl;
        
        // Small attention
        auto stats7 = benchmark.benchmarkPagedAttention(1, 12, 64, 512);
        std::cout << "Small Attention (1x12x64x512):" << std::endl;
        stats7.print();
        
        // Medium attention
        auto stats8 = benchmark.benchmarkPagedAttention(8, 16, 64, 1024);
        std::cout << "\nMedium Attention (8x16x64x1024):" << std::endl;
        stats8.print();
        
        // Large attention
        auto stats9 = benchmark.benchmarkPagedAttention(32, 32, 128, 2048);
        std::cout << "\nLarge Attention (32x32x128x2048):" << std::endl;
        stats9.print();
        
        std::cout << "\nLatency Benchmark completed successfully!" << std::endl;
        std::cout << "Total Memory Usage: " << 
            (stats1.memory_usage_bytes + stats2.memory_usage_bytes + stats3.memory_usage_bytes +
             stats4.memory_usage_bytes + stats5.memory_usage_bytes + stats6.memory_usage_bytes +
             stats7.memory_usage_bytes + stats8.memory_usage_bytes + stats9.memory_usage_bytes) / 9 
            << " bytes (average)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}