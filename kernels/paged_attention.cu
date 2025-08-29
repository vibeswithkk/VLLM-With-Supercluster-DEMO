#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <algorithm>

// Enterprise-grade error handling macros
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// Constants for attention computation
constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kMaxBlocksPerGrid = 65535;
constexpr float kNegInf = -1e20f;

// Paged attention kernel configuration structure
struct PagedAttentionConfig {
    int num_heads;
    int head_size;
    int num_seqs;
    int max_seq_len;
    int max_num_blocks_per_seq;
    int block_size;
    float scale;
    
    // Validation
    bool is_valid() const {
        return num_heads > 0 && head_size > 0 && num_seqs > 0 && 
               max_seq_len > 0 && max_num_blocks_per_seq > 0 && block_size > 0;
    }
};

// Optimized paged attention kernel for efficient attention computation
__global__ void paged_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale,
    float* __restrict__ out,
    const int num_heads,
    const int head_size,
    const int num_seqs,
    const int max_seq_len,
    const int max_num_blocks_per_seq,
    const int block_size
) {
    // Calculate thread indices
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    // Bounds checking
    if (head_idx >= num_heads || seq_idx >= num_seqs) return;
    
    // Get context length for this sequence
    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;
    
    // Calculate query offset
    const int query_offset = seq_idx * num_heads * head_size + head_idx * head_size;
    
    // Shared memory for attention scores and reduction
    extern __shared__ float shared_mem[];
    float* shared_scores = shared_mem;
    float* shared_values = shared_mem + max_seq_len;
    
    // Initialize shared memory
    for (int i = tid; i < max_seq_len; i += blockDim.x) {
        shared_scores[i] = (i < context_len) ? 0.0f : kNegInf;
    }
    __syncthreads();
    
    // Compute attention scores for each key token
    for (int block_idx = 0; block_idx < max_num_blocks_per_seq; block_idx++) {
        // Get the physical block number
        const int physical_block_number = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        
        // Process all tokens in this block
        for (int i = 0; i < block_size; i++) {
            const int token_idx = block_idx * block_size + i;
            
            // Skip if beyond context length
            if (token_idx >= context_len) break;
            
            // Calculate key offset
            const int key_offset = physical_block_number * block_size * num_heads * head_size + 
                                 i * num_heads * head_size + 
                                 head_idx * head_size;
            
            // Compute dot product between query and key using warp-level reduction
            float sum = 0.0f;
            for (int j = tid; j < head_size; j += blockDim.x) {
                sum += query[query_offset + j] * key_cache[key_offset + j];
            }
            
            // Warp-level reduction
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            
            // Store result in shared memory
            if (tid == 0) {
                shared_scores[token_idx] = sum * scale;
            }
            __syncthreads();
        }
    }
    
    // Apply softmax to attention scores with numerical stability
    if (tid == 0) {
        // Find max value for numerical stability
        float max_val = shared_scores[0];
        for (int i = 1; i < context_len; i++) {
            max_val = fmaxf(max_val, shared_scores[i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < context_len; i++) {
            shared_scores[i] = expf(shared_scores[i] - max_val);
            sum += shared_scores[i];
        }
        
        // Normalize
        const float inv_sum = 1.0f / sum;
        for (int i = 0; i < context_len; i++) {
            shared_scores[i] *= inv_sum;
        }
    }
    __syncthreads();
    
    // Compute weighted sum of values
    const int out_offset = seq_idx * num_heads * head_size + head_idx * head_size;
    
    // Initialize output to zero
    for (int i = tid; i < head_size; i += blockDim.x) {
        out[out_offset + i] = 0.0f;
    }
    __syncthreads();
    
    // Accumulate weighted values
    for (int block_idx = 0; block_idx < max_num_blocks_per_seq; block_idx++) {
        const int physical_block_number = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        
        for (int i = 0; i < block_size; i++) {
            const int token_idx = block_idx * block_size + i;
            
            if (token_idx >= context_len) break;
            
            // Get attention weight
            const float attn_weight = shared_scores[token_idx];
            
            // Calculate value offset
            const int value_offset = physical_block_number * block_size * num_heads * head_size + 
                                   i * num_heads * head_size + 
                                   head_idx * head_size;
            
            // Accumulate weighted values
            for (int j = tid; j < head_size; j += blockDim.x) {
                out[out_offset + j] += attn_weight * value_cache[value_offset + j];
            }
        }
    }
}

// Optimized paged attention kernel for half precision
__global__ void paged_attention_kernel_half(
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const float scale,
    __half* __restrict__ out,
    const int num_heads,
    const int head_size,
    const int num_seqs,
    const int max_seq_len,
    const int max_num_blocks_per_seq,
    const int block_size
) {
    // Calculate thread indices
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    // Bounds checking
    if (head_idx >= num_heads || seq_idx >= num_seqs) return;
    
    // Get context length for this sequence
    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;
    
    // Calculate query offset
    const int query_offset = seq_idx * num_heads * head_size + head_idx * head_size;
    
    // Shared memory for attention scores and reduction
    extern __shared__ float shared_mem[];
    float* shared_scores = shared_mem;
    float* shared_values = shared_mem + max_seq_len;
    
    // Initialize shared memory
    for (int i = tid; i < max_seq_len; i += blockDim.x) {
        shared_scores[i] = (i < context_len) ? 0.0f : kNegInf;
    }
    __syncthreads();
    
    // Compute attention scores for each key token
    for (int block_idx = 0; block_idx < max_num_blocks_per_seq; block_idx++) {
        // Get the physical block number
        const int physical_block_number = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        
        // Process all tokens in this block
        for (int i = 0; i < block_size; i++) {
            const int token_idx = block_idx * block_size + i;
            
            // Skip if beyond context length
            if (token_idx >= context_len) break;
            
            // Calculate key offset
            const int key_offset = physical_block_number * block_size * num_heads * head_size + 
                                 i * num_heads * head_size + 
                                 head_idx * head_size;
            
            // Compute dot product between query and key using warp-level reduction
            float sum = 0.0f;
            for (int j = tid; j < head_size; j += blockDim.x) {
                sum += __half2float(query[query_offset + j]) * __half2float(key_cache[key_offset + j]);
            }
            
            // Warp-level reduction
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            
            // Store result in shared memory
            if (tid == 0) {
                shared_scores[token_idx] = sum * scale;
            }
            __syncthreads();
        }
    }
    
    // Apply softmax to attention scores with numerical stability
    if (tid == 0) {
        // Find max value for numerical stability
        float max_val = shared_scores[0];
        for (int i = 1; i < context_len; i++) {
            max_val = fmaxf(max_val, shared_scores[i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < context_len; i++) {
            shared_scores[i] = expf(shared_scores[i] - max_val);
            sum += shared_scores[i];
        }
        
        // Normalize
        const float inv_sum = 1.0f / sum;
        for (int i = 0; i < context_len; i++) {
            shared_scores[i] *= inv_sum;
        }
    }
    __syncthreads();
    
    // Compute weighted sum of values
    const int out_offset = seq_idx * num_heads * head_size + head_idx * head_size;
    
    // Initialize output to zero
    for (int i = tid; i < head_size; i += blockDim.x) {
        out[out_offset + i] = __float2half(0.0f);
    }
    __syncthreads();
    
    // Accumulate weighted values
    for (int block_idx = 0; block_idx < max_num_blocks_per_seq; block_idx++) {
        const int physical_block_number = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];
        
        for (int i = 0; i < block_size; i++) {
            const int token_idx = block_idx * block_size + i;
            
            if (token_idx >= context_len) break;
            
            // Get attention weight
            const float attn_weight = shared_scores[token_idx];
            
            // Calculate value offset
            const int value_offset = physical_block_number * block_size * num_heads * head_size + 
                                   i * num_heads * head_size + 
                                   head_idx * head_size;
            
            // Accumulate weighted values
            for (int j = tid; j < head_size; j += blockDim.x) {
                const float current = __half2float(out[out_offset + j]);
                const float value = __half2float(value_cache[value_offset + j]);
                out[out_offset + j] = __float2half(current + attn_weight * value);
            }
        }
    }
}

// Host function to launch paged attention kernel with enhanced error checking
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
) {
    // Input validation
    if (!query || !key_cache || !value_cache || !block_tables || 
        !context_lens || !out) {
        return cudaErrorInvalidValue;
    }
    
    if (num_heads <= 0 || head_size <= 0 || num_seqs <= 0 || 
        max_seq_len <= 0 || max_num_blocks_per_seq <= 0 || block_size <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Validate shared memory requirements
    const size_t shared_mem_size = 2 * max_seq_len * sizeof(float);
    if (shared_mem_size > 48 * 1024) { // 48KB shared memory limit
        return cudaErrorInvalidValue;
    }
    
    // Define grid and block dimensions
    const dim3 block_dim(min(kMaxThreadsPerBlock, max(kWarpSize, head_size)));
    const dim3 grid_dim(num_heads, num_seqs);
    
    // Launch kernel
    paged_attention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        query, key_cache, value_cache, block_tables, context_lens, scale,
        out, num_heads, head_size, num_seqs, max_seq_len, 
        max_num_blocks_per_seq, block_size
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    return cudaSuccess;
}

// Host function to launch half-precision paged attention kernel
cudaError_t launch_paged_attention_half(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    const int* block_tables,
    const int* context_lens,
    const float scale,
    __half* out,
    const int num_heads,
    const int head_size,
    const int num_seqs,
    const int max_seq_len,
    const int max_num_blocks_per_seq,
    const int block_size,
    cudaStream_t stream
) {
    // Input validation
    if (!query || !key_cache || !value_cache || !block_tables || 
        !context_lens || !out) {
        return cudaErrorInvalidValue;
    }
    
    if (num_heads <= 0 || head_size <= 0 || num_seqs <= 0 || 
        max_seq_len <= 0 || max_num_blocks_per_seq <= 0 || block_size <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Validate shared memory requirements
    const size_t shared_mem_size = 2 * max_seq_len * sizeof(float);
    if (shared_mem_size > 48 * 1024) { // 48KB shared memory limit
        return cudaErrorInvalidValue;
    }
    
    // Define grid and block dimensions
    const dim3 block_dim(min(kMaxThreadsPerBlock, max(kWarpSize, head_size)));
    const dim3 grid_dim(num_heads, num_seqs);
    
    // Launch kernel
    paged_attention_kernel_half<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        query, key_cache, value_cache, block_tables, context_lens, scale,
        out, num_heads, head_size, num_seqs, max_seq_len, 
        max_num_blocks_per_seq, block_size
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    return cudaSuccess;
}

// Batched paged attention for processing multiple sequences
cudaError_t launch_paged_attention_batched(
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
    const int batch_size,
    cudaStream_t stream
) {
    // Input validation
    if (!query || !key_cache || !value_cache || !block_tables || 
        !context_lens || !out || batch_size <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Process each batch
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Calculate offsets for this batch
        const int query_offset = batch_idx * num_seqs * num_heads * head_size;
        const int out_offset = batch_idx * num_seqs * num_heads * head_size;
        const int block_tables_offset = batch_idx * num_seqs * max_num_blocks_per_seq;
        const int context_lens_offset = batch_idx * num_seqs;
        
        // Launch kernel for this batch
        cudaError_t err = launch_paged_attention(
            query + query_offset,
            key_cache,
            value_cache,
            block_tables + block_tables_offset,
            context_lens + context_lens_offset,
            scale,
            out + out_offset,
            num_heads,
            head_size,
            num_seqs,
            max_seq_len,
            max_num_blocks_per_seq,
            block_size,
            stream
        );
        
        if (err != cudaSuccess) {
            return err;
        }
    }
    
    return cudaSuccess;
}

// Configurable paged attention with runtime parameters
cudaError_t launch_paged_attention_configurable(
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int* block_tables,
    const int* context_lens,
    const float scale,
    void* out,
    const PagedAttentionConfig& config,
    cudaDataType_t data_type,
    cudaStream_t stream
) {
    // Validate configuration
    if (!config.is_valid()) {
        return cudaErrorInvalidValue;
    }
    
    // Validate inputs
    if (!query || !key_cache || !value_cache || !block_tables || 
        !context_lens || !out) {
        return cudaErrorInvalidValue;
    }
    
    // Validate shared memory requirements
    const size_t shared_mem_size = 2 * config.max_seq_len * sizeof(float);
    if (shared_mem_size > 48 * 1024) { // 48KB shared memory limit
        return cudaErrorInvalidValue;
    }
    
    // Define grid and block dimensions
    const dim3 block_dim(min(kMaxThreadsPerBlock, max(kWarpSize, config.head_size)));
    const dim3 grid_dim(config.num_heads, config.num_seqs);
    
    // Launch appropriate kernel based on data type
    if (data_type == CUDA_R_32F) {
        paged_attention_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
            static_cast<const float*>(query),
            static_cast<const float*>(key_cache),
            static_cast<const float*>(value_cache),
            block_tables, context_lens, scale,
            static_cast<float*>(out),
            config.num_heads, config.head_size, config.num_seqs,
            config.max_seq_len, config.max_num_blocks_per_seq, config.block_size
        );
    } else if (data_type == CUDA_R_16F) {
        paged_attention_kernel_half<<<grid_dim, block_dim, shared_mem_size, stream>>>(
            static_cast<const __half*>(query),
            static_cast<const __half*>(key_cache),
            static_cast<const __half*>(value_cache),
            block_tables, context_lens, scale,
            static_cast<__half*>(out),
            config.num_heads, config.head_size, config.num_seqs,
            config.max_seq_len, config.max_num_blocks_per_seq, config.block_size
        );
    } else {
        return cudaErrorNotSupported;
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    return cudaSuccess;
}