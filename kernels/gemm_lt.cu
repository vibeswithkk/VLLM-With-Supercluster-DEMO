#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <iostream>
#include <memory>
#include <cstring>
#include <algorithm>
#include <cassert>

// Enterprise-grade error handling and logging
#define CUBLAS_SAFE_CALL(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, err); \
        return err; \
    } \
} while(0)

#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return cudaError_t(err); \
    } \
} while(0)

// Constants for performance optimization
constexpr size_t kDefaultWorkspaceSize = 256 * 1024 * 1024; // 256MB
constexpr size_t kMaxWorkspaceSize = 1024 * 1024 * 1024;    // 1GB
constexpr int kMaxAlgorithmHeuristics = 10;

// Enhanced GEMM configuration with improved resource management
struct GemmConfig {
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    size_t workspaceSize;
    void* workspace;
    bool initialized;
    
    // Constructor
    GemmConfig() : ltHandle(nullptr), operationDesc(nullptr), 
                   Adesc(nullptr), Bdesc(nullptr), Cdesc(nullptr), Ddesc(nullptr),
                   preference(nullptr), workspaceSize(0), workspace(nullptr), initialized(false) {}
    
    // Destructor with proper cleanup
    ~GemmConfig() {
        cleanup();
    }
    
    // Cleanup resources
    void cleanup() {
        if (workspace) {
            cudaFree(workspace);
            workspace = nullptr;
        }
        if (preference) {
            cublasLtMatmulPreferenceDestroy(preference);
            preference = nullptr;
        }
        if (Ddesc) {
            cublasLtMatrixLayoutDestroy(Ddesc);
            Ddesc = nullptr;
        }
        if (Cdesc) {
            cublasLtMatrixLayoutDestroy(Cdesc);
            Cdesc = nullptr;
        }
        if (Bdesc) {
            cublasLtMatrixLayoutDestroy(Bdesc);
            Bdesc = nullptr;
        }
        if (Adesc) {
            cublasLtMatrixLayoutDestroy(Adesc);
            Adesc = nullptr;
        }
        if (operationDesc) {
            cublasLtMatmulDescDestroy(operationDesc);
            operationDesc = nullptr;
        }
        if (ltHandle) {
            cublasLtDestroy(ltHandle);
            ltHandle = nullptr;
        }
        initialized = false;
    }
};

// Initialize GEMM configuration with enhanced error handling
cudaError_t init_gemm_config(GemmConfig& config, size_t workspace_size = kDefaultWorkspaceSize) {
    // Cleanup any existing resources
    config.cleanup();
    
    // Validate workspace size
    if (workspace_size == 0) {
        workspace_size = kDefaultWorkspaceSize;
    } else if (workspace_size > kMaxWorkspaceSize) {
        workspace_size = kMaxWorkspaceSize;
    }
    
    // Create cuBLASLt handle
    CUBLAS_SAFE_CALL(cublasLtCreate(&config.ltHandle));
    
    // Create matrix multiplication descriptor
    CUBLAS_SAFE_CALL(cublasLtMatmulDescCreate(&config.operationDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F));
    
    // Set transpose operations (no transpose for all matrices)
    cublasOperation_t opTransA = CUBLAS_OP_N;
    cublasOperation_t opTransB = CUBLAS_OP_N;
    CUBLAS_SAFE_CALL(cublasLtMatmulDescSetAttribute(config.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTransA, sizeof(opTransA)));
    CUBLAS_SAFE_CALL(cublasLtMatmulDescSetAttribute(config.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTransB, sizeof(opTransB)));
    
    // Set Epilogue to support bias addition
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_SAFE_CALL(cublasLtMatmulDescSetAttribute(config.operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    
    // Create matrix layout descriptors
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutCreate(&config.Adesc, CUDA_R_16F, 0, 0, 0)); // Will be set in gemm_execute
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutCreate(&config.Bdesc, CUDA_R_16F, 0, 0, 0)); // Will be set in gemm_execute
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutCreate(&config.Cdesc, CUDA_R_32F, 0, 0, 0)); // Will be set in gemm_execute
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutCreate(&config.Ddesc, CUDA_R_32F, 0, 0, 0)); // Will be set in gemm_execute
    
    // Create preference handle
    CUBLAS_SAFE_CALL(cublasLtMatmulPreferenceCreate(&config.preference));
    
    // Set workspace size
    config.workspaceSize = workspace_size;
    CUBLAS_SAFE_CALL(cublasLtMatmulPreferenceSetAttribute(
        config.preference, 
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &config.workspaceSize, 
        sizeof(config.workspaceSize)
    ));
    
    // Allocate workspace
    CUDA_SAFE_CALL(cudaMalloc(&config.workspace, config.workspaceSize));
    
    config.initialized = true;
    return cudaSuccess;
}

// Execute GEMM operation: D = alpha * A * B + beta * C + bias
cudaError_t gemm_execute(
    GemmConfig& config,
    float alpha,
    const void* A, cudaDataType_t Atype, int64_t m, int64_t k, int64_t lda,
    const void* B, cudaDataType_t Btype, int64_t k_, int64_t n, int64_t ldb,
    float beta,
    const void* C, cudaDataType_t Ctype, int64_t m_, int64_t n_, int64_t ldc,
    void* D, cudaDataType_t Dtype, int64_t m__, int64_t n__, int64_t ldd,
    const void* bias, cudaDataType_t biasType, int64_t biasSize,
    cudaStream_t stream
) {
    // Validate inputs
    if (!config.initialized) {
        return cudaErrorNotInitialized;
    }
    
    if (!A || !B || !C || !D) {
        return cudaErrorInvalidValue;
    }
    
    if (m <= 0 || n <= 0 || k <= 0) {
        return cudaErrorInvalidValue;
    }
    
    if (k != k_ || m != m_ || n != n_ || m != m__ || n != n__) {
        return cudaErrorInvalidValue;
    }
    
    // Update matrix layout descriptors with current dimensions
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_COLS, &k, sizeof(k)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_LD, &lda, sizeof(lda)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Atype, sizeof(Atype)));
    
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &k, sizeof(k)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldb, sizeof(ldb)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Btype, sizeof(Btype)));
    
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldc, sizeof(ldc)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Ctype, sizeof(Ctype)));
    
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldd, sizeof(ldd)));
    CUBLAS_SAFE_CALL(cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Dtype, sizeof(Dtype)));
    
    // Set bias if provided
    if (bias) {
        CUBLAS_SAFE_CALL(cublasLtMatmulDescSetAttribute(config.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        CUBLAS_SAFE_CALL(cublasLtMatmulDescSetAttribute(config.operationDesc, CUBLASLT_MATMUL_DESC_BIAS_TYPE, &biasType, sizeof(biasType)));
    }
    
    // Get algorithm heuristic
    int returnedResults = 0;
    CUBLAS_SAFE_CALL(cublasLtMatmulAlgoGetHeuristic(
        config.ltHandle,
        config.operationDesc,
        config.Adesc,
        config.Bdesc,
        config.Cdesc,
        config.Ddesc,
        config.preference,
        kMaxAlgorithmHeuristics,
        &config.heuristicResult,
        &returnedResults
    ));
    
    if (returnedResults == 0) {
        fprintf(stderr, "No suitable algorithm found for GEMM operation\n");
        return cudaErrorNotSupported;
    }
    
    // Set stream for the operation
    CUBLAS_SAFE_CALL(cublasLtSetStream(config.ltHandle, stream));
    
    // Execute matrix multiplication
    CUBLAS_SAFE_CALL(cublasLtMatmul(
        config.ltHandle,
        config.operationDesc,
        &alpha,
        A,
        config.Adesc,
        B,
        config.Bdesc,
        &beta,
        C,
        config.Cdesc,
        D,
        config.Ddesc,
        &config.heuristicResult.algo,
        config.workspace,
        config.workspaceSize,
        stream
    ));
    
    return cudaSuccess;
}

// Specialized GEMM for common FP16 x FP16 -> FP32 operations
cudaError_t gemm_fp16_fp32(
    GemmConfig& config,
    float alpha,
    const __half* A, int64_t m, int64_t k, int64_t lda,
    const __half* B, int64_t k_, int64_t n, int64_t ldb,
    float beta,
    const float* C, int64_t m_, int64_t n_, int64_t ldc,
    float* D, int64_t m__, int64_t n__, int64_t ldd,
    const float* bias, int64_t biasSize,
    cudaStream_t stream
) {
    return gemm_execute(
        config,
        alpha,
        A, CUDA_R_16F, m, k, lda,
        B, CUDA_R_16F, k_, n, ldb,
        beta,
        C, CUDA_R_32F, m_, n_, ldc,
        D, CUDA_R_32F, m__, n__, ldd,
        bias, CUDA_R_32F, biasSize,
        stream
    );
}

// Batched GEMM operation for processing multiple matrices
cudaError_t gemm_batched(
    GemmConfig& config,
    float alpha,
    const void* A[], cudaDataType_t Atype, int64_t m, int64_t k, int64_t lda,
    const void* B[], cudaDataType_t Btype, int64_t k_, int64_t n, int64_t ldb,
    float beta,
    const void* C[], cudaDataType_t Ctype, int64_t m_, int64_t n_, int64_t ldc,
    void* D[], cudaDataType_t Dtype, int64_t m__, int64_t n__, int64_t ldd,
    int64_t batchCount,
    cudaStream_t stream
) {
    // Validate inputs
    if (!config.initialized) {
        return cudaErrorNotInitialized;
    }
    
    if (!A || !B || !C || !D) {
        return cudaErrorInvalidValue;
    }
    
    if (batchCount <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // For batched operations, we'll use a loop for now
    // In a production system, we'd use cublasLtMatmulStridedBatched
    for (int64_t i = 0; i < batchCount; ++i) {
        cudaError_t err = gemm_execute(
            config,
            alpha,
            A[i], Atype, m, k, lda,
            B[i], Btype, k_, n, ldb,
            beta,
            C[i], Ctype, m_, n_, ldc,
            D[i], Dtype, m__, n__, ldd,
            nullptr, CUDA_R_32F, 0,
            stream
        );
        
        if (err != cudaSuccess) {
            return err;
        }
    }
    
    return cudaSuccess;
}

// Query available algorithms for a given configuration
int query_algorithms(
    GemmConfig& config,
    const void* A, cudaDataType_t Atype, int64_t m, int64_t k, int64_t lda,
    const void* B, cudaDataType_t Btype, int64_t k_, int64_t n, int64_t ldb,
    const void* C, cudaDataType_t Ctype, int64_t m_, int64_t n_, int64_t ldc,
    void* D, cudaDataType_t Dtype, int64_t m__, int64_t n__, int64_t ldd,
    cublasLtMatmulHeuristicResult_t results[],
    int maxResults
) {
    if (!config.initialized || !results || maxResults <= 0) {
        return 0;
    }
    
    // Update matrix layout descriptors
    cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m));
    cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_COLS, &k, sizeof(k));
    cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_LD, &lda, sizeof(lda));
    cublasLtMatrixLayoutSetAttribute(config.Adesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Atype, sizeof(Atype));
    
    cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &k, sizeof(k));
    cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n));
    cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldb, sizeof(ldb));
    cublasLtMatrixLayoutSetAttribute(config.Bdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Btype, sizeof(Btype));
    
    cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m));
    cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n));
    cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldc, sizeof(ldc));
    cublasLtMatrixLayoutSetAttribute(config.Cdesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Ctype, sizeof(Ctype));
    
    cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_ROWS, &m, sizeof(m));
    cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_COLS, &n, sizeof(n));
    cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_LD, &ldd, sizeof(ldd));
    cublasLtMatrixLayoutSetAttribute(config.Ddesc, CUBLASLT_MATRIX_LAYOUT_TYPE, &Dtype, sizeof(Dtype));
    
    // Get algorithm heuristic
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(
        config.ltHandle,
        config.operationDesc,
        config.Adesc,
        config.Bdesc,
        config.Cdesc,
        config.Ddesc,
        config.preference,
        maxResults,
        results,
        &returnedResults
    );
    
    return returnedResults;
}