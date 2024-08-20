#include <iostream>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

// Can't go higher than 8192 rows because of shared memory
// limit of 256KB
// 16384 rows -> 16384 * 4 bytes (unsigned int) = 64KB
// 16384 -> 4 blocks -> 4 * 64 = 262KB (higher than 256KB)
#define ROWS            8192
#define COLUMNS         8192
#define SPARSITY_RATIO  0.2
#define BLOCK_SIZE      1024

template <typename T>
void move_coo_matrix_to_device(COOMatrix<T>& h_matrix, COOMatrix<T>& d_matrix) {
    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.rowIdx, h_matrix.num_nonzero * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.colIdx, h_matrix.num_nonzero * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.value, h_matrix.num_nonzero * sizeof(T)));

    // Set dimensions
    d_matrix.R = h_matrix.R;
    d_matrix.C = h_matrix.C;
    d_matrix.num_nonzero = h_matrix.num_nonzero;

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.rowIdx, h_matrix.rowIdx, h_matrix.num_nonzero * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.colIdx, h_matrix.colIdx, h_matrix.num_nonzero * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.value, h_matrix.value, h_matrix.num_nonzero * sizeof(T), cudaMemcpyHostToDevice));
}

__device__ int get_bin_device(int x, unsigned int BIN_COUNT) {
    return x % BIN_COUNT;
}

template <typename T>
__global__ void histo_kernel(T* data, T N, T* histo, unsigned int bin_count) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ T histo_s[];

    for(int bin=threadIdx.x; bin<bin_count; bin+=blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    for(unsigned int i=tid; i<N; i+=(blockDim.x * gridDim.x)) {
        int idx = get_bin_device(data[i], bin_count);
        atomicAdd(&(histo_s[idx]), 1);
    }

    __syncthreads();
    for(int bin=threadIdx.x; bin<bin_count; bin+=blockDim.x) {
        T binVal = histo_s[bin];
        if(binVal > 0) {
            atomicAdd(&histo[bin], binVal);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

template <typename T>
void compute_histogram(T* data_d, unsigned int N, T* histo_d, unsigned int bin_count) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int multiprocessor_count = deviceProp.multiProcessorCount;
    std::cout<<multiprocessor_count<<std::endl;
    // dim3 gridSize(16 * multiprocessor_count);
    dim3 gridSize(cdiv(N, BLOCK_SIZE));
    dim3 blockSize(BLOCK_SIZE);

    unsigned int shared_mem_size = bin_count*sizeof(T);
    std::cout<<shared_mem_size<<std::endl;
    histo_kernel<T><<<gridSize, blockSize, shared_mem_size>>>(data_d, N, histo_d, bin_count);

    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
__global__ void kogge_stone_segmented_scan_kernel(T* X, T* Y, T* partialSums, unsigned int N) {
    __shared__ T XY_s[BLOCK_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Important for Exclusive scan
    if(i < N and threadIdx.x > 0) {
        // Important for Exclusive scan
        XY_s[threadIdx.x] = X[i-1];
    }    
    else {
        XY_s[threadIdx.x] = 0.0f;
    }

    for(unsigned int stride=1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        T temp;
        if(threadIdx.x >= stride) {
            temp = XY_s[threadIdx.x] + XY_s[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride) {
            XY_s[threadIdx.x] = temp;
        }
    }

    if(threadIdx.x == BLOCK_SIZE-1) {
        // Important for Exclusive scan
        partialSums[blockIdx.x] = XY_s[threadIdx.x] + X[i];
    }
    if(i < N) {
        Y[i] = XY_s[threadIdx.x];
    }
}

template <typename T>
__global__ void kogge_stone_scan_kernel(T* partialSums, unsigned int N) {
    __shared__ T XY_s[BLOCK_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        XY_s[threadIdx.x] = partialSums[i];
    }    
    else {
        XY_s[threadIdx.x] = 0.0f;
    }

    for(unsigned int stride=1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        T temp;
        if(threadIdx.x >= stride) {
            temp = XY_s[threadIdx.x] + XY_s[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride) {
            XY_s[threadIdx.x] = temp;
        }
    }

    if(i < N) {
        partialSums[i] = XY_s[threadIdx.x];
    }
}

template <typename T>
__global__ void redistribute_sum(T* Y, T* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(blockIdx.x > 0)    
        Y[i] += partialSums[blockIdx.x-1];
}

template <typename T>
void compute_scan(T* X_d, T* Y_d, T* partialSums_d, unsigned int N, unsigned int partialSumsLen) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(N, BLOCK_SIZE));

    kogge_stone_segmented_scan_kernel<T><<<gridSize, blockSize>>>(X_d, Y_d, partialSums_d, N);

    blockSize = dim3(min(partialSumsLen, BLOCK_SIZE));
    gridSize = dim3(cdiv(partialSumsLen, blockSize.x));

    kogge_stone_scan_kernel<T><<<gridSize, blockSize>>>(partialSums_d, partialSumsLen);

    blockSize = dim3(BLOCK_SIZE);
    gridSize = dim3(cdiv(N, BLOCK_SIZE));
    redistribute_sum<T><<<gridSize, blockSize>>>(Y_d, partialSums_d, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
void run_engine(float sparsity_ratio, unsigned int R, unsigned int C, float abs_tol, double rel_tol) {
    SparseMatrix<T> sparse_matrix = generate_sparse_matrix<T>(sparsity_ratio, R, C);

    COOMatrix<T> A_coo_h = sparse_to_coo<T>(sparse_matrix);
    CSRMatrix<T> A_csr_h = sparse_to_csr<T>(sparse_matrix);

    unsigned int BIN_COUNT = A_coo_h.R;
    unsigned int *histo_h = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&histo_h, BIN_COUNT*sizeof(unsigned int)));
    std::fill(histo_h, histo_h+BIN_COUNT, 0);

    unsigned int *cumsum_h = nullptr;
    unsigned int *partialSums_h = nullptr;
    unsigned int partialSumsLen = cdiv(A_coo_h.R+1, BLOCK_SIZE);

    CHECK_CUDA_ERROR(cudaMallocHost(&cumsum_h, (A_coo_h.R+1)*sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&partialSums_h, partialSumsLen*sizeof(T)));
    random_initialize_array(cumsum_h, (A_coo_h.R+1), 1337);
    random_initialize_array(partialSums_h, partialSumsLen, 1337);

    COOMatrix<T> A_coo_d;

    move_coo_matrix_to_device(A_coo_h, A_coo_d);
    
    unsigned int *histo_d;

    CHECK_CUDA_ERROR(cudaMalloc(&histo_d, BIN_COUNT*sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemcpy(histo_d, histo_h, BIN_COUNT*sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int *cumsum_d;
    unsigned int *partialSums_d = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&cumsum_d, (A_coo_h.R+1)*sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&partialSums_d, partialSumsLen*sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemcpy(cumsum_d, cumsum_h, (A_coo_h.R+1)*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(partialSums_d, partialSums_h, partialSumsLen*sizeof(unsigned int), cudaMemcpyHostToDevice));

    compute_histogram<unsigned int>(A_coo_d.rowIdx, A_coo_d.num_nonzero, histo_d, BIN_COUNT);
    compute_scan<unsigned int>(histo_d, cumsum_d, partialSums_d, A_coo_d.R+1, partialSumsLen);
    
    CHECK_CUDA_ERROR(cudaMemcpy(partialSums_h, partialSums_d, partialSumsLen*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cumsum_h, cumsum_d, (A_coo_d.R+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // print_array<unsigned int>(cumsum_h, A_csr_h.R+1, "CSR CUDA");
    // print_array<unsigned int>(A_csr_h.rowPtrs, A_csr_h.R+1, "CSR Ref");
    std::cout   << "(Original CSR rowPtrs) vs (COO->CSR rowPtrs) allclose: "
                << (all_close<unsigned int>(A_csr_h.rowPtrs, cumsum_h, A_csr_h.R+1, abs_tol, rel_tol) ? "true" : "false")
                << std::endl; 

    CHECK_CUDA_ERROR(cudaFree(A_coo_d.rowIdx));
    CHECK_CUDA_ERROR(cudaFree(A_coo_d.colIdx));
    CHECK_CUDA_ERROR(cudaFree(A_coo_d.value));
    CHECK_CUDA_ERROR(cudaFree(histo_d));
    CHECK_CUDA_ERROR(cudaFree(cumsum_d));
    CHECK_CUDA_ERROR(cudaFree(partialSums_d));
    free(A_coo_h.rowIdx);
    free(A_coo_h.colIdx);
    free(A_coo_h.value);
    free(A_csr_h.rowPtrs);
    free(A_csr_h.colIdx);
    free(A_csr_h.value);
    CHECK_CUDA_ERROR(cudaFreeHost(histo_h));
    CHECK_CUDA_ERROR(cudaFreeHost(cumsum_h));
    CHECK_CUDA_ERROR(cudaFreeHost(partialSums_h));
}

int main() {
    unsigned int R = ROWS;
    unsigned int C = COLUMNS;
    float sparsity_ratio = SPARSITY_RATIO;

    float abs_tol = 1.0e-8f;
    double rel_tol = 1.0e-8f;

    run_engine<float>(sparsity_ratio, R, C, abs_tol, rel_tol);

    return 0;
}