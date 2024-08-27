#include <iostream>
#include <torch/torch.h>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

#define ROWS            4096
#define COLUMNS         2048
#define SPARSITY_RATIO  0.2
#define BLOCK_SIZE      32
#define DENSE_COLS      1024

template <typename T>
void move_bsr_matrix_to_device(BSRMatrix<T>& h_matrix, BSRMatrix<T>& d_matrix) {
    unsigned int R_b = (h_matrix.R + h_matrix.block_size - 1) /  h_matrix.block_size;
    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.rowPtrs, (R_b+1) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.colIdx, h_matrix.size_colIdx * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.value, h_matrix.size_value * sizeof(T)));

    // Set dimensions
    d_matrix.R = h_matrix.R;
    d_matrix.C = h_matrix.C;
    d_matrix.num_nonzero = h_matrix.num_nonzero;
    d_matrix.size_colIdx = h_matrix.size_colIdx;
    d_matrix.size_value = h_matrix.size_value;
    d_matrix.block_size = h_matrix.block_size;

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.rowPtrs, h_matrix.rowPtrs, (R_b+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.colIdx, h_matrix.colIdx, h_matrix.size_colIdx * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.value, h_matrix.value, h_matrix.size_value * sizeof(T), cudaMemcpyHostToDevice));
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Kernel that processes a row block per thread block
template <typename T>
__global__ void spmm_bsr_dsd_kernel(BSRMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int b = A.block_size;
    unsigned int R_b = (A.R + b - 1) / b;
    unsigned int block_row = blockIdx.x;
    unsigned int thread_col = threadIdx.x;
    unsigned int thread_row = threadIdx.y;

    if(block_row < R_b) {
        for(unsigned int idx = A.rowPtrs[block_row]; idx < A.rowPtrs[block_row + 1]; idx++) {
            unsigned int block_col = A.colIdx[idx];
            T* block = &A.value[idx * b * b];

            for(unsigned int n = 0; n < N; n += blockDim.x) {
                unsigned int col = n + thread_col;
                if(col < N) {
                    T temp_sum = 0.0f;
                    for(unsigned int j = 0; j < b; j++) {
                        unsigned int B_row = block_col * b + j;
                        if(B_row < A.C) {
                            temp_sum += block[thread_row * b + j] * B[B_row * N + col];
                        }
                    }
                    unsigned int C_row = block_row * b + thread_row;
                    if(C_row < A.R) {
                        atomicAdd(&C[C_row * N + col], temp_sum);
                    }
                }
            }
        }
    }
}

template <typename T>
void spmm_bsr_dsd(BSRMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int b = A.block_size;
    unsigned int R_b = (A.R + b - 1) / b;
    dim3 blockSize(32, b);
    dim3 gridSize(R_b);

    spmm_bsr_dsd_kernel<T><<<gridSize, blockSize>>>(A, B, C, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

// Kernel that assigns one thread block per BSR block
template <typename T>
__global__ void spmm_bsr_dsd_block_kernel(BSRMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int b = A.block_size;
    unsigned int block_idx = blockIdx.x;
    unsigned int thread_row = threadIdx.y;
    unsigned int thread_col = threadIdx.x;

    unsigned int block_row = 0;
    unsigned int left = 0, right = (A.R + b - 1) / b;

    while(left < right) {
        unsigned int mid = (left + right) / 2;
        if(block_idx < A.rowPtrs[mid]) {
            right = mid;
        }
        else if(block_idx >= A.rowPtrs[mid + 1]) {
            left = mid + 1;
        }
        else {
            block_row = mid;
            break;
        }
    }

    if(left == right) {
        block_row = left;
    }

    unsigned int block_col = A.colIdx[block_idx];
    const T* block = &A.value[block_idx * b * b];

    for(unsigned int n = 0; n < N; n += blockDim.x) {
        unsigned int col = n + thread_col;
        if(col < N) {
            T temp_sum = 0.0f;
            for(unsigned int j = 0; j < b; j++) {
                unsigned int B_row = block_col * b + j;
                if(B_row < A.C) {
                    temp_sum += block[thread_row * b + j] * B[B_row * N + col];
                }
            }
            unsigned int C_row = block_row * b + thread_row;
            if(C_row < A.R) {
                atomicAdd(&C[C_row * N + col], temp_sum);
            }
        }
    }
}

template <typename T>
void spmm_bsr_dsd_block(BSRMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int b = A.block_size;
    unsigned int num_blocks = A.size_value / (b * b);
    dim3 blockSize(32, b);
    dim3 gridSize(num_blocks);

    spmm_bsr_dsd_block_kernel<T><<<gridSize, blockSize>>>(A, B, C, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

// Kernel that assigns one thread block per BSR block with shared memory
template <typename T>
__global__ void spmm_bsr_dsd_block_shared_kernel(BSRMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    extern __shared__ T shared_B[];

    unsigned int b = A.block_size;
    unsigned int block_idx = blockIdx.x;
    unsigned int thread_row = threadIdx.y;
    unsigned int thread_col = threadIdx.x;

    unsigned int block_row = 0;
    unsigned int left = 0, right = (A.R + b - 1) / b;

    // Linear search to find the block_row
    // while(block_idx >= A.rowPtrs[block_row + 1]) {
    //     block_row++;
    // }
    
    // Binary search to find the block_row
    while(left < right) {
        unsigned int mid = (left + right) / 2;
        if(block_idx < A.rowPtrs[mid]) {
            right = mid;
        }
        else if(block_idx >= A.rowPtrs[mid + 1]) {
            left = mid + 1;
        }
        else {
            block_row = mid;
            break;
        }
    }

    if(left == right) {
        block_row = left;
    }

    unsigned int block_col = A.colIdx[block_idx];
    const T* block = &A.value[block_idx * b * b];

    for(unsigned int n = 0; n < N; n += blockDim.x) {
        unsigned int col = n + thread_col;

        // Load B into shared memory
        if(col < N and thread_row < b) {
            unsigned int B_row = block_col * b + thread_row;
            if(B_row < A.C) {
                shared_B[thread_row * blockDim.x + thread_col] = B[B_row * N + col];
            } 
            else {
                shared_B[thread_row * blockDim.x + thread_col] = 0;
            }
        }

        __syncthreads();

        if(col < N) {
            T temp_sum = 0.0f;
            for(unsigned int j = 0; j < b; j++) {
                temp_sum += block[thread_row * b + j] * shared_B[j * blockDim.x + thread_col];
            }
            unsigned int C_row = block_row * b + thread_row;
            if(C_row < A.R) {
                atomicAdd(&C[C_row * N + col], temp_sum);
            }
        }
        
        __syncthreads();
    }
}

template <typename T>
void spmm_bsr_dsd_block_shared(BSRMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> C x N
    // C -> dense matrix -> C = A @ B -> R x N
    unsigned int b = A.block_size;
    unsigned int num_blocks = A.size_value / (b * b);
    dim3 blockSize(32, b);
    dim3 gridSize(num_blocks);

    size_t sharedMemSize = blockSize.x * b * sizeof(T);

    spmm_bsr_dsd_block_shared_kernel<T><<<gridSize, blockSize, sharedMemSize>>>(A, B, C, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
T compute_torch_mm(T A, T B) {
    T ans = torch::matmul(A, B);
    return ans;
}

template <typename T>
void run_engine(float sparsity_ratio, unsigned int R, unsigned int C, unsigned int N, float abs_tol, double rel_tol) {
    SparseMatrix<T> sparse_matrix = generate_sparse_matrix<T>(sparsity_ratio, R, C);
    
    unsigned int bsr_block_size = 32;
    BSRMatrix<T> A_h = sparse_to_bsr<T>(sparse_matrix, bsr_block_size);

    T* B_h = nullptr;
    T* C_h = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&B_h, C * N * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_h, R * N * sizeof(T)));

    random_initialize_array(B_h, C * N, 1337);
    std::fill(C_h, C_h + R * N, static_cast<T>(0));

    BSRMatrix<T> A_d;
    T *B_d, *C_d;

    move_bsr_matrix_to_device(A_h, A_d);

    CHECK_CUDA_ERROR(cudaMalloc(&B_d, C * N * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_d, R * N * sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, C * N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_d, C_h, R * N * sizeof(T), cudaMemcpyHostToDevice));

    spmm_bsr_dsd<T>(A_d, B_d, C_d, N);
    // spmm_bsr_dsd_block<T>(A_d, B_d, C_d, N);
    // spmm_bsr_dsd_block_shared<T>(A_d, B_d, C_d, N);

    CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, R * N * sizeof(T), cudaMemcpyDeviceToHost));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor A_t = torch::from_blob(sparse_matrix.mat, {R, C}, options).clone().cuda();
    torch::Tensor B_t = torch::from_blob(B_h, {C, N}, options).clone().cuda();
    torch::Tensor C_cuda = torch::from_blob(C_h, {R, N}, options).clone();

    torch::Tensor C_t = compute_torch_mm<torch::Tensor>(A_t, B_t).cpu();

    std::cout << "CUDA vs Torch allclose: "
              << (torch::allclose(C_cuda, C_t, abs_tol, rel_tol) ? "true" : "false")
              << std::endl;

    CHECK_CUDA_ERROR(cudaFree(A_d.rowPtrs));
    CHECK_CUDA_ERROR(cudaFree(A_d.colIdx));
    CHECK_CUDA_ERROR(cudaFree(A_d.value));
    CHECK_CUDA_ERROR(cudaFree(B_d));
    CHECK_CUDA_ERROR(cudaFree(C_d));
    free(A_h.rowPtrs);
    free(A_h.colIdx);
    free(A_h.value);
    CHECK_CUDA_ERROR(cudaFreeHost(B_h));
    CHECK_CUDA_ERROR(cudaFreeHost(C_h));
}

int main() {
    unsigned int R = ROWS;
    unsigned int C = COLUMNS;
    unsigned int N = DENSE_COLS;
    float sparsity_ratio = SPARSITY_RATIO;

    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    run_engine<float>(sparsity_ratio, R, C, N, abs_tol, rel_tol);

    return 0;
}
