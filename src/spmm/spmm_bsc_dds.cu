#include <iostream>
#include <torch/torch.h>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

#define ROWS            4096
#define COLUMNS         2048
#define SPARSITY_RATIO  0.2
#define BLOCK_SIZE      32
#define DENSE_ROWS      1024

template <typename T>
void move_bsc_matrix_to_device(BSCMatrix<T>& h_matrix, BSCMatrix<T>& d_matrix) {
    unsigned int C_b = (h_matrix.C + h_matrix.block_size - 1) /  h_matrix.block_size;
    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.colPtrs, (C_b+1) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.rowIdx, h_matrix.size_rowIdx * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.value, h_matrix.size_value * sizeof(T)));

    // Set dimensions
    d_matrix.R = h_matrix.R;
    d_matrix.C = h_matrix.C;
    d_matrix.num_nonzero = h_matrix.num_nonzero;
    d_matrix.size_rowIdx = h_matrix.size_rowIdx;
    d_matrix.size_value = h_matrix.size_value;
    d_matrix.block_size = h_matrix.block_size;

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.colPtrs, h_matrix.colPtrs, (C_b+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.rowIdx, h_matrix.rowIdx, h_matrix.size_rowIdx * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.value, h_matrix.value, h_matrix.size_value * sizeof(T), cudaMemcpyHostToDevice));
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// Kernel that processes a column block per thread block
template <typename T>
__global__ void spmm_bsc_dds_kernel(BSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    unsigned int b = A.block_size;
    unsigned int C_b = (A.C + b - 1) / b;
    unsigned int block_col = blockIdx.x;
    unsigned int thread_col = threadIdx.x;
    unsigned int thread_row = threadIdx.y;

    if(block_col < C_b) {
        for(unsigned int idx = A.colPtrs[block_col]; idx < A.colPtrs[block_col + 1]; idx++) {
            unsigned int block_row = A.rowIdx[idx];
            T* block = &A.value[idx * b * b];

            for(unsigned int n = 0; n < N; n += blockDim.y) {
                unsigned int row = n + thread_row;
                if(row < N) {
                    T temp_sum = 0.0f;
                    for(unsigned int i = 0; i < b; i++) {
                        unsigned int B_col = block_row * b + i;
                        if(B_col < A.R) {
                            temp_sum += B[row * A.R + B_col] * block[i * b + thread_col];
                        }
                    }
                    unsigned int C_col = block_col * b + thread_col;
                    if(C_col < A.C) {
                        atomicAdd(&C[row * A.C + C_col], temp_sum);
                    }
                }
            }
        }
    }
}

template <typename T>
void spmm_bsc_dds(BSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    unsigned int b = A.block_size;
    unsigned int C_b = (A.C + b - 1) / b;
    dim3 blockSize(b, 32);
    dim3 gridSize(C_b);

    spmm_bsc_dds_kernel<T><<<gridSize, blockSize>>>(A, B, C, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

// Kernel that assigns one thread block per BSC block
template <typename T>
__global__ void spmm_bsc_dds_block_kernel(BSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    unsigned int b = A.block_size;
    unsigned int block_idx = blockIdx.x;
    unsigned int thread_row = threadIdx.y;
    unsigned int thread_col = threadIdx.x;

    unsigned int block_col = 0;
    unsigned int left = 0, right = (A.C + b - 1) / b;

    // Linear search to find the block_col
    // while (block_idx >= A.colPtrs[block_col + 1]) {
    //     block_col++;
    // }
    
    // Binary search to find the block_col
    while (left < right) {
        unsigned int mid = (left + right) / 2;
        if (block_idx < A.colPtrs[mid]) {
            right = mid;
        }
        else if (block_idx >= A.colPtrs[mid + 1]) {
            left = mid + 1;
        }
        else {
            block_col = mid;
            break;
        }
    }

    if (left == right) {
        block_col = left;
    }

    unsigned int block_row = A.rowIdx[block_idx];
    const T* block = &A.value[block_idx * b * b];

    for (unsigned int n = 0; n < N; n += blockDim.y) {
        unsigned int row = n + thread_row;
        if (row < N) {
            T temp_sum = 0.0f;
            for (unsigned int i = 0; i < b; i++) {
                unsigned int B_col = block_row * b + i;
                if (B_col < A.R) {
                    temp_sum += B[row * A.R + B_col] * block[i * b + thread_col];
                }
            }
            unsigned int C_col = block_col * b + thread_col;
            if (C_col < A.C) {
                atomicAdd(&C[row * A.C + C_col], temp_sum);
            }
        }
    }
}

template <typename T>
void spmm_bsc_dds_block(BSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    unsigned int b = A.block_size;
    unsigned int num_blocks = A.size_value / (b * b);
    dim3 blockSize(b, 32);
    dim3 gridSize(num_blocks);

    spmm_bsc_dds_block_kernel<T><<<gridSize, blockSize>>>(A, B, C, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

// Kernel that assigns one thread block per BSC block
template <typename T>
__global__ void spmm_bsc_dds_block_shared_kernel(BSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    extern __shared__ T shared_B[];

    unsigned int b = A.block_size;
    unsigned int block_idx = blockIdx.x;
    unsigned int thread_row = threadIdx.y;
    unsigned int thread_col = threadIdx.x;

    unsigned int block_col = 0;
    unsigned int left = 0, right = (A.C + b - 1) / b;

    // Linear search to find the block_col
    // while (block_idx >= A.colPtrs[block_col + 1]) {
    //     block_col++;
    // }
    
    // Binary search to find the block_col
    while (left < right) {
        unsigned int mid = (left + right) / 2;
        if (block_idx < A.colPtrs[mid]) {
            right = mid;
        }
        else if (block_idx >= A.colPtrs[mid + 1]) {
            left = mid + 1;
        }
        else {
            block_col = mid;
            break;
        }
    }

    if (left == right) {
        block_col = left;
    }

    unsigned int block_row = A.rowIdx[block_idx];
    const T* block = &A.value[block_idx * b * b];

    for (unsigned int n = 0; n < N; n += blockDim.y) {
        unsigned int row = n + thread_row;

        // Load B into shared memory
        if(row < N and thread_col < b) {
            unsigned int B_col = block_row * b + thread_col;
            if(B_col < A.R) {
                shared_B[thread_col * blockDim.y + thread_row] = B[row * A.R + B_col];
            }
            else {
                shared_B[thread_col * blockDim.y + thread_row] = 0;
            }
        }

        __syncthreads();

        if (row < N) {
            T temp_sum = 0.0f;
            for (unsigned int i = 0; i < b; i++) {
                temp_sum += shared_B[i * blockDim.y + thread_row] * block[i * b + thread_col];
            }
            unsigned int C_col = block_col * b + thread_col;
            if (C_col < A.C) {
                atomicAdd(&C[row * A.C + C_col], temp_sum);
            }
        }

        __syncthreads();
    }
}

template <typename T>
void spmm_bsc_dds_block_shared(BSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    unsigned int b = A.block_size;
    unsigned int num_blocks = A.size_value / (b * b);
    dim3 blockSize(b, 16);
    dim3 gridSize(num_blocks);

    size_t sharedMemSize = b * blockSize.y * sizeof(T);
    spmm_bsc_dds_block_shared_kernel<T><<<gridSize, blockSize, sharedMemSize>>>(A, B, C, N);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
T compute_torch_mm(T A, T B) {
    T ans = torch::matmul(A, B);
    return ans;
}

template <typename T>
void run_engine(float sparsity_ratio, unsigned int R, unsigned int C, unsigned int N, float abs_tol, double rel_tol) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    SparseMatrix<T> sparse_matrix = generate_sparse_matrix<T>(sparsity_ratio, R, C);

    unsigned int bsc_block_size = 2;
    unsigned int bsc_C_b = (sparse_matrix.C + bsc_block_size - 1) / bsc_block_size;

    BSCMatrix<T> A_h = sparse_to_bsc<T>(sparse_matrix, bsc_block_size);

    T* B_h = nullptr;
    T* C_h = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&B_h, N * R * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_h, N * C * sizeof(T)));

    random_initialize_array(B_h, N * R, 1337);
    std::fill(C_h, C_h + N * C, static_cast<T>(0));

    BSCMatrix<T> A_d;
    T *B_d, *C_d;

    move_bsc_matrix_to_device(A_h, A_d);

    CHECK_CUDA_ERROR(cudaMalloc(&B_d, N * R * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_d, N * C * sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, N * R * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_d, C_h, N * C * sizeof(T), cudaMemcpyHostToDevice));

    std::cout<<"Starting kernel"<<std::endl;
    // spmm_bsc_dds<T>(A_d, B_d, C_d, N);
    // spmm_bsc_dds_block<T>(A_d, B_d, C_d, N);
    spmm_bsc_dds_block_shared<T>(A_d, B_d, C_d, N);
    std::cout<<"Kernel done"<<std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, N * C * sizeof(T), cudaMemcpyDeviceToHost));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor A_t = torch::from_blob(sparse_matrix.mat, {R, C}, options).clone().cuda();
    torch::Tensor B_t = torch::from_blob(B_h, {N, R}, options).clone().cuda();
    torch::Tensor C_cuda = torch::from_blob(C_h, {N, C}, options).clone();

    torch::Tensor C_t = compute_torch_mm<torch::Tensor>(B_t, A_t).cpu();

    std::cout << "CUDA vs Torch allclose: "
              << (torch::allclose(C_cuda, C_t, abs_tol, rel_tol) ? "true" : "false")
              << std::endl;

    CHECK_CUDA_ERROR(cudaFree(A_d.colPtrs));
    CHECK_CUDA_ERROR(cudaFree(A_d.rowIdx));
    CHECK_CUDA_ERROR(cudaFree(A_d.value));
    CHECK_CUDA_ERROR(cudaFree(B_d));
    CHECK_CUDA_ERROR(cudaFree(C_d));
    free(A_h.colPtrs);
    free(A_h.rowIdx);
    free(A_h.value);
    CHECK_CUDA_ERROR(cudaFreeHost(B_h));
    CHECK_CUDA_ERROR(cudaFreeHost(C_h));
}

int main() {
    unsigned int R = ROWS;
    unsigned int C = COLUMNS;
    unsigned int N = DENSE_ROWS;
    float sparsity_ratio = SPARSITY_RATIO;

    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    run_engine<float>(sparsity_ratio, R, C, N, abs_tol, rel_tol);

    return 0;
}