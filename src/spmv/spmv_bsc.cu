#include <iostream>
#include <torch/torch.h>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

#define ROWS            16384
#define COLUMNS         16384
#define SPARSITY_RATIO  0.2
#define BLOCK_SIZE      1024

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
    return (a + b - 1)/b;
}

template <typename T>
__global__ void spmv_bsc_kernel(BSCMatrix<T> A, T* x, T* y) {
    unsigned int b = A.block_size;
    unsigned int C_b = (A.C + b - 1) / b;
    unsigned int block_col = blockIdx.x * blockDim.x + threadIdx.x;

    if(block_col < C_b) {
        for(int idx = A.colPtrs[block_col]; idx < A.colPtrs[block_col+1]; idx++) {
            unsigned int block_row = A.rowIdx[idx];
            T* block = &A.value[idx * b * b];

            for(unsigned int i=0; i<b; i++) {
                unsigned int row = block_row * b + i;
                T temp_sum = 0.0f;

                if(row < A.R) {
                    for(unsigned int j=0; j<b; j++) {
                        unsigned int col = block_col * b + j;
                        if(col < A.C) {
                            temp_sum += block[i*b + j] * x[col];
                        }
                    }
                }
                atomicAdd(&y[row], temp_sum);
            }
        }
    }
}

template <typename T>
void spmv_bsc(BSCMatrix<T> A, T* x, T* y) {
    unsigned int C_b = (A.C + A.block_size - 1) /  A.block_size;
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(C_b, blockSize.x));

    spmv_bsc_kernel<T><<<gridSize, blockSize>>>(A, x, y);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
__global__ void spmv_bsc_block_kernel(BSCMatrix<T> A, T* x, T* y) {
    extern __shared__ T shared_x[];

    unsigned int b = A.block_size;
    unsigned int block_idx = blockIdx.x;
    unsigned int block_row_idx = threadIdx.x;

    // Identify the block row by finding which range of colPtrs the block_idx falls into
    unsigned int block_col = 0;
    // Linear search to find the block_col
    // while (block_idx >= A.colPtrs[block_col + 1]) {
    //     block_col++;
    // }
    
    // Binary search to find the block_col
    unsigned int left = 0, right = (A.C + b - 1) / b;

    while(left < right) {
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

    if(left == right) {
        block_col = left;
    }

    // The specific block within this row
    unsigned int local_block_idx = block_idx - A.colPtrs[block_col];
    unsigned int block_row = A.rowIdx[A.colPtrs[block_col] + local_block_idx];

    if(block_row_idx < b) {
        shared_x[block_row_idx] = x[block_col * b + block_row_idx];
    }
    __syncthreads();

    if(block_row_idx < b) {
        T* block = &A.value[block_idx * b * b];
        T temp_sum = 0;
        for(int j = 0; j < b; j++) {
            temp_sum += block[block_row_idx * b + j] * shared_x[j];
        }
        
        unsigned int row = block_row * b + block_row_idx;
        if(row < A.R) {
            atomicAdd(&y[row], temp_sum);
        }
    }
}

template <typename T>
void spmv_bsc_block(BSCMatrix<T> A, T* x, T* y) {
    unsigned int b = A.block_size;
    unsigned int num_blocks = A.size_value / (b * b);
    dim3 blockSize(b);
    dim3 gridSize(num_blocks);

    size_t sharedMemSize = b * sizeof(T);

    spmv_bsc_block_kernel<T><<<gridSize, blockSize, sharedMemSize>>>(A, x, y);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
T compute_torch_mv(T A, T x) {
    T ans = torch::matmul(A, x);
    return ans;
}

template <typename T>
void spmv_bsc_cpu(BSCMatrix<T> bsc, T* x, T* y) {
    unsigned int b = bsc.block_size;
    unsigned int C_b = (bsc.C + b - 1)/b;

    for(unsigned int block_col = 0; block_col < C_b; block_col++) {
        for(unsigned int idx = bsc.colPtrs[block_col]; idx < bsc.colPtrs[block_col+1]; idx++) {
            unsigned int block_row = bsc.rowIdx[idx];
            T* block = &bsc.value[idx * b * b];

            for(unsigned int i=0; i<b; i++) {
                unsigned int row = block_row * b + i;
                if(row >= bsc.R)
                    continue;
        
                for(unsigned j=0; j<b; j++) {
                    unsigned int col = block_col * b + j;
                    if(col < bsc.C) {
                        y[row] += block[i * b + j] * x[col];
                    }
                }
            }
        }
    }
}


template <typename T>
void run_engine(float sparsity_ratio, unsigned int R, unsigned int C, float abs_tol, double rel_tol) {
    SparseMatrix<T> sparse_matrix = generate_sparse_matrix<T>(sparsity_ratio, R, C);
    
    unsigned int bsc_block_size = 32;
    unsigned int R_b = (sparse_matrix.R + bsc_block_size - 1) / bsc_block_size;
    unsigned int C_b = (sparse_matrix.C + bsc_block_size - 1) / bsc_block_size;

    BSCMatrix<T> A_h = sparse_to_bsc<T>(sparse_matrix, bsc_block_size);

    T* x_h = nullptr;
    T* y_h = nullptr;
    T* y_h_cpu_ref = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&x_h, C*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&y_h, R*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&y_h_cpu_ref, R*sizeof(T)));

    random_initialize_array(x_h, C, 1337);
    std::fill(y_h, y_h+R, static_cast<T>(0));
    std::fill(y_h_cpu_ref, y_h_cpu_ref+R, static_cast<T>(0));

    spmv_bsc_cpu<T>(A_h, x_h, y_h_cpu_ref);
    // print_array<T>(y_h_cpu_ref, R, "SpMV output CPU");

    BSCMatrix<T> A_d;
    T *x_d, *y_d;

    move_bsc_matrix_to_device(A_h, A_d);

    CHECK_CUDA_ERROR(cudaMalloc(&x_d, C*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&y_d, R*sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(x_d, x_h, C*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(y_d, y_h, R*sizeof(T), cudaMemcpyHostToDevice));

    spmv_bsc<T>(A_d, x_d, y_d);
    // spmv_bsc_block<T>(A_d, x_d, y_d);

    CHECK_CUDA_ERROR(cudaMemcpy(y_h, y_d, R*sizeof(T), cudaMemcpyDeviceToHost));
    // print_array<T>(y_h, R, "SpMV output CUDA");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor A_t = torch::from_blob(sparse_matrix.mat, {R, C}, options).clone().cuda();
    torch::Tensor x_t = torch::from_blob(x_h, {C}, options).clone().cuda();
    torch::Tensor y_cpu = torch::from_blob(y_h_cpu_ref, {R}, options).clone();
    torch::Tensor y_cuda = torch::from_blob(y_h, {R}, options).clone();

    torch::Tensor y_t = compute_torch_mv<torch::Tensor>(A_t, x_t).cpu();

    // std::cout<<"From CPU "<<y_cpu<<std::endl;
    // std::cout<<"From CUDA "<<y_cuda<<std::endl;
    // std::cout<<"From Torch "<<y_t<<std::endl;

    std::cout   << "CPU vs CUDA allclose: "
                << (all_close<T>(y_h_cpu_ref, y_h, sparse_matrix.R, abs_tol, rel_tol) ? "true" : "false")
                << std::endl;

    std::cout   << "CPU vs Torch allclose: "
                << (torch::allclose(y_cpu, y_t, abs_tol, rel_tol) ? "true" : "false")
                << std::endl;

    std::cout   << "CUDA vs Torch allclose: "
                << (torch::allclose(y_cuda, y_t, abs_tol, rel_tol) ? "true" : "false")
                << std::endl;

    CHECK_CUDA_ERROR(cudaFree(A_d.colPtrs));
    CHECK_CUDA_ERROR(cudaFree(A_d.rowIdx));
    CHECK_CUDA_ERROR(cudaFree(A_d.value));
    CHECK_CUDA_ERROR(cudaFree(x_d));
    CHECK_CUDA_ERROR(cudaFree(y_d));
    free(A_h.colPtrs);
    free(A_h.rowIdx);
    free(A_h.value);
    CHECK_CUDA_ERROR(cudaFreeHost(x_h));
    CHECK_CUDA_ERROR(cudaFreeHost(y_h));
}

int main() {
    unsigned int R = ROWS;
    unsigned int C = COLUMNS;
    float sparsity_ratio = SPARSITY_RATIO;

    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;

    run_engine<float>(sparsity_ratio, R, C, abs_tol, rel_tol);

    return 0;
}