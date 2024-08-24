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
void move_csc_matrix_to_device(CSCMatrix<T>& h_matrix, CSCMatrix<T>& d_matrix) {
    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.colPtrs, (h_matrix.C+1) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.rowIdx, h_matrix.num_nonzero * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.value, h_matrix.num_nonzero * sizeof(T)));

    // Set dimensions
    d_matrix.R = h_matrix.R;
    d_matrix.C = h_matrix.C;
    d_matrix.num_nonzero = h_matrix.num_nonzero;

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.colPtrs, h_matrix.colPtrs, (h_matrix.C+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.rowIdx, h_matrix.rowIdx, h_matrix.num_nonzero * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.value, h_matrix.value, h_matrix.num_nonzero * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
__global__ void spmm_csc_dds_kernel(CSCMatrix<T> A, T* B, T* C, unsigned int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(c < A.C and r < N) {
        T sum = 0.0f;
        unsigned int col_start = A.colPtrs[c];
        unsigned int col_end = A.colPtrs[c+1];

        for(unsigned int i=col_start; i<col_end; i++) {
            unsigned int r_idx = A.rowIdx[i];
            T val = A.value[i];
            sum += B[r*A.R + r_idx] * val;
        }
        atomicAdd(&C[r*A.C + c], sum);
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

template <typename T>
void spmm_csc_dds(CSCMatrix<T> A, T* B, T* C, int N) {
    // A -> sparse matrix -> R x C
    // B -> dense matrix -> N x R
    // C -> dense matrix -> C = B @ A -> N x C
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(cdiv(A.C, blockSize.x), cdiv(N, blockSize.y));

    spmm_csc_dds_kernel<T><<<gridSize, blockSize>>>(A, B, C, N);
    
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
    
    CSCMatrix<T> A_h = sparse_to_csc<T>(sparse_matrix);
    T* B_h = nullptr;
    T* C_h = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&B_h, N * R * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&C_h, N * C * sizeof(T)));

    random_initialize_array(B_h, N * R, 1337);
    std::fill(C_h, C_h + N * C, static_cast<T>(0));

    CSCMatrix<T> A_d;
    T *B_d, *C_d;

    move_csc_matrix_to_device(A_h, A_d);

    CHECK_CUDA_ERROR(cudaMalloc(&B_d, N * R * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&C_d, N * C * sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, N * R * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(C_d, C_h, N * C * sizeof(T), cudaMemcpyHostToDevice));

    std::cout<<"Starting kernel"<<std::endl;
    spmm_csc_dds<T>(A_d, B_d, C_d, N);
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
