#include <iostream>
#include <torch/torch.h>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

#define ROWS            16384
#define COLUMNS         16384
#define SPARSITY_RATIO  0.2
#define BLOCK_SIZE      1024

template <typename T>
void move_csr_matrix_to_device(CSRMatrix<T>& h_matrix, CSRMatrix<T>& d_matrix) {
    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.rowPtrs, (h_matrix.R+1) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.colIdx, h_matrix.num_nonzero * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.value, h_matrix.num_nonzero * sizeof(T)));

    // Set dimensions
    d_matrix.R = h_matrix.R;
    d_matrix.C = h_matrix.C;
    d_matrix.num_nonzero = h_matrix.num_nonzero;

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.rowPtrs, h_matrix.rowPtrs, (h_matrix.R+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.colIdx, h_matrix.colIdx, h_matrix.num_nonzero * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.value, h_matrix.value, h_matrix.num_nonzero * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
__global__ void spmv_csr_kernel(CSRMatrix<T> A, T* x, T* y) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < A.R) {
        float row_sum = 0.0f;
        for(unsigned int i=A.rowPtrs[row]; i<A.rowPtrs[row+1]; i++) {
            unsigned int col_idx = A.colIdx[i];
            T value = A.value[i];

            row_sum += value * x[col_idx];
        }
        y[row] += row_sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1)/b;
}

template <typename T>
void spmv_csr(CSRMatrix<T> A, T* x, T* y) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(cdiv(A.R, blockSize.x));

    spmv_csr_kernel<T><<<gridSize, blockSize>>>(A, x, y);
    
    CHECK_LAST_CUDA_ERROR();    
}

template <typename T>
T compute_torch_mv(T A, T x) {
    T ans = torch::matmul(A, x);
    return ans;
}

template <typename T>
void run_engine(float sparsity_ratio, unsigned int R, unsigned int C, float abs_tol, double rel_tol) {
    SparseMatrix<T> sparse_matrix = generate_sparse_matrix<T>(sparsity_ratio, R, C);
    
    CSRMatrix<T> A_h = sparse_to_csr<T>(sparse_matrix);
    T* x_h = nullptr;
    T* y_h = nullptr;

    CHECK_CUDA_ERROR(cudaMallocHost(&x_h, C*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMallocHost(&y_h, R*sizeof(T)));

    random_initialize_array(x_h, C, 1337);
    std::fill(y_h, y_h+R, static_cast<T>(0));

    CSRMatrix<T> A_d;
    T *x_d, *y_d;

    move_csr_matrix_to_device(A_h, A_d);

    CHECK_CUDA_ERROR(cudaMalloc(&x_d, C*sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&y_d, R*sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpy(x_d, x_h, C*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(y_d, y_h, R*sizeof(T), cudaMemcpyHostToDevice));

    spmv_csr<T>(A_d, x_d, y_d);

    CHECK_CUDA_ERROR(cudaMemcpy(y_h, y_d, R*sizeof(T), cudaMemcpyDeviceToHost));
    // print_array<T>(y_h, R, "SpMV output");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor A_t = torch::from_blob(sparse_matrix.mat, {R, C}, options).clone().cuda();
    torch::Tensor x_t = torch::from_blob(x_h, {C}, options).clone().cuda();
    torch::Tensor y_cuda = torch::from_blob(y_h, {R}, options).clone();

    torch::Tensor y_t = compute_torch_mv<torch::Tensor>(A_t, x_t).cpu();

    // std::cout<<"From CUDA "<<y_cuda<<std::endl;
    // std::cout<<"From Torch "<<y_t<<std::endl;

    std::cout   << "CUDA vs Torch allclose: "
                << (torch::allclose(y_cuda, y_t, abs_tol, rel_tol) ? "true" : "false")
                << std::endl;

    CHECK_CUDA_ERROR(cudaFree(A_d.rowPtrs));
    CHECK_CUDA_ERROR(cudaFree(A_d.colIdx));
    CHECK_CUDA_ERROR(cudaFree(A_d.value));
    CHECK_CUDA_ERROR(cudaFree(x_d));
    CHECK_CUDA_ERROR(cudaFree(y_d));
    free(A_h.rowPtrs);
    free(A_h.colIdx);
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