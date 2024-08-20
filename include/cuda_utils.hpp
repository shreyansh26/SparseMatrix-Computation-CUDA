#pragma once

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line);

#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)
void check_cuda_last(const char* const file, const int line);

template <typename T>
void random_initialize_array(T* A, unsigned int N, unsigned int seed);

template <typename T>
void random_initialize_sparse_matrix(T* A, float sparsity_ratio, unsigned int R, unsigned int C);

template <typename T>
void print_array(T* A, unsigned int N, std::string msg);

template <typename T>
void print_matrix(T* A, unsigned int R, unsigned int C, std::string msg);

template <typename T>
bool all_close(T* A, T* A_ref, unsigned int N, float abs_tol, double rel_tol);