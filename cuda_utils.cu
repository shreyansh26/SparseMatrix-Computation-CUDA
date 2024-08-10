#include <iostream>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

#include "cuda_utils.hpp"

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_cuda_last(const char* const file, const int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void random_initialize_array(T* A, unsigned int N, unsigned int seed) {
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    auto const rand = [&dis, &eng]() {return dis(eng);};

    for(unsigned int i=0; i<N; i++)
        A[i] = static_cast<T>(rand());
}

template <typename T>
void random_initialize_sparse_matrix(T* A, float sparsity_ratio, unsigned int R, unsigned int C) {
    // Initialize random number generators
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> dis_value(-10.0f, 10.0f);
    std::uniform_int_distribution<unsigned int> dis_row(0, R-1);
    std::uniform_int_distribution<unsigned int> dis_col(0, C-1);
    
    auto const rand = [&dis_value, &eng]() {return dis_value(eng);}; 
    
    // Calculate the number of non-zero elements based on the sparsity ratio
    unsigned int num_elements = R * C;
    unsigned int num_nonzero = static_cast<unsigned int>(num_elements * sparsity_ratio);
    
    // Set all initial elements to zero
    std::fill(A, A + num_elements, static_cast<T>(0));
    
    // Insert random non-zero elements
    for(unsigned int i = 0; i<num_nonzero; i++) {
        unsigned int row, col;
        do {
            row = dis_row(eng);
            col = dis_col(eng);
        } while(A[row * C + col] != 0); // Ensure unique locations
        
        A[row * C + col] = static_cast<T>(rand());
    }
}

template <typename T>
void print_array(T* A, unsigned int N, std::string msg) {
    std::cout<<msg<<std::endl;
    for(unsigned int i=0; i<N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout<<std::endl;
}

template <typename T>
void print_matrix(T* A, unsigned int R, unsigned int C, std::string msg) {
    std::cout<<msg<<std::endl;
    for(unsigned int i=0; i<R; i++) {
        for(unsigned int j=0; j<C; j++) {
            std::cout << A[i * C + j] << "\t";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

template <typename T>
bool all_close(T* A, T* A_ref, unsigned int N, float abs_tol, double rel_tol) {
    bool is_close = true;
    for(unsigned int i=0; i<N; i++) {
        double A_val = static_cast<double>(A[i]);
        double A_ref_val = static_cast<double>(A_ref[i]);

        double diff_val = std::abs(A_val - A_ref_val);

        if(diff_val > std::max(static_cast<double>(abs_tol), static_cast<double>(std::abs(A_ref_val)) * rel_tol)) {
            std::cout   << "A[" << i << "] = " << A_val
                        << ", A_ref[" << i << "] = " << A_ref_val
                        << ", Abs Diff Threshold: "
                        << static_cast<double>(abs_tol)
                        << ", Rel->Abs Diff Threshold: "
                        << static_cast<double>(static_cast<double>(std::abs(A_ref_val)) * rel_tol)
                        << std::endl;
            is_close = false;
            return is_close;
        }
    }
    return is_close;
}

template void random_initialize_array<unsigned int>(unsigned int*, unsigned int, unsigned int);
template void print_array<unsigned int>(unsigned int*, unsigned int, std::string);
template bool all_close<unsigned int>(unsigned int*, unsigned int*, unsigned int, float, double);

template void random_initialize_array<float>(float*, unsigned int, unsigned int);
template void random_initialize_sparse_matrix<float>(float*, float, unsigned int, unsigned int);
template void print_array<float>(float*, unsigned int, std::string);
template void print_matrix<float>(float*, unsigned int, unsigned int, std::string);
template bool all_close<float>(float*, float*, unsigned int, float, double);

template void random_initialize_array<double>(double*, unsigned int, unsigned int);
template void random_initialize_sparse_matrix<double>(double*, float, unsigned int, unsigned int);
template void print_array<double>(double*, unsigned int, std::string);
template void print_matrix<double>(double*, unsigned int, unsigned int, std::string);
template bool all_close<double>(double*, double*, unsigned int, float, double);