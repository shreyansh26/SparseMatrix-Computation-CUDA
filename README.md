# Sparse Matrix Computations in CUDA

This project implements various sparse matrix computations in CUDA and C++. It includes conversion routines between sparse matrix formats and efficient CUDA kernels for Sparse Matrix-Vector Multiplication (SpMV) and Sparse Matrix-Matrix Multiplication (SpMM). 

## Directory Structure

```
sparse_matrix_computation/
├── include/
│   ├── cuda_utils.hpp              # Common CUDA utility functions
│   └── sparse_matrix_utils.hpp     # Sparse matrix utilities
├── lib/                            # Place libtorch here after downloading
├── src/
│   ├── sparse_matrix_conversion/
│   │   ├── coo_to_csr.cu           # CUDA kernel for COO to CSR conversion
│   │   ├── Makefile_coo_to_csr     # Makefile for COO to CSR conversion
│   │   ├── Makefile_sparse_matrix_conversion # Makefile for format conversions
│   │   └── sparse_matrix_conversion.cpp  # C++ driver for matrix conversions
│   ├── spmm/                       # Sparse Matrix-Matrix Multiplication (SpMM)
│   │   ├── Makefile_spmm_*         # Makefiles for various SpMM kernels
│   │   ├── spmm_*.cu               # SpMM kernels for different sparse formats
│   ├── spmv/                       # Sparse Matrix-Vector Multiplication (SpMV)
│   │   ├── Makefile_spmv_*         # Makefiles for various SpMV kernels
│   │   ├── spmv_*.cu               # SpMV kernels for different sparse formats
│   ├── cuda_utils.cu               # CUDA utility functions
│   └── sparse_matrix_utils.cpp     # C++ utility functions for sparse matrices
├── Makefile                        # Root Makefile for all the kernels in the project
└── README.md
```

## Dependencies

- **CUDA**: The core of this project runs on CUDA-enabled GPUs. Kernels have been tested on H100s though do not include any arch specific operations.
- **Libtorch**: Some kernels require libtorch. Download and place it in the `lib/` directory.

### Libtorch Setup

Libtorch has been used to ensure correctness of some of the matmul kernels by performing the operations in Pytorch (Libtorch provides a convenient way to use it in C++). For kernels that require libtorch, download the appropriate version from [Pytorch's official site](https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu121.zip), unzip it and place it in the `lib/` directory. Ensure the `Makefile` paths are correctly configured to link against libtorch.

## Sparse Matrix Formats

The project has implementations for the following sparse matrix formats:

- **COO (Coordinate Format)**: Each non-zero element is stored with its row and column indices.
- **CSR (Compressed Sparse Row)**: Rows are compressed into a single array, making row-wise access faster.
- **CSC (Compressed Sparse Column)**: Like CSR but columns are compressed.
- **BSR (Block Sparse Row)**: Similar to CSR but groups values into blocks for better memory access.
- **BSC (Block Sparse Column)**: Column version of BSR.
- **ELL (Ellpack)**: Stores data in a fixed-width format for better SIMD processing.

## Key Components

### 1. Sparse Matrix Conversion

The `sparse_matrix_conversion` directory contains CUDA kernels and C++ code to convert between different sparse matrix formats. It also supports format-to-format conversions like:

- Full matrix to sparse formats (COO, CSR, CSC, etc.) and back to full matrix
- COO to CSR conversion (`coo_to_csr.cu`)

### 2. Sparse Matrix-Vector Multiplication (SpMV)

The `spmv` directory contains CUDA implementations of SpMV kernels.

- **spmv_coo.cu**: SpMV for COO format
- **spmv_csr.cu**: SpMV for CSR format
- **spmv_csc.cu**: SpMV for CSC format
- **spmv_bsr.cu**: SpMV for BSR format
- **spmv_bsc.cu**: SpMV for BSC format
- **spmv_ell.cu**: SpMV for ELL format

### 3. Sparse Matrix-Matrix Multiplication (SpMM)

The `spmm` folder contains kernels for Sparse Matrix-Matrix Multiplication (SpMM), supporting DDS (Dense = Dense x Sparse) and DSD (Dense = Sparse x Dense) operations. The formats supported include:

- **spmm_bsr_dsd.cu** - DSD SpMM with sparse matrix in BSR format
- **spmm_bsc_dds.cu** - DDS SpMM with sparse matrix in BSC format
- **spmm_csr_dsd.cu** - DSD SpMM with sparse matrix in CSR format
- **spmm_csc_dds.cu** - DDS SpMM with sparse matrix in CSC format

## Compilation

To compile the project, navigate to the respective directories (`spmv`, `spmm`, or `sparse_matrix_conversion`) and run the provided `Makefile` for the specific kernel. For example:

```bash
cd src/spmv
make -f Makefile_spmv_csr
```

## Future Work

- Optimizing kernel performance. Currently, the kernels are not highly-optimized.
- Adding some block-sparse matmul implementations in Triton