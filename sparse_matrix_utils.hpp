#pragma once

template<typename T>
struct SparseMatrix {
    T* mat;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
struct COOMatrix {
    unsigned int* rowIdx;
    unsigned int* colIdx;
    T* value;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
struct CSRMatrix {
    unsigned int* rowPtrs;
    unsigned int* colIdx;
    T* value;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
struct CSCMatrix {
    unsigned int* colPtrs;
    unsigned int* rowIdx;
    T* value;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
}; 

template <typename T>
struct ELLMatrix {
    unsigned int* rowPtrs;
    unsigned int* colIdx;
    T* value;
    unsigned int max_nz_in_row;
    unsigned int R;
    unsigned int C;
    unsigned int num_nonzero;
};

template <typename T>
struct BSRMatrix {
    unsigned int* rowPtrs;    // Points to the beginning of each row of blocks
    unsigned int* colIdx;     // Column indices of the blocks
    T* value;                 // Non-zero blocks stored in a flat array
    unsigned int R;           // Number of rows in the original matrix
    unsigned int C;           // Number of columns in the original matrix
    unsigned int num_nonzero; // Number of non-zero blocks
    unsigned int size_colIdx; // Number of total blocks
    unsigned int size_value;  // Total elements in flat array value
    unsigned int block_size;  // Size of the block (b x b)
};

template <typename T>
struct BSCMatrix {
    unsigned int* colPtrs;    // Points to the beginning of each column of blocks
    unsigned int* rowIdx;     // Row indices of the blocks
    T* value;                 // Non-zero blocks stored in a flat array
    unsigned int R;           // Number of rows in the original matrix
    unsigned int C;           // Number of columns in the original matrix
    unsigned int num_nonzero; // Number of non-zero blocks
    unsigned int size_rowIdx; // Number of total blocks
    unsigned int size_value;  // Total elements in flat array value
    unsigned int block_size;  // Size of the block (b x b)
};

template <typename T>
int get_num_nonzero(T*A, unsigned int R, unsigned int C);

template <typename T>
SparseMatrix<T> generate_sparse_matrix(float sparsity_ratio, unsigned int R, unsigned int C);

template <typename T>
COOMatrix<T> sparse_to_coo(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> coo_to_sparse(COOMatrix<T> A);

template <typename T>
CSRMatrix<T> sparse_to_csr(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> csr_to_sparse(CSRMatrix<T> A);

template <typename T>
CSCMatrix<T> sparse_to_csc(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> csc_to_sparse(CSCMatrix<T> A);

template <typename T>
ELLMatrix<T> sparse_to_ell(SparseMatrix<T> A);

template <typename T>
SparseMatrix<T> ell_to_sparse(ELLMatrix<T> A);

template <typename T>
BSRMatrix<T> sparse_to_bsr(SparseMatrix<T> A, unsigned int block_size);

template <typename T>
SparseMatrix<T> bsr_to_sparse(BSRMatrix<T> A);

template <typename T>
BSCMatrix<T> sparse_to_bsc(SparseMatrix<T> A, unsigned int block_size);

template <typename T>
SparseMatrix<T> bsc_to_sparse(BSCMatrix<T> A);

