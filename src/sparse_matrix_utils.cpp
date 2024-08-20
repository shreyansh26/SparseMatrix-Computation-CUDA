#include <iostream>
#include <vector>
#include <climits>

#include "sparse_matrix_utils.hpp"
#include "cuda_utils.hpp"

#define ROWS            10
#define COLUMNS         10
#define SPARSITY_RATIO  0.2
#define PAD_VAL         (1<<20)

template <typename T>
int get_num_nonzero(T*A, unsigned int R, unsigned int C) {
    unsigned int num_nonzero = 0;
    for(unsigned int row=0; row<R; row++) {
        for(unsigned int col=0; col<C; col++) {
            if(A[row*C + col] != 0) {
                num_nonzero++;
            }
        }
    }
    return num_nonzero;
}

template <typename T>
SparseMatrix<T> generate_sparse_matrix(float sparsity_ratio, unsigned int R, unsigned int C) {
    T* A = (T*)malloc(R*C*sizeof(T));

    random_initialize_sparse_matrix<T>(A, sparsity_ratio, R, C);
    unsigned int num_nonzero = get_num_nonzero(A, R, C);

    SparseMatrix<T> sparse_matrix = {A, R, C, num_nonzero};
    return sparse_matrix;
}

template <typename T>
COOMatrix<T> sparse_to_coo(SparseMatrix<T> A) {
    unsigned int* rowIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    unsigned int* colIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    T* value = (T*)malloc(A.num_nonzero * sizeof(T));

    int cntr = 0;
    for(unsigned int row=0; row<A.R; row++) {
        for(unsigned int col=0; col<A.C; col++) {
            if(A.mat[row*A.C + col] != 0) {
                rowIdx[cntr] = row;
                colIdx[cntr] = col;
                value[cntr] = A.mat[row*A.C + col];
                cntr++;
            }
        }
    }
    COOMatrix<T> coo_matrix = {rowIdx, colIdx, value, A.R, A.C, A.num_nonzero};
    return coo_matrix;
}

template <typename T>
SparseMatrix<T> coo_to_sparse(COOMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i<A.num_nonzero; i++) {
        mat[A.rowIdx[i]*A.C + A.colIdx[i]] = A.value[i];
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
CSRMatrix<T> sparse_to_csr(SparseMatrix<T> A) {
    unsigned int* rowPtrs = (unsigned int*)malloc((A.R+1) * sizeof(unsigned int));
    unsigned int* colIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    T* value = (T*)malloc(A.num_nonzero * sizeof(T));

    unsigned int row_cntr = 0;
    unsigned int cntr = 0;
    for(unsigned int row=0; row<A.R; row++) {
        rowPtrs[row_cntr] = cntr;
        for(unsigned int col=0; col<A.C; col++) {
            if(A.mat[row*A.C + col] != 0) {
                colIdx[cntr] = col;
                value[cntr] = A.mat[row*A.C + col];
                cntr++;
            }
        }
        row_cntr++;
    }
    rowPtrs[row_cntr] = cntr;
    CSRMatrix<T> csr_matrix = {rowPtrs, colIdx, value, A.R, A.C, A.num_nonzero};
    return csr_matrix;
}

template <typename T>
SparseMatrix<T> csr_to_sparse(CSRMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i<A.R; i++) {
        for(unsigned int j=A.rowPtrs[i]; j<A.rowPtrs[i+1]; j++) {
            unsigned int row_idx = i;
            unsigned int col_idx = A.colIdx[j];
            mat[row_idx*A.C + col_idx] = A.value[j];
        }
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
CSCMatrix<T> sparse_to_csc(SparseMatrix<T> A) {
    unsigned int* colPtrs = (unsigned int*)malloc((A.C+1) * sizeof(unsigned int));
    unsigned int* rowIdx = (unsigned int*)malloc(A.num_nonzero * sizeof(unsigned int));
    T* value = (T*)malloc(A.num_nonzero * sizeof(T));

    unsigned int col_cntr = 0;
    unsigned int cntr = 0;
    for(unsigned int col=0; col<A.C; col++) {
        colPtrs[col_cntr] = cntr;
        for(unsigned int row=0; row<A.R; row++) {
            if(A.mat[row*A.C + col] != 0) {
                rowIdx[cntr] = row;
                value[cntr] = A.mat[row*A.C + col];
                cntr++;
            }
        }
        col_cntr++;
    }
    colPtrs[col_cntr] = cntr;
    CSCMatrix<T> csc_matrix = {colPtrs, rowIdx, value, A.R, A.C, A.num_nonzero};
    return csc_matrix;
}

template <typename T>
SparseMatrix<T> csc_to_sparse(CSCMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i<A.C; i++) {
        for(unsigned int j=A.colPtrs[i]; j<A.colPtrs[i+1]; j++) {
            unsigned int col_idx = i;
            unsigned int row_idx = A.rowIdx[j];
            mat[row_idx*A.C + col_idx] = A.value[j];
        }
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
ELLMatrix<T> sparse_to_ell(SparseMatrix<T> A) {
    unsigned int max_nz_in_row = 0;

    for(unsigned int i=0; i<A.R; i++) {
        unsigned int nz_in_row = 0;
        for(unsigned int j=0; j<A.C; j++) {
            if(A.mat[i*A.C + j] != 0) {
                nz_in_row++;
            }
        }
        max_nz_in_row = std::max(max_nz_in_row, nz_in_row);
    }
    unsigned int* rowPtrs = (unsigned int*)malloc((A.R+1) * sizeof(unsigned int));
    unsigned int* colIdx = (unsigned int*)malloc((A.R * max_nz_in_row) * sizeof(unsigned int));
    T* value = (T*)malloc((A.R * max_nz_in_row) * sizeof(T));

    std::fill(colIdx, colIdx+(A.R * max_nz_in_row), static_cast<unsigned int>(PAD_VAL));
    std::fill(value, value+(A.R * max_nz_in_row), static_cast<T>(PAD_VAL));

    unsigned int row_cntr = 0;
    unsigned int col_cntr = 0;
    unsigned int cntr = 0;
    for(unsigned int row=0; row<A.R; row++) {
        rowPtrs[row_cntr] = cntr;
        for(unsigned int col=0; col<A.C; col++) {
            if(A.mat[row*A.C + col] != 0) {
                // i, j -> row, col_cntr
                // => j, i -> col_cntr, row
                // colIdx[row*max_nz_in_row + col_cntr] = col;
                // value[row*max_nz_in_row + col_cntr] = A.mat[row*A.C + col];
                colIdx[col_cntr*A.R + row] = col;
                value[col_cntr*A.R + row] = A.mat[row*A.C + col];
                cntr++;
                col_cntr++;
            }
        }
        col_cntr = 0;
        row_cntr++;
    }
    rowPtrs[row_cntr] = cntr;

    ELLMatrix<T> ell_matrix = {rowPtrs, colIdx, value, max_nz_in_row, A.R, A.C, A.num_nonzero};
    return ell_matrix;
}

template <typename T>
SparseMatrix<T> ell_to_sparse(ELLMatrix<T> A) {
    T* mat = (T*)malloc(A.R * A.C * sizeof(T));

    std::fill(mat, mat+A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i<A.R; i++) {
        for(unsigned int j=i; j<A.max_nz_in_row*A.R; j+=A.R) {
            unsigned int col_idx = A.colIdx[j];
            if(col_idx != static_cast<T>(PAD_VAL))
                mat[i*A.C + col_idx] = A.value[j];
        }
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
BSRMatrix<T> sparse_to_bsr(SparseMatrix<T> A, unsigned int block_size) {
    unsigned int b = block_size;
    unsigned int R_b = (A.R + b - 1) / b;  // Number of block rows
    unsigned int C_b = (A.C + b - 1) / b;  // Number of block columns

    std::vector<unsigned int> rowPtrs(R_b + 1, 0);
    std::vector<unsigned int> colIdx;
    std::vector<T> values;

    for(unsigned int i = 0; i < R_b; i++) {
        for(unsigned int j = 0; j < C_b; j++) {
            bool block_nonzero = false;
            for(unsigned int bi = 0; bi < b; bi++) {
                for(unsigned int bj = 0; bj < b; bj++) {
                    unsigned int row = i * b + bi;
                    unsigned int col = j * b + bj;
                    if(row < A.R and col < A.C and A.mat[row*A.C + col] != 0) {
                        block_nonzero = true;
                        break;
                    }
                }
                if(block_nonzero) 
                    break;
            }
            if(block_nonzero) {
                colIdx.push_back(j);
                for(unsigned int bi = 0; bi < b; bi++) {
                    for (unsigned int bj = 0; bj < b; bj++) {
                        unsigned int row = i * b + bi;
                        unsigned int col = j * b + bj;
                        if(row < A.R and col < A.C) {
                            values.push_back(A.mat[row*A.C + col]);
                        } 
                        else {
                            values.push_back(0);
                        }
                    }
                }
                rowPtrs[i+1]++;
            }
        }
    }

    // Convert rowPtrs to cumulative sum
    for(unsigned int i = 1; i <= R_b; i++) {
        rowPtrs[i] += rowPtrs[i - 1];
    }
    
    // Create the BSR matrix
    unsigned int* bsr_row_ptrs = (unsigned int*)malloc(rowPtrs.size() * sizeof(unsigned int));
    unsigned int* bsr_col_idx = (unsigned int*)malloc(colIdx.size() * sizeof(unsigned int));
    T* bsr_value = (T*)malloc(values.size() * sizeof(T));

    std::copy(rowPtrs.begin(), rowPtrs.end(), bsr_row_ptrs);
    std::copy(colIdx.begin(), colIdx.end(), bsr_col_idx);
    std::copy(values.begin(), values.end(), bsr_value);

    BSRMatrix<T> bsr_matrix = {bsr_row_ptrs, bsr_col_idx, bsr_value, A.R, A.C, A.num_nonzero, (unsigned int)colIdx.size(), (unsigned int)values.size(), b};
    return bsr_matrix;
}

template <typename T>
SparseMatrix<T> bsr_to_sparse(BSRMatrix<T> A) {
    unsigned int b = A.block_size;
    T* mat = (T*)malloc(A.R*A.C * sizeof(T));
    std::fill(mat, mat + A.R*A.C, static_cast<T>(0));

    for(unsigned int i=0; i < (A.R + b - 1) / b; i++) {
        for(unsigned int j = A.rowPtrs[i]; j < A.rowPtrs[i+1]; j++) {
            unsigned int col_block = A.colIdx[j];
            for(unsigned int bi = 0; bi < b; bi++) {
                for(unsigned int bj = 0; bj < b; bj++) {
                    unsigned int row = i * b + bi;
                    unsigned int col = col_block * b + bj;
                    if(row < A.R and col < A.C) {
                        mat[row*A.C + col] = A.value[j * b * b + bi * b + bj];
                    }
                }
            }
        }
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}

template <typename T>
BSCMatrix<T> sparse_to_bsc(SparseMatrix<T> A, unsigned int block_size) {
    unsigned int b = block_size;
    unsigned int R_b = (A.R + b - 1) / b;  // Number of block rows
    unsigned int C_b = (A.C + b - 1) / b;  // Number of block columns

    std::vector<unsigned int> colPtrs(C_b + 1, 0);
    std::vector<unsigned int> rowIdx;
    std::vector<T> values;

    for(unsigned int j = 0; j < C_b; j++) {
        for(unsigned int i = 0; i < R_b; i++) {
            bool block_nonzero = false;
            for(unsigned int bj = 0; bj < b; bj++) {
                for(unsigned int bi = 0; bi < b; bi++) {
                    unsigned int row = i * b + bi;
                    unsigned int col = j * b + bj;
                    if(row < A.R and col < A.C and A.mat[row*A.C + col] != 0) {
                        block_nonzero = true;
                        break;
                    }
                }
                if(block_nonzero) 
                    break;
            }
            if(block_nonzero) {
                rowIdx.push_back(i);
                for(unsigned int bi = 0; bi < b; bi++) {
                    for (unsigned int bj = 0; bj < b; bj++) {
                        unsigned int row = i * b + bi;
                        unsigned int col = j * b + bj;
                        if(row < A.R and col < A.C) {
                            values.push_back(A.mat[row*A.C + col]);
                        } 
                        else {
                            values.push_back(0);
                        }
                    }
                }
                colPtrs[j+1]++;
            }
        }
    }

    // Convert colPtrs to cumulative sum
    for(unsigned int j = 1; j <= C_b; j++) {
        colPtrs[j] += colPtrs[j - 1];
    }
    
    // Create the BSC matrix
    unsigned int* bsc_col_ptrs = (unsigned int*)malloc(colPtrs.size() * sizeof(unsigned int));
    unsigned int* bsc_row_idx = (unsigned int*)malloc(rowIdx.size() * sizeof(unsigned int));
    T* bsc_value = (T*)malloc(values.size() * sizeof(T));

    std::copy(colPtrs.begin(), colPtrs.end(), bsc_col_ptrs);
    std::copy(rowIdx.begin(), rowIdx.end(), bsc_row_idx);
    std::copy(values.begin(), values.end(), bsc_value);

    BSCMatrix<T> bsc_matrix = {bsc_col_ptrs, bsc_row_idx, bsc_value, A.R, A.C, A.num_nonzero, (unsigned int)rowIdx.size(), (unsigned int)values.size(), b};
    return bsc_matrix;
}

template <typename T>
SparseMatrix<T> bsc_to_sparse(BSCMatrix<T> A) {
    unsigned int b = A.block_size;
    T* mat = (T*)malloc(A.R*A.C * sizeof(T));
    std::fill(mat, mat + A.R*A.C, static_cast<T>(0));

    for(unsigned int j = 0; j < (A.C + b - 1) / b; j++) {
        for(unsigned int i = A.colPtrs[j]; i < A.colPtrs[j+1]; i++) {
            unsigned int row_block = A.rowIdx[i];
            for(unsigned int bj = 0; bj < b; bj++) {
                for(unsigned int bi = 0; bi < b; bi++) {
                    unsigned int row = row_block * b + bi;
                    unsigned int col = j * b + bj;
                    if(row < A.R and col < A.C) {
                        mat[row*A.C + col] = A.value[i * b * b + bi * b + bj];
                    }
                }
            }
        }
    }

    SparseMatrix<T> sparse_matrix = {mat, A.R, A.C, A.num_nonzero};
    return sparse_matrix;
}


template int get_num_nonzero<float>(float*, unsigned int, unsigned int);
template SparseMatrix<float> generate_sparse_matrix<float>(float, unsigned int, unsigned int);
template COOMatrix<float> sparse_to_coo<float>(SparseMatrix<float>);
template SparseMatrix<float> coo_to_sparse<float>(COOMatrix<float>);
template CSRMatrix<float> sparse_to_csr<float>(SparseMatrix<float>);
template SparseMatrix<float> csr_to_sparse<float>(CSRMatrix<float>);
template CSCMatrix<float> sparse_to_csc<float>(SparseMatrix<float>);
template SparseMatrix<float> csc_to_sparse<float>(CSCMatrix<float>);
template ELLMatrix<float> sparse_to_ell<float>(SparseMatrix<float>);
template SparseMatrix<float> ell_to_sparse<float>(ELLMatrix<float>);
template BSRMatrix<float> sparse_to_bsr<float>(SparseMatrix<float>, unsigned int);
template SparseMatrix<float> bsr_to_sparse<float>(BSRMatrix<float>);
template BSCMatrix<float> sparse_to_bsc<float>(SparseMatrix<float>, unsigned int);
template SparseMatrix<float> bsc_to_sparse<float>(BSCMatrix<float>);
