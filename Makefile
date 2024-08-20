all: sparse_matrix_conversion coo_to_csr spmv

sparse_matrix_conversion:
	$(MAKE) -C src/sparse_matrix_conversion -f Makefile_sparse_matrix_conversion

coo_to_csr:
	$(MAKE) -C src/sparse_matrix_conversion -f Makefile_coo_to_csr

spmv:
	$(MAKE) -C src/spmv -f Makefile_spmv_coo
	$(MAKE) -C src/spmv -f Makefile_spmv_csr
	$(MAKE) -C src/spmv -f Makefile_spmv_csc
	$(MAKE) -C src/spmv -f Makefile_spmv_ell
	$(MAKE) -C src/spmv -f Makefile_spmv_bsr
	$(MAKE) -C src/spmv -f Makefile_spmv_bsc

spmv_coo:
	$(MAKE) -C src/spmv -f Makefile_spmv_coo

spmv_csr:
	$(MAKE) -C src/spmv -f Makefile_spmv_csr

spmv_csc:
	$(MAKE) -C src/spmv -f Makefile_spmv_csc

spmv_ell:
	$(MAKE) -C src/spmv -f Makefile_spmv_ell

spmv_bsr:
	$(MAKE) -C src/spmv -f Makefile_spmv_bsr

spmv_bsc:
	$(MAKE) -C src/spmv -f Makefile_spmv_bsc

clean:
	$(MAKE) -C src/sparse_matrix_conversion -f Makefile_sparse_matrix_conversion clean
	$(MAKE) -C src/sparse_matrix_conversion -f Makefile_coo_to_csr clean
	$(MAKE) -C src/spmv -f Makefile_spmv_coo clean
	$(MAKE) -C src/spmv -f Makefile_spmv_csr clean
	$(MAKE) -C src/spmv -f Makefile_spmv_csc clean
	$(MAKE) -C src/spmv -f Makefile_spmv_ell clean
	$(MAKE) -C src/spmv -f Makefile_spmv_bsr clean
	$(MAKE) -C src/spmv -f Makefile_spmv_bsc clean
