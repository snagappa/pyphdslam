#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       blas_tools_c_code.py
#       
#       Copyright 2012 Sharad Nagappa <snagappa@gmail.com>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#      

EXTRA_COMPILE_ARGS = ["-O3 -g -fopenmp"]
lopenblas = ["openblas"]
lblas = ["blas"]
lgsl = ["gsl"]
llapack = ["lapack"]
lptf77blas = ["ptf77blas"]+llapack
lgomp = ["gomp"]

omp_headers = """
#include <omp.h>
"""

omp_code = """
    int nthreads, tid;
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, vec_len, alpha, alpha_offset, x, inc, y, nthreads) private(i, tid)
    """
    
lapack_headers = """
#include <atlas/clapack.h>

extern "C" {
    int dpotri_(char *UPLO, int *N, double *A, int *LDA, int *INFO);
    int dtrtri_(char *UPLO, char *DIAG, int *N, double *A, int *LDA, int *INFO);
}

"""

gsl_blas_headers = """
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

extern "C" {
     int clapack_dtrtri ( const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_DIAG Diag, const int N, double *A, const int lda );

}
"""

gsl_la_headers = """
#include <gsl/gsl_linalg.h>

"""


f77_blas_headers = """
#include <atlas/cblas.h>
#include <atlas/clapack.h>

extern "C" {
    double ddot_(int *N, double *X, int *INCX, double *Y, int *INCY);
    double dnrm2_(int *N, double *X, int *INCX);
    double dasum_(int *N, double *X, int *INCX);
    int idamax_(int *N, double *X, int *INCX);
    int daxpy_(int *N, double *ALPHA, double *X, int *INCX, double *Y, int *INCY);
    int dscal_(int *N, double *ALPHA, double *X, int *INCX);
    int dcopy_(int *N, double *X, int *INCX, double *Y, int *INCY);
    int dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);
    //int dger_(int *M, )
}

inline void copylist1d(py::indexed_ref x, int ncols, double *x_copy) {
    int i;
    for (i=0; i<ncols; i++)
        x_copy[i] = x[i];
}

inline void copylist1d(double *x, int ncols, py::indexed_ref x_copy) {
    int i;
    for (i=0; i<ncols; i++)
        x_copy[i] = x[i];
}

inline void copylist2d(py::indexed_ref A, int nrows, int ncols, double *A_copy) {
    int i, j;
    for (i=0; i<nrows; i++)
        for (j=0; j<ncols; j++)
            A_copy[i*ncols + j] = A[i][j];
}

inline void copylist2d(double *A, int nrows, int ncols, py::indexed_ref A_copy) {
    int i, j;
    for (i=0; i<nrows; i++)
        for (j=0; j<ncols; j++)
            A_copy[i][j] = A[i*ncols + j];
}

"""


class lcopy:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["cblas", "lapack"]
    python_vars = ["x", "y", "itis"]
    code = """
    int i, j, k, ndims;
    py::object shape;
    ndims = (int)itis.attr("ndims");
    shape = itis.attr("shape");
    
    if (ndims == 1)
        for (i=0; i<(int)shape[0]; i++)
            y[i] = x[i];
    else if (ndims == 2)
        for (i=0; i<(int)shape[0]; i++)
            for (j=0; j<(int)shape[1]; j++)
                y[i][j] = x[i][j];
    else if (ndims == 3)
        for (i=0; i<(int)shape[0]; i++)
            for (j=0; j<(int)shape[1]; j++)
                for (k=0; k<(int)shape[2]; k++)
                    y[i][j][k] = x[i][j][k];
    """
    

###############################################################################
# ddot -- dot product of two vectors
class ddot:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["x", "y","xt_dot_y"]
    code = """
    int i, num_x, num_y, max_num;
    int vec_len, x_offset, y_offset, inc;
    //gsl_vector_view gsl_x, gsl_y;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    num_y = Ny[0];
    max_num = num_x>num_y?num_x:num_y;
    
    x_offset = num_x==1?0:vec_len;
    y_offset = num_y==1?0:vec_len;
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(max_num, xt_dot_y, vec_len, x, x_offset, inc, y, y_offset) private(i)
    for (i=0; i<max_num; i++) {
        //gsl_x = gsl_vector_view_array(x+(i*x_offset), vec_len);
        //gsl_y = gsl_vector_view_array(y+(i*y_offset), vec_len);
        //gsl_blas_ddot (&gsl_x.vector, &gsl_y.vector, xt_dot_y+i);
        xt_dot_y[i] = ddot_(&vec_len, x+(i*x_offset), &inc, y+(i*y_offset), &inc);
    }
    """


###############################################################################
# dnrm2 -- Euclidean norm of a vector
class dnrm2:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["x", "nrm2"]
    code = """
    int i, num_x;
    int vec_len, inc;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, nrm2, vec_len, x, inc) private(i)
    for (i=0; i<num_x;i++)
        nrm2[i] = dnrm2_(&vec_len, x+(i*vec_len), &inc);
    """

    
###############################################################################
# dasum -- sum of absolute values
class dasum:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["x", "asum"]
    code = """
    int i, num_x;
    int vec_len, inc;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, asum, vec_len, x, inc) private(i)
    for (i=0; i<num_x;i++)
        asum[i] = dasum_(&vec_len, x+(i*vec_len), &inc);
    """


# idamax -- find index of element having the max absolute value
class idamax:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["x", "max_idx"]
    code = """
    int i, num_x;
    int vec_len, inc;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, max_idx, vec_len, x, inc) private(i)
    for (i=0; i<num_x;i++)
        max_idx[i] = idamax_(&vec_len, x+(i*vec_len), &inc);
    """


# daxpy -- y = ax+y
class daxpy:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["alpha", "x", "y"]
    code = """
    int i, num_x, num_alpha, num_y;
    int vec_len, x_offset, alpha_offset, y_offset, inc;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    x_offset = num_x==1?0:Nx[1];
    num_alpha = Nalpha[0];
    alpha_offset = !(num_alpha==1);
    num_y = Ny[0];
    y_offset = Ny[1];
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_y, y_offset, alpha, alpha_offset, x, x_offset, inc, y) private(i)
    for (i=0; i<num_y;i++) {
        daxpy_(&y_offset, alpha+(i*alpha_offset), x+(i*x_offset), &inc, y+(i*y_offset), &inc);
    }
    
    """
    
        
# dscal -- x = ax
class dscal:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["alpha", "x"]
    code = """
    int i, num_x, num_alpha;
    int vec_len, x_offset, alpha_offset, inc;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    num_alpha = Nalpha[0];
    
    alpha_offset = !(num_alpha==1);
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, vec_len, alpha, alpha_offset, x, inc) private(i)
    for (i=0; i<num_x;i++)
        dscal_(&vec_len, alpha+(i*alpha_offset), x+(i*vec_len), &inc);
    """


class dcopy:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["x", "y"]
    code = """
    int nthreads, tid;
    int i, num_x, num_y;
    int x_offset, inc, y_offset;
    
    inc = 1;
    num_x = Nx[0];
    x_offset = num_x==1?0:Nx[1];
    num_y = Ny[0];
    y_offset = num_y==1?0:Ny[1];
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, x_offset, x, inc, y, y_offset) private(i)
    for (i=0; i<num_x;i++)
        dcopy_(&x_offset, x+(i*x_offset), &inc, y+(i*y_offset), &inc);
    """
    

class _dcopy_:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers+omp_headers
    libraries = lptf77blas+lgomp
    extra_compile_args = []
    python_vars = ["x", "y", "vec_len", "x_offset", "y_offset"]
    code = """
    dcopy_(&vec_len, x+x_offset, 1, y+y_offset, 1);
    """
    
    
    
###############################################################################
##
## LEVEL 2 BLAS Functions
##
###############################################################################

###############################################################################
# dgemv -- y = alpha*A*x + beta*y OR y = alpha*A**T*x + beta*y
class dgemv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+llapack+lgomp+lgsl
    extra_compile_args = []
    python_vars = ["A", "x", "y", "alpha", "beta", "TRANSPOSE_A"]
    code = """
    int i, num_A, num_x, num_alpha, num_beta, num_y;
    int nrows, ncols, A_offset, x_offset, x_vec_len, alpha_offset, beta_offset, y_offset, y_vec_len;
    CBLAS_TRANSPOSE_t TransA;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x, gsl_y;
    int nthreads, tid;
    
    num_A = NA[0];
    num_x = Nx[0];
    num_alpha = Nalpha[0];
    num_beta = Nbeta[0];
    num_y = Ny[0];
    
    nrows = NA[1];
    ncols = NA[2];
    x_vec_len = Nx[1];
    y_vec_len = Ny[1];
    
    A_offset = num_A==1?0:nrows*ncols;
    x_offset = num_x==1?0:Nx[1];
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    TransA = (int)TRANSPOSE_A?CblasTrans:CblasNoTrans;
    if (!(int)TRANSPOSE_A) {
        TransA = CblasNoTrans;
        y_offset = nrows;
    }
    else {
        TransA = CblasTrans;
        y_offset = ncols;
    }
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_y, A, A_offset, nrows, ncols, x, x_offset, x_vec_len, y, y_offset, y_vec_len, alpha, alpha_offset, beta, beta_offset) private(i, gsl_A, gsl_x, gsl_y)
    for (i=0; i<num_y; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, ncols);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), x_vec_len);
        gsl_y = gsl_vector_view_array(y+(i*y_offset), y_vec_len);
        gsl_blas_dgemv (TransA, alpha[i*alpha_offset], &gsl_A.matrix, &gsl_x.vector, 
                        beta[i*beta_offset], &gsl_y.vector);
    }
    """
    old_python_vars = ["A", "x", "y", "alpha", "beta", "TRANSPOSE_A", "C_CONTIGUOUS"]
    old_code = """
    int i, num_A, num_x, num_alpha, num_beta, max_num;
    int A_offset, x_offset, alpha_offset, beta_offset, y_offset, inc;
    int nrows, ncols, lda;
    char f_transA;
    int nthreads, tid;
    
    // Row major
    if ((int)C_CONTIGUOUS) {
        if (!(int)TRANSPOSE_A) {
            f_transA = 't';
            nrows = NA[2];
            ncols = NA[1];
            y_offset = NA[1];
        }
        else {
            f_transA = 'n';
            nrows = NA[2];
            ncols = NA[1];
            y_offset = NA[2];
        }
    }
    /*
    // Column major (Fortran)
    else {
        nrows = NA[1];
        ncols = NA[2];
        if (!(int)TRANSPOSE_A) {
            f_transA = 'n';
            y_offset = nrows;
        }
        else {
            f_transA = 't';
            y_offset = ncols;
        }
    }
    */
    
    inc = 1;
    num_A = NA[0];
    num_x = Nx[0];
    num_alpha = Nalpha[0];
    num_beta = Nbeta[0];
    
    
    
    A_offset = num_A==1?0:nrows*ncols;
    x_offset = num_x==1?0:Nx[1];
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    max_num = num_A>num_x?num_A:num_x;
    
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    //std::cout << "nrows=" << nrows <<" ncols="<< ncols << " A_offset=" << A_offset <<" x_offset=" << x_offset << " alpha_offset=" << alpha_offset << " beta_offset=" << beta_offset << " y_offset=" << y_offset << std::endl;
    
    #pragma omp parallel for \
        shared(max_num, f_transA, nrows, ncols, alpha, alpha_offset, A, A_offset, x, x_offset, inc, beta, beta_offset, y) private(i)
    for (i=0; i<max_num; i++)
        dgemv_(&f_transA, &nrows, &ncols, alpha+(i*alpha_offset), 
               A+(i*A_offset), &nrows, x+(i*x_offset), &inc, beta+(i*beta_offset), 
               y+(i*y_offset), &inc);
    
    """


###############################################################################
# dtrmv -- x := A*x,   or   x := A**T*x, A is nxn triangular
class dtrmv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "x", "UPLO", "TRANSPOSE_A"]
    code = """
    int i, num_A, num_x;
    int nrows, vec_len, x_offset, A_offset;
    CBLAS_UPLO_t Uplo;
    CBLAS_TRANSPOSE_t TransA;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x;
    int nthreads, tid;
    
    num_x = Nx[0];
    num_A = NA[0];
    nrows = NA[1];
    
    vec_len = Nx[1];
    x_offset = num_x==1?0:vec_len;
    A_offset = num_A==1?0:nrows*nrows;
    
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    TransA = (int)TRANSPOSE_A?CblasTrans:CblasNoTrans;
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, A, A_offset, nrows, x, x_offset, vec_len, Uplo, TransA) private(i, gsl_A, gsl_x)
    for (i=0; i<num_x; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, nrows);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), vec_len);
        gsl_blas_dtrmv (Uplo, TransA, CblasNonUnit, &gsl_A.matrix, &gsl_x.vector);
    }
    """
    
    
###############################################################################
# dtrsv -- Solve A*x = b,   or   A**T*x = b, A is nxn triangular
class dtrsv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "x", "UPLO", "TRANSPOSE_A"]
    code = """
    int i, num_A, num_x;
    int nrows, vec_len, x_offset, A_offset;
    CBLAS_UPLO_t Uplo;
    CBLAS_TRANSPOSE_t TransA;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x;
    int nthreads, tid;
    
    num_x = Nx[0];
    num_A = NA[0];
    nrows = NA[1];
    
    vec_len = Nx[1];
    x_offset = num_x==1?0:vec_len;
    A_offset = num_A==1?0:nrows*nrows;
    
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    TransA = (int)TRANSPOSE_A?CblasTrans:CblasNoTrans;
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_x, A, A_offset, nrows, x, x_offset, vec_len, Uplo, TransA) private(i, gsl_A, gsl_x)
    for (i=0; i<num_x; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, nrows);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), vec_len);
        gsl_blas_dtrsv (Uplo, TransA, CblasNonUnit, &gsl_A.matrix, &gsl_x.vector);
    }
    """
    

###############################################################################
# dsymv -- y = alpha*A*x + beta*y, A is symmetric
class dsymv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["alpha", "A", "x", "beta", "y", "UPLO"]
    code = """
    int i, num_alpha, num_A, num_x, num_beta, num_y;
    int nrows, vec_len, x_offset, A_offset, alpha_offset, beta_offset, y_offset;
    CBLAS_UPLO_t Uplo;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x;
    gsl_vector_view gsl_y;
    int nthreads, tid;
    
    num_x = Nx[0];
    num_A = NA[0];
    nrows = NA[1];
    num_alpha = Nalpha[0];
    num_beta = Nbeta[0];
    num_y = Ny[0];
    
    vec_len = Nx[1];
    x_offset = num_x==1?0:vec_len;
    A_offset = num_A==1?0:nrows*nrows;
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    y_offset = vec_len;
    
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_y, A, A_offset, nrows, x, x_offset, vec_len, Uplo, alpha, beta) private(i, gsl_A, gsl_x, gsl_y)
    for (i=0; i<num_y; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, nrows);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), vec_len);
        gsl_y = gsl_vector_view_array(y+(i*y_offset), vec_len);
        gsl_blas_dsymv (Uplo, alpha[i*alpha_offset], &gsl_A.matrix, 
                        &gsl_x.vector, beta[i*beta_offset], &gsl_y.vector);
    }
    """
    

###############################################################################
# dger -- rank 1 operation, A = alpha*x*y**T + A
class dger:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "x", "y", "alpha"]
    code = """
    // A is stored in row major order and there is no option to transpose so we 
    // use gsl_blas to perform this
    
    int i, num_x, num_y, num_alpha, num_A;
    int nrows, ncols, x_vec_len, x_offset, y_vec_len, y_offset, alpha_offset, A_offset, inc;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x, gsl_y;
    int nthreads, tid;
    
    inc = 1;
    num_x = Nx[0];
    num_y = Ny[0];
    num_alpha = Nalpha[0];
    num_A = NA[0];
    
    x_vec_len = Nx[1];
    y_vec_len = Ny[1];
    
    nrows = NA[1];
    ncols = NA[2];
    
    x_offset = num_x==1?0:x_vec_len;
    y_offset = num_y==1?0:y_vec_len;
    alpha_offset = !(num_alpha==1);
    A_offset = nrows*ncols;
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_A, A, A_offset, nrows, ncols, x, x_offset, x_vec_len, y, y_offset, y_vec_len, alpha, alpha_offset) private(i, gsl_A, gsl_x, gsl_y)
    for (i=0; i<num_A; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, ncols);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), x_vec_len);
        gsl_y = gsl_vector_view_array(y+(i*y_offset), y_vec_len);
        gsl_blas_dger (alpha[i*alpha_offset], &gsl_x.vector, &gsl_y.vector, &gsl_A.matrix);
    }
        //dger(&nrows, &ncols, alpha+(i*alpha_offset), x+(i*x_offset), &inc, 
        //     y+(i*y_offset), &inc, A+(i*A_offset), &nrows);
    """

###############################################################################
# dsyr -- symmetric rank 1 operation, A = alpha*x*x**T + A, A is nxn symmetric
class dsyr:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "x", "alpha", "UPLO"]
    code = """
    int i, num_x, num_A, num_alpha;
    int nrows, x_vec_len, x_offset, A_offset, alpha_offset;
    CBLAS_UPLO_t Uplo;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x;
    int nthreads, tid;
    
    num_x = Nx[0];
    num_A = NA[0];
    x_vec_len = Nx[1];
    nrows = NA[1];
    num_alpha = Nalpha[0];
    
    x_offset = num_x==1?0:x_vec_len;
    A_offset = nrows*nrows;
    alpha_offset = !(num_alpha==1);
    
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    
    #pragma omp parallel shared(nthreads) private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_A, A, A_offset, nrows, x, x_offset, x_vec_len, alpha, alpha_offset) private(i, gsl_A, gsl_x)
    for (i=0; i<num_A; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, nrows);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), x_vec_len);
        gsl_blas_dsyr (Uplo, alpha[i*alpha_offset], &gsl_x.vector, &gsl_A.matrix);
    }
    """

###############################################################################
# dsyr2 -- symmetric rank 2 operation, A = alpha*x*y**T + alpha*y*x**T + A


###############################################################################
##
## Level 3 BLAS
##
###############################################################################

###############################################################################
# dgemm -- matrix-matrix operation, C := alpha*op( A )*op( B ) + beta*C,
#           where  op( X ) is one of
#           op( X ) = X   or   op( X ) = X**T
class dgemm:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "B", "C", "alpha", "beta", "TRANSPOSE_A", "TRANSPOSE_B"]
    code = """
    int i, num_A, num_B, num_C, num_alpha, num_beta;
    int A_rows, A_cols, B_rows, B_cols, C_rows, C_cols;
    int A_offset, B_offset, C_offset, alpha_offset, beta_offset;
    CBLAS_TRANSPOSE_t TransA, TransB;
    gsl_matrix_view gsl_A, gsl_B, gsl_C;
    int nthreads, tid;
    
    num_A = NA[0]; A_rows = NA[1]; A_cols = NA[2];
    num_B = NB[0]; B_rows = NB[1]; B_cols = NB[2];
    num_C = NC[0]; C_rows = NC[1]; C_cols = NC[2];
    num_alpha = Nalpha[0];
    num_beta = Nbeta[0];
    
    A_offset = num_A==1?0:A_rows*A_cols;
    B_offset = num_B==1?0:B_rows*B_cols;
    C_offset = num_C==1?0:C_rows*C_cols;
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    
    TransA = (int)TRANSPOSE_A?CblasTrans:CblasNoTrans;
    TransB = (int)TRANSPOSE_B?CblasTrans:CblasNoTrans;
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_C, A, A_offset, A_rows, A_cols, B, B_offset, B_rows, B_cols, C, C_offset, C_rows, C_cols, TransA, TransB, alpha, alpha_offset, beta, beta_offset) private(i, gsl_A, gsl_B, gsl_C)
    for (i=0; i<num_C; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_cols);
        gsl_B = gsl_matrix_view_array(B+(i*B_offset), B_rows, B_cols);
        gsl_C = gsl_matrix_view_array(C+(i*C_offset), C_rows, C_cols);
        gsl_blas_dgemm (TransA, TransB, alpha[i*alpha_offset], &gsl_A.matrix, 
                        &gsl_B.matrix, beta[i*beta_offset], &gsl_C.matrix);
    }
    """

###############################################################################
# dsymm -- matrix-matrix operation, A is symmetric, B and C are mxn
#           C := alpha*A*B + beta*C, or
#           C := alpha*B*A + beta*C
class dsymm:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "B", "C", "alpha", "beta", "SIDE", "UPLO"]
    code = """
    int i, num_A, num_B, num_C, num_alpha, num_beta;
    int A_rows, B_rows, B_cols, C_rows, C_cols;
    int A_offset, B_offset, C_offset, alpha_offset, beta_offset;
    CBLAS_SIDE_t Side;
    CBLAS_UPLO_t Uplo;
    gsl_matrix_view gsl_A, gsl_B, gsl_C;
    int nthreads, tid;
    
    num_A = NA[0]; A_rows = NA[1];
    num_B = NB[0]; B_rows = NB[1]; B_cols = NB[2];
    num_C = NC[0]; C_rows = NC[1]; C_cols = NC[2];
    num_alpha = Nalpha[0];
    num_beta = Nbeta[0];
    
    A_offset = num_A==1?0:A_rows*A_rows;
    B_offset = num_B==1?0:B_rows*B_cols;
    C_offset = num_C==1?0:C_rows*C_cols;
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    
    Side = std::tolower(SIDE[0])=='l'?CblasLeft:CblasRight;
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_C, A, A_offset, A_rows, B, B_offset, B_rows, B_cols, C, C_offset, C_rows, C_cols, alpha, alpha_offset, beta, beta_offset) private(i, gsl_A, gsl_B, gsl_C)
    for (i=0; i<num_C; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_rows);
        gsl_B = gsl_matrix_view_array(B+(i*B_offset), B_rows, B_cols);
        gsl_C = gsl_matrix_view_array(C+(i*C_offset), C_rows, C_cols);
        gsl_blas_dsymm (Side, Uplo, alpha[i*alpha_offset], &gsl_A.matrix, 
                        &gsl_B.matrix, beta[i*beta_offset], &gsl_C.matrix);
    }
    """


###############################################################################
# dsyrk -- symmetric rank k operation, C is symmetric
#           C := alpha*A*A**T + beta*C, or
#           C := alpha*A**T*A + beta*C
class dsyrk:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "alpha", "C", "beta", "TRANSPOSE_A", "UPLO"]
    code = """
    int i, num_A, num_C, num_alpha, num_beta;
    int A_rows, A_cols, C_rows;
    int A_offset, C_offset, alpha_offset, beta_offset;
    CBLAS_TRANSPOSE_t TransA;
    CBLAS_UPLO_t Uplo;
    gsl_matrix_view gsl_A, gsl_C;
    int nthreads, tid;
    
    num_A = NA[0]; A_rows = NA[1]; A_cols = NA[2];
    num_C = NC[0]; C_rows = NC[1];
    num_alpha = Nalpha[0];
    num_beta = Nbeta[0];
    
    A_offset = num_A==1?0:A_rows*A_cols;
    C_offset = num_C==1?0:C_rows*C_rows;
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    
    TransA = (int)TRANSPOSE_A?CblasTrans:CblasNoTrans;
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_C, A, A_offset, A_rows, C, C_offset, C_rows, alpha, alpha_offset, beta, beta_offset) private(i, gsl_A, gsl_C)
    for (i=0; i<num_C; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_cols);
        gsl_C = gsl_matrix_view_array(C+(i*C_offset), C_rows, C_rows);
        gsl_blas_dsyrk (Uplo, TransA, alpha[i*alpha_offset], &gsl_A.matrix, 
                        beta[i*beta_offset], &gsl_C.matrix);
    }
    """



###############################################################################
##
## Linear Algebra
##
###############################################################################

###############################################################################
class dgetrf:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A", "ipiv", "signum"]
    code = """
    int i, j, num_A;
    int A_rows, A_cols, A_offset;
    gsl_matrix_view gsl_A;
    gsl_permutation * p;
    int s, COPY_IPIV;
    size_t *p_data;
    if (sizeof(size_t) == sizeof(int))
        COPY_IPIV = 0;
    
    num_A = NA[0];
    A_rows = NA[1]; A_cols = NA[2];
    A_offset = num_A==1?0:A_rows*A_cols;
    
    #pragma omp parallel shared(A_rows, num_A, A, A_offset, COPY_IPIV, ipiv, signum) private(p, p_data, i, gsl_A, s, j)
    {
    p = gsl_permutation_alloc (A_rows);
    p_data = p->data;
    
    #pragma omp for
    for (i=0; i<num_A; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_rows);
        if (!COPY_IPIV)
            p->data = (size_t*)ipiv+(i*A_rows);
        gsl_linalg_LU_decomp (&gsl_A.matrix, p, &s);
        signum[i] = s;
        // memcpy p to ipiv
        if (COPY_IPIV)
            //memcpy(ipiv+(i*A_rows), p->data, A_rows*sizeof(size_t));
            for (j=0; j<A_rows; j++)
                ipiv[i*A_rows+j] = p->data[j];
    }
    p->data = p_data;
    gsl_permutation_free(p);
    }
    """


###############################################################################
class dgetrs:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["LU", "ipiv", "b", "x"]
    code = """
    int i, j, num_LU, num_b, num_x;
    int LU_rows, LU_cols, LU_offset, b_offset, x_offset, ipiv_offset;
    gsl_matrix_view gsl_LU;
    gsl_vector_view gsl_x, gsl_b;
    gsl_permutation * p;
    int COPY_IPIV;
    size_t *p_data;
    if (sizeof(size_t) == sizeof(int))
        COPY_IPIV = 0;
    
    num_LU = NLU[0];
    LU_rows = NLU[1];
    LU_offset = num_LU==1?0:LU_rows*LU_rows;
    ipiv_offset = num_LU==1?0:LU_rows;
    
    num_b = Nb[0];
    b_offset = num_b==1?0:LU_rows;
    num_x = Nx[0];
    x_offset = LU_rows;
    
    #pragma omp parallel shared(LU_rows, num_x, LU, LU_offset, b, b_offset, x, x_offset, COPY_IPIV, ipiv, ipiv_offset) private(p, p_data, i, gsl_LU, gsl_b, gsl_x, j)
    {
    p = gsl_permutation_alloc (LU_rows);
    p_data = p->data;
    
    #pragma omp for
    for (i=0; i<num_x; i++) {
        gsl_LU = gsl_matrix_view_array(LU+(i*LU_offset), LU_rows, LU_rows);
        gsl_b = gsl_vector_view_array(b+(i*b_offset), LU_rows);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), LU_rows);
        if (!COPY_IPIV)
            p->data = (size_t*)(ipiv+(i*ipiv_offset));
        else
            for (j=0; j<LU_rows; j++)
                p->data[j] = ipiv[i*ipiv_offset+j];
        gsl_linalg_LU_solve (&gsl_LU.matrix, p, &gsl_b.vector, &gsl_x.vector);
        //for (int j=0; j<LU_rows; j++)
        //    std::cout << "p["<<j<<"]=" << p->data[j] << "  ipiv["<<j<<"]="<<ipiv[i*ipiv_offset+j] << std::endl;
    }
    p->data = p_data;
    gsl_permutation_free(p);
    }
    """
    
    
###############################################################################
class dgetrsx:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["LU", "ipiv", "b"]
    code = """
    int i, j, num_LU, num_x;
    int LU_rows, LU_cols, LU_offset, x_offset, ipiv_offset;
    gsl_matrix_view gsl_LU;
    gsl_vector_view gsl_x;
    gsl_permutation * p;
    int COPY_IPIV;
    size_t *p_data;
    if (sizeof(size_t) == sizeof(int))
        COPY_IPIV = 0;
    
    num_LU = NLU[0];
    LU_rows = NLU[1];
    LU_offset = num_LU==1?0:LU_rows*LU_rows;
    ipiv_offset = num_LU==1?0:LU_rows;
    num_x = Nb[0];
    x_offset = LU_rows;
    
    #pragma omp parallel shared(LU_rows, num_x, LU, LU_offset, b, x_offset, COPY_IPIV, ipiv, ipiv_offset) private(p, p_data, i, gsl_LU, gsl_x, j)
    {
    p = gsl_permutation_alloc (LU_rows);
    p_data = p->data;
    
    #pragma omp for
    for (i=0; i<num_x; i++) {
        gsl_LU = gsl_matrix_view_array(LU+(i*LU_offset), LU_rows, LU_rows);
        gsl_x = gsl_vector_view_array(b+(i*x_offset), LU_rows);
        if (!COPY_IPIV)
            p->data = (size_t*)(ipiv+(i*ipiv_offset));
        else
            for (j=0; j<LU_rows; j++)
                p->data[j] = ipiv[i*LU_rows+j];
        gsl_linalg_LU_svx (&gsl_LU.matrix, p, &gsl_x.vector);
        //for (int j=0; j<LU_rows; j++)
        //    std::cout << "p["<<j<<"]=" << p->data[j] << "  ipiv["<<j<<"]="<<ipiv[i*LU_offset+j] << std::endl;
    }
    p->data = p_data;
    gsl_permutation_free(p);
    }
    """
    
    
###############################################################################
class dgetri:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["LU", "ipiv", "invA"]
    code = """
    int i, j, num_LU;
    int LU_rows, LU_cols, LU_offset, ipiv_offset;
    gsl_matrix_view gsl_LU, gsl_invA;
    gsl_permutation * p;
    int COPY_IPIV;
    size_t *p_data;
    if (sizeof(size_t) == sizeof(int))
        COPY_IPIV = 0;
    
    num_LU = NLU[0];
    LU_rows = NLU[1];
    LU_offset = LU_rows*LU_rows;
    ipiv_offset = LU_rows;
    
    #pragma omp parallel shared(LU_rows, LU, LU_offset, COPY_IPIV, ipiv, ipiv_offset) private(p, p_data, i, gsl_LU, j)
    {
    p = gsl_permutation_alloc (LU_rows);
    p_data = p->data;
    
    #pragma omp for
    for (i=0; i<num_LU; i++) {
        gsl_LU = gsl_matrix_view_array(LU+(i*LU_offset), LU_rows, LU_rows);
        gsl_invA = gsl_matrix_view_array(invA+(i*LU_offset), LU_rows, LU_rows);
        if (!COPY_IPIV)
            p->data = (size_t*)(ipiv+(i*ipiv_offset));
        else
            for (j=0; j<LU_rows; j++)
                p->data[j] = ipiv[i*LU_rows+j];
        gsl_linalg_LU_invert (&gsl_LU.matrix, p, &gsl_invA.matrix);
        //for (int j=0; j<LU_rows; j++)
        //    std::cout << "p["<<j<<"]=" << p->data[j] << "  ipiv["<<j<<"]="<<ipiv[i*LU_offset+j] << std::endl;
    }
    p->data = p_data;
    gsl_permutation_free(p);
    }
    """
    
    
###############################################################################
class dgetrdet:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["LU", "signum", "det_vec"]
    code = """
    int i, j, num_LU;
    int LU_rows, LU_cols, LU_offset, s;
    gsl_matrix_view gsl_LU;
    
    num_LU = NLU[0];
    LU_rows = NLU[1];
    LU_offset = LU_rows*LU_rows;
    
    #pragma omp parallel for \
        shared(LU, LU_offset, LU_rows, signum) private(i, gsl_LU, s)
    for (i=0; i<num_LU; i++) {
        gsl_LU = gsl_matrix_view_array(LU+(i*LU_offset), LU_rows, LU_rows);
        s = signum[i];
        gsl_linalg_LU_det (&gsl_LU.matrix, s);
    }
    """



###############################################################################
class dpotrf:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A"]
    code = """
    int i, num_A;
    int A_rows, A_cols, A_offset;
    gsl_matrix_view gsl_A;
    gsl_error_handler_t *gsl_default_error_handler;
    gsl_default_error_handler = gsl_set_error_handler_off();
    int gsl_error_value;
    int error_occurred = 0;
    
    num_A = NA[0];
    A_rows = NA[1]; A_cols = NA[2];
    A_offset = A_rows*A_cols;
    
    #pragma omp parallel for shared(num_A, error_occurred, A, A_offset, A_rows) private(i, gsl_A, gsl_error_value)
    for (i=0; i<num_A; i++) {
        if (!error_occurred) {
            gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_rows);
            gsl_error_value = gsl_linalg_cholesky_decomp (&gsl_A.matrix);
            if (gsl_error_value == GSL_EDOM) {
                error_occurred = 1;
                #pragma omp flush (error_occurred)
            }
        }
    }
    exception_occurred = error_occurred;
    gsl_set_error_handler(gsl_default_error_handler);
    """


###############################################################################
class dpotrs:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["cholA", "b", "x"]
    code = """
    int i, num_A, num_b, num_x;
    int A_rows, A_cols, A_offset, b_offset, x_offset;
    gsl_matrix_view gsl_cholA;
    gsl_vector_view gsl_x, gsl_b;
    gsl_error_handler_t *gsl_default_error_handler;
    gsl_default_error_handler = gsl_set_error_handler_off();
    int gsl_error_value;
    int error_occurred = 0;
    
    num_A = NcholA[0];
    A_rows = NcholA[1];
    A_offset = num_A==1?0:A_rows*A_rows;
    
    num_b = Nb[0];
    b_offset = num_b==1?0:A_rows;
    num_x = Nx[0];
    x_offset = A_rows;
    
    #pragma omp parallel for shared(num_x, error_occurred, cholA, A_offset, A_rows, b, b_offset, x, x_offset) private(i, gsl_cholA, gsl_b, gsl_x, gsl_error_value)
    for (i=0; i<num_x; i++) {
        if (!error_occurred) {
            gsl_cholA = gsl_matrix_view_array(cholA+(i*A_offset), A_rows, A_rows);
            gsl_b = gsl_vector_view_array(b+(i*b_offset), A_rows);
            gsl_x = gsl_vector_view_array(x+(i*x_offset), A_rows);
            gsl_error_value = gsl_linalg_cholesky_solve (&gsl_cholA.matrix, &gsl_b.vector, &gsl_x.vector);
            if (gsl_error_value == GSL_EDOM) {
                error_occurred = 1;
                #pragma omp flush (error_occurred)
            }
        }
    }
    exception_occurred = error_occurred;
    gsl_set_error_handler(gsl_default_error_handler);
    """
    
    
###############################################################################
class dpotrsx:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["cholA", "b"]
    code = """
    int i, num_A, num_x;
    int A_rows, A_cols, A_offset, x_offset;
    gsl_matrix_view gsl_cholA;
    gsl_vector_view gsl_x;
    gsl_error_handler_t *gsl_default_error_handler;
    gsl_default_error_handler = gsl_set_error_handler_off();
    int gsl_error_value;
    int error_occurred = 0;
    
    num_A = NcholA[0];
    A_rows = NcholA[1];
    A_offset = num_A==1?0:A_rows*A_rows;
    
    num_x = Nb[0];
    x_offset = A_rows;
    
    #pragma omp parallel for shared(num_x, error_occurred, cholA, A_offset, A_rows, b, x_offset) private(i, gsl_cholA, gsl_x, gsl_error_value)
    for (i=0; i<num_x; i++) {
        if (!error_occurred) {
            gsl_cholA = gsl_matrix_view_array(cholA+(i*A_offset), A_rows, A_rows);
            gsl_x = gsl_vector_view_array(b+(i*x_offset), A_rows);
            gsl_error_value = gsl_linalg_cholesky_svx (&gsl_cholA.matrix, &gsl_x.vector);
            if (gsl_error_value == GSL_EDOM) {
                error_occurred = 1;
                #pragma omp flush (error_occurred)
            }
        }
    }
    exception_occurred = error_occurred;
    gsl_set_error_handler(gsl_default_error_handler);
    """
    
    
###############################################################################
class dpotri:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_la_headers+omp_headers
    libraries = lptf77blas+lgomp+llapack+lgsl
    extra_compile_args = []
    python_vars = ["A"]
    code = """
    int i, j, num_A;
    int A_rows, A_cols, A_offset;
    gsl_matrix_view gsl_A;
    gsl_error_handler_t *gsl_default_error_handler;
    gsl_default_error_handler = gsl_set_error_handler_off();
    int gsl_error_value;
    int error_occurred = 0;
    
    num_A = NA[0];
    A_rows = NA[1];
    A_offset = A_rows*A_rows;
    
    #pragma omp parallel for shared(num_A, error_occurred, A, A_offset, A_rows) private(i, gsl_A, gsl_error_value)
    for (i=0; i<num_A; i++) {
        if (!error_occurred) {
            gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_rows);
            gsl_error_value = gsl_linalg_cholesky_invert (&gsl_A.matrix);
            if (gsl_error_value == GSL_EDOM) {
                error_occurred = 1;
                #pragma omp flush (error_occurred)
            }
        }
    }
    exception_occurred = error_occurred;
    gsl_set_error_handler(gsl_default_error_handler);
    """
    
    
###############################################################################
class dtrtri:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = omp_headers+gsl_blas_headers
    libraries = lptf77blas+lgomp+llapack+lgsl + ["cblas", "lapack_atlas"] 
    extra_compile_args = []
    python_vars = ["A"]
    code = """
    int i, j, num_A;
    int A_rows, A_cols, A_offset;
    char uplo, diag;
    int info;
    uplo = 'l';
    diag = 'n';
    gsl_error_handler_t *gsl_default_error_handler;
    gsl_default_error_handler = gsl_set_error_handler_off();
    int gsl_error_value;
    int error_occurred = 0;
    
    num_A = NA[0];
    A_rows = NA[1];
    A_offset = A_rows*A_rows;
    
    #pragma omp parallel for shared(num_A, error_occurred, uplo, diag, A_rows, A, A_offset) private(i, info, gsl_error_value)
    for (i=0; i<num_A; i++) {
        if (!error_occurred) {
            //dtrtri_(&uplo, &diag, &A_rows, A+(i*A_offset), &A_rows, &info);
            gsl_error_value = clapack_dtrtri(CblasRowMajor, CblasLower, CblasNonUnit, A_rows, A+(i*A_offset), A_rows);
            if (gsl_error_value == GSL_EDOM) {
                error_occurred = 1;
                #pragma omp flush (error_occurred)
            }
        }
    }
    exception_occurred = error_occurred;
    gsl_set_error_handler(gsl_default_error_handler);
    """

###############################################################################
class symmetrise:
    def __call__(self):
        return python_vars, code, support_code, libs
    helper_code = """
    inline void symmetrise_upper(int M, int N, double *A) {
        int i, j;
        for (i=1; i<M; i++)
            for (j=0; j<i; j++)
                A[i*N+j] = A[j*N+i];
    }
    
    inline void symmetrise_lower(int M, int N, double *A) {
        int i, j;
        for (i=0; i<M-1; i++)
            for (j=i+1; j<N; j++)
                A[i*N+j] = A[j*N+i];
    }
    """
    support_code = omp_headers+helper_code+gsl_blas_headers
    libraries = lgomp
    extra_compile_args = []
    python_vars = ["A", "UPLO"]
    code = """
    int i, num_A, tid, nthreads;
    int nrows, ncols, A_offset;
    CBLAS_UPLO_t Uplo;
    
    num_A = NA[0];
    nrows = NA[1];
    ncols = NA[2];
    A_offset = nrows*ncols;
    
    Uplo = std::tolower(UPLO[0])=='l'?CblasLower:CblasUpper;
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_A, Uplo, nrows, ncols, A, A_offset) private(i)
    for (i=0; i<num_A; i++) {
        if (Uplo==CblasLower)
            symmetrise_lower(nrows, ncols, A+(i*A_offset));
        else //if (Uplo==CblasUpper)
            symmetrise_upper(nrows, ncols, A+(i*A_offset));
    }
    """

###############################################################################
class mktril:
    def __call__(self):
        return python_vars, code, support_code, libs
    helper_code = """
    inline void mktril(int M, int N, double *A) {
        int i, j;
        for (i=0; i<M; i++)
            for (j=i+1; j<N; j++)
                A[i*N+j] = 0;
    }
    """
    support_code = omp_headers+helper_code
    libraries = lgomp
    extra_compile_args = []
    python_vars = ["A"]
    code = """
    int i, num_A, tid, nthreads;
    int nrows, ncols, A_offset;
    
    num_A = NA[0];
    nrows = NA[1];
    ncols = NA[2];
    A_offset = nrows*ncols;
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_A, nrows, ncols, A, A_offset) private(i)
    for (i=0; i<num_A; i++) {
        mktril(nrows, ncols, A+(i*A_offset));
    }
    """
    
###############################################################################
class mktriu:
    def __call__(self):
        return python_vars, code, support_code, libs
    helper_code = """
    inline void mktriu(int M, int N, double *A) {
        int i, j;
        for (i=1; i<M; i++)
            for (j=0; j<i; j++)
                A[i*N+j] = 0;
    }
    """
    support_code = omp_headers+helper_code
    libraries = lgomp
    extra_compile_args = []
    python_vars = ["A"]
    code = """
    int i, num_A, tid, nthreads;
    int nrows, ncols, A_offset;
    
    num_A = NA[0];
    nrows = NA[1];
    ncols = NA[2];
    A_offset = nrows*ncols;
    
    #pragma omp parallel private(i, tid)
    {
    tid = omp_get_thread_num();
    if (tid==0) {
        nthreads = omp_get_num_threads();
        // std::cout << "Using " << nthreads << " OMP threads" << std::endl;
    }
    }
    
    #pragma omp parallel for \
        shared(num_A, nrows, ncols, A, A_offset) private(i)
    for (i=0; i<num_A; i++) {
        mktriu(nrows, ncols, A+(i*A_offset));
    }
    """
###############################################################################
###############################################################################
class ldgemv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["blas", "atlas", "lapack"]
    extra_compile_args = []
    python_vars = ["A", "x", "y", "A_shape", "x_shape", "alpha", "beta", "TRANSPOSE_A"]
    code = """
    int i, j, k, inc;
    int num_A, num_x;
    int nrows, ncols;
    double *A_copy;
    double *x_copy;
    double *y_copy;
    
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA;
    char f_transA;
    
    // Row major is always true, so transpose should be inverted
    order = CblasRowMajor;
    transA = (int)TRANSPOSE_A>0? CblasTrans : CblasNoTrans;
    // if Fortran ordering
    //f_transA = (int)TRANSPOSE_A>0? 't' : 'n';
    f_transA = (int)TRANSPOSE_A>0? 'n' : 't';
    
    inc = 1;
    num_A = (int)A_shape[0];
    num_x = (int)x_shape[0];
    
    nrows = (int)A_shape[1];
    ncols = (int)A_shape[2];
    
    A_copy = (double*)malloc(sizeof(double)*nrows*ncols);
    x_copy = (double*)malloc(sizeof(double)*ncols);
    y_copy = (double*)malloc(sizeof(double)*ncols);
    
    if (num_A == num_x) {
        for (i=0; i<num_A; i++){
            copylist2d(A[i], nrows, ncols, A_copy);
            copylist1d(x[i], ncols, x_copy);
            copylist1d(y[i], ncols, y_copy);
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A_copy, nrows, x_copy, 1, beta, y_copy, 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A_copy, &nrows, x_copy, &inc, &beta, y_copy, &inc);
            copylist1d(y_copy, ncols, y[i]);
        }
    }
    else if (num_A == 1) {
        copylist2d(A[0], nrows, ncols, A_copy);
        for (i=0; i<num_x; i++){
            copylist1d(x[i], ncols, x_copy);
            copylist1d(y[i], ncols, y_copy);
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A_copy, nrows, x_copy, 1, beta, y_copy, 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A_copy, &nrows, x_copy, &inc, &beta, y_copy, &inc);
            copylist1d(y_copy, ncols, y[i]);
        }
    }
    else if (num_x == 1) {
        copylist1d(x[0], ncols, x_copy);
        for (i=0; i<num_A; i++) {
            copylist2d(A[i], nrows, ncols, A_copy);
            copylist1d(y[i], ncols, y_copy);
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A_copy, nrows, x_copy, 1, beta, y_copy, 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A_copy, &nrows, x_copy, &inc, &beta, y_copy, &inc);
            copylist1d(y_copy, ncols, y[i]);
        }
    }
    else {
        exception_occurred = 1;
    }
    free(A_copy);
    free(x_copy);
    free(y_copy);
    """
    
class npdgemv_old_plus_broken_code_now:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["blas", "lapack"]
    extra_compile_args = []
    python_vars = ["A", "x", "y", "alpha", "beta", "TRANSPOSE_A", "C_CONTIGUOUS"]
    code = """
    int i, j, k, base_offset, inc;
    int num_A, num_x, num_alpha, A_offset, x_offset, alpha_offset, max_num;
    int nrows, ncols;
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA;
    char f_transA;
    
    order = CblasRowMajor;
    transA = (int)TRANSPOSE_A>0? CblasTrans : CblasNoTrans;
    
    // Row major
    if ((int)C_CONTIGUOUS)
        f_transA = (int)TRANSPOSE_A>0? 'n' : 't';
    // Column major (Fortran)
    else
        f_transA = (int)TRANSPOSE_A>0? 't' : 'n';
    
    inc = 1;
    num_A = NA[0];
    num_x = Nx[0];
    num_alpha = Nalpha[0];
    
    nrows = NA[1];
    ncols = NA[2];
    
    A_offset = num_A==1?0:nrows*ncols;
    x_offset = num_x==1?0:ncols;
    alpha_offset = !(num_alpha==1);
    
    max_num = num_A>num_x:num_A:num_x;
    for (i=0; i<max_num; i++)
        //cblas_dgemv(order, transA, nrows, ncols, alpha, A+(i*A_offset), nrows, x+(i*ncols), 1, beta, y+(i*ncols), 1);
        dgemv_(&f_transA, &nrows, &ncols, &alpha, A+(i*A_offset), &nrows, x+(i*ncols), &inc, &beta, y+(i*ncols), &inc);
    if (num_A == num_x) {
        for (i=0; i<num_A; i++)
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A+(i*A_offset), nrows, x+(i*ncols), 1, beta, y+(i*ncols), 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A+(i*A_offset), &nrows, x+(i*ncols), &inc, &beta, y+(i*ncols), &inc);
    }
    else if (num_A == 1) {
        for (i=0; i<num_x; i++)
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A, nrows, x+(i*ncols), 1, beta, y+(i*ncols), 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A, &nrows, x+(i*ncols), &inc, &beta, y+(i*ncols), &inc);
    }
    else if (num_x == 1) {
        for (i=0; i<num_A; i++)
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A+(i*A_offset), nrows, x, 1, beta, y+(i*ncols), 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A+(i*A_offset), &nrows, x, &inc, &beta, y+(i*ncols), &inc);
    }
    else {
        exception_occurred = 1;
    }
    """
