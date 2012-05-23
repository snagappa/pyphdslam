# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:05:54 2012

@author: snagappa
"""

EXTRA_COMPILE_ARGS = ["-O3"]

gsl_blas_headers = """
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

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
    support_code = global_support_code
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
    support_code = f77_blas_headers
    libraries = ["blas", "lapack", "gsl"]
    python_vars = ["x", "y","xt_dot_y"]
    code = """
    int i, num_x, num_y, max_num;
    int vec_len, x_offset, y_offset, inc;
    //gsl_vector_view gsl_x, gsl_y;
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    num_y = Ny[0];
    max_num = num_x>num_y?num_x:num_y;
    
    x_offset = num_x==1?0:vec_len;
    y_offset = num_y==1?0:vec_len;
    
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
    support_code = global_support_code
    libraries = ["blas", "lapack"]
    python_vars = ["x", "nrm2"]
    code = """
    int i, num_x;
    int vec_len, inc;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    
    for (i=0; i<num_x;i++)
        nrm2[i] = dnrm2_(&vec_len, x+(i*vec_len), &inc);
    """

    
###############################################################################
# dasum -- sum of absolute values
class dasum:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["blas", "lapack"]
    python_vars = ["x", "asum"]
    code = """
    int i, num_x;
    int vec_len, inc;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    
    for (i=0; i<num_x;i++)
        asum[i] = dasum_(&vec_len, x+(i*vec_len), &inc);
    """


# idamax -- find index of element having the max absolute value
class idamax:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["blas", "lapack"]
    python_vars = ["x", "max_idx"]
    code = """
    int i, num_x;
    int vec_len, inc;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    
    for (i=0; i<num_x;i++)
        max_idx[i] = idamax_(&vec_len, x+(i*vec_len), &inc);
    """


# daxpy -- y = ax+y
class daxpy:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["blas", "lapack"]
    python_vars = ["alpha", "x", "y"]
    code = """
    int i, num_x, num_alpha;
    int vec_len, x_offset, alpha_offset, inc;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    num_alpha = Nalpha[0];
    alpha_offset = !(num_alpha==1);
        
    for (i=0; i<num_x;i++)
        daxpy_(&vec_len, alpha+(i*alpha_offset), x+(i*vec_len), &inc, y+(i*vec_len), &inc);
    """
    

# dscal -- x = ax
class dscal:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = f77_blas_headers
    libraries = ["blas", "lapack"]
    python_vars = ["alpha", "x"]
    code = """
    int i, num_x, num_alpha;
    int vec_len, x_offset, alpha_offset, inc;
    
    inc = 1;
    num_x = Nx[0];
    vec_len = Nx[1];
    num_alpha = Nalpha[0];
    
    alpha_offset = !(num_alpha==1);
    for (i=0; i<num_x;i++)
        dscal_(&vec_len, alpha+(i*alpha_offset), x+(i*vec_len), &inc);
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
    support_code = f77_blas_headers
    libraries = ["blas", "lapack"]
    python_vars = ["A", "x", "y", "alpha", "beta", "TRANSPOSE_A", "C_CONTIGUOUS"]
    code = """
    int i, num_A, num_x, num_alpha, num_beta, max_num;
    int A_offset, x_offset, alpha_offset, beta_offset, inc;
    int nrows, ncols;
    char f_transA;
    
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
    num_beta = Nbeta[0];
    
    nrows = NA[1];
    ncols = NA[2];
    
    A_offset = num_A==1?0:nrows*ncols;
    x_offset = num_x==1?0:ncols;
    alpha_offset = !(num_alpha==1);
    beta_offset = !(num_beta==1);
    
    max_num = num_A>num_x?num_A:num_x;
    for (i=0; i<max_num; i++)
        dgemv_(&f_transA, &nrows, &ncols, alpha+(i*alpha_offset), 
               A+(i*A_offset), &nrows, x+(i*x_offset), &inc, beta+(i*beta_offset), 
               y+(i*ncols), &inc);
    """


###############################################################################
# dsymv -- y = alpha*A*x + beta*y, A is symmetric


###############################################################################
# dger -- rank 1 operation, A = alpha*x*y**T + A
class dger:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = gsl_blas_headers
    libraries = ["blas", "lapack", "gsl"]
    python_vars = ["A", "x", "y", "alpha"]
    code = """
    // A is stored in row major order and there is no option to transpose so we 
    // use gsl_blas to perform this
    
    int i, num_x, num_y, num_alpha, num_A;
    int nrows, ncols, x_vec_len, x_offset, y_vec_len, y_offset, alpha_offset, A_offset, inc;
    gsl_matrix_view gsl_A;
    gsl_vector_view gsl_x;
    gsl_vector_view gsl_y;
    
    inc = 1;
    num_x = Nx[0];
    num_y = Ny[0];
    num_alpha = Nalpha[0];
    
    x_vec_len = Nx[1];
    y_vec_len = Ny[1];
    
    nrows = NA[1];
    ncols = NA[2];
    
    x_offset = num_x==1?0:x_vec_len;
    y_offset = num_y==1?0:y_vec_len;
    alpha_offset = !(num_alpha==1);
    A_offset = nrows*ncols;
    
    
    for (i=1; i<num_A; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), nrows, ncols);
        gsl_x = gsl_vector_view_array(x+(i*x_offset), x_vec_len);
        gsl_y = gsl_vector_view_array(y+(i*y_offset), y_vec_len);
        gsl_blas_dger (alpha[i*alpha_offset], x, y, A.matrix);
    }
        //dger(&nrows, &ncols, alpha+(i*alpha_offset), x+(i*x_offset), &inc, 
        //     y+(i*y_offset), &inc, A+(i*A_offset), &nrows);
    """

###############################################################################
# dsyr -- symmetric rank 1 operation, A = alpha*x*x**T + A


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
    support_code = global_support_code
    libraries = ["blas", "lapack", "gsl"]
    python_vars = ["A", "B", "C", "alpha", "beta"]
    code = """
    int i, num_A, num_B, num_C, num_alpha, num_beta;
    int A_rows, A_cols, B_rows, B_cols, C_rows, C_cols;
    int A_offset, B_offset, C_offset, alpha_offset, beta_offset;
    
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
    
    for (i=0; i<num_C; i++) {
        gsl_A = gsl_matrix_view_array(A+(i*A_offset), A_rows, A_cols);
        gsl_B = gsl_matrix_view_array(B+(i*B_offset), B_rows, B_cols);
        gsl_C = gsl_matrix_view_array(C+(i*C_offset), C_rows, C_cols);
        gsl_blas_dgemm (TransA, TransB, alpha[i*alpha_offset], &A.matrix, 
                        &B.matrix, beta[i*beta_offset], &C.matrix);
    }
    """

###############################################################################
# dsymm -- matrix-matrix operation, A is symmetric
#           C := alpha*A*B + beta*C, or
#           C := alpha*B*A + beta*C


###############################################################################
# dsyrk -- symmetric rank k operation, C is symmetric
#           C := alpha*A*A**T + beta*C, or
#           C := alpha*A**T*A + beta*C



###############################################################################
###############################################################################
class ldgemv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = global_support_code
    libraries = ["blas", "atlas", "lapack"]
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
    support_code = global_support_code
    libraries = ["blas", "lapack"]
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