# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:05:54 2012

@author: snagappa
"""

EXTRA_COMPILE_ARGS = ["-O2"]

global_support_code = """
#include <atlas/cblas.h>
#include <atlas/clapack.h>

extern "C" {
    int dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);
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
    

# dsdot -- inner product of two vectors


# ddot -- dot product of two vectors


# dnrm2 -- Euclidean norm of a vector


# dasum -- sum of absolute values


# idamax -- find index of element having the max absolute value


# daxpy -- y = ax+y
class daxpy:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = global_support_code
    libraries = ["blas", "lapack"]
    python_vars = ["alpha", "x", "y"]
    code = """
    int inc;
    int num_x;
    
    inc = 1;
    num_x = Nx[0];
    daxpy_(&num_x, &alpha, x, &inc, y, &inc);
    """
    

# dscal -- x = ax
class dscal:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = global_support_code
    libraries = ["blas", "lapack"]
    python_vars = ["alpha", "x"]
    code = """
    int inc;
    int num_x;
    
    inc = 1;
    num_x = Nx[0];
    dscal_(&num_x, &alpha, x, &inc);
    """


###############################################################################
# dgemv -- y = alpha*A*x + beta*y OR y = alpha*A**T*x + beta*y
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
        exception_occured = 1;
    }
    free(A_copy);
    free(x_copy);
    free(y_copy);
    """
    
class npdgemv:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = global_support_code
    libraries = ["blas", "lapack"]
    python_vars = ["A", "x", "y", "alpha", "beta", "TRANSPOSE_A", "C_CONTIGUOUS"]
    code = """
    int i, j, k, base_offset, inc;
    int num_A, num_x;
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
    
    nrows = NA[1];
    ncols = NA[2];
    
    base_offset = nrows*ncols;
    if (num_A == num_x) {
        for (i=0; i<num_A; i++)
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A+(i*base_offset), nrows, x+(i*ncols), 1, beta, y+(i*ncols), 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A+(i*base_offset), &nrows, x+(i*ncols), &inc, &beta, y+(i*ncols), &inc);
    }
    else if (num_A == 1) {
        for (i=0; i<num_x; i++)
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A, nrows, x+(i*ncols), 1, beta, y+(i*ncols), 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A, &nrows, x+(i*ncols), &inc, &beta, y+(i*ncols), &inc);
    }
    else if (num_x == 1) {
        for (i=0; i<num_A; i++)
            //cblas_dgemv(order, transA, nrows, ncols, alpha, A+(i*base_offset), nrows, x, 1, beta, y+(i*ncols), 1);
            dgemv_(&f_transA, &nrows, &ncols, &alpha, A+(i*base_offset), &nrows, x, &inc, &beta, y+(i*ncols), &inc);
    }
    else {
        exception_occured = 1;
    }
    """

# dsymv -- y = alpha*A*x + beta*y, A is symmetric


# dger -- rank 1 operation, A = alpha*x*y**T + A


# dsyr -- symmetric rank 1 operation, A = alpha*x*x**T + A


# dsyr2 -- symmetric rank 2 operation, A = alpha*x*y**T + alpha*y*x**T + A



## Level 3 BLAS

# dgemm -- matrix-matrix operation, C := alpha*op( A )*op( B ) + beta*C,
#           where  op( X ) is one of
#           op( X ) = X   or   op( X ) = X**T


# dsymm -- matrix-matrix operation, A is symmetric
#           C := alpha*A*B + beta*C, or
#           C := alpha*B*A + beta*C


# dsyrk -- symmetric rank k operation, C is symmetric
#           C := alpha*A*A**T + beta*C, or
#           C := alpha*A**T*A + beta*C