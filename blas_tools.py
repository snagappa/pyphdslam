# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:08:44 2012

@author: snagappa
"""

import numpy as np
import copy
from scipy import weave
import blas_tools_c_code
#import phdmisctools
#import collections


default_dtype = np.float32

def myarray(*args, **kw):
    if 'dtype' not in kw.keys():
        return np.array(*args, dtype=default_dtype, **kw)
    else:
        return np.array(*args, **kw) 


DEBUG = True
NUM_ELEM_MISMATCH = "  *Did not recieve correct number of elements*  "
V_V_DIM_MISMATCH = "  *Vectors have incompatible dimensions*  "
M_V_DIM_MISMATCH = "  *Matrix and vector have incompatible dimensions*  "
M_M_DIM_MISMATCH = "  *Matrices have incompatible dimensions*  "


def blas_weaver(subroutine_string):
    subroutine = getattr(blas_tools_c_code, subroutine_string)
    compile_args = getattr(blas_tools_c_code, "EXTRA_COMPILE_ARGS", [])
    compile_args += getattr(subroutine, "extra_compile_args", [])
    
    fn_string = "blas_tools_c_code."+subroutine_string
    exec_string = "weave.inline(" + fn_string + ".code, " + fn_string+".python_vars" + ", libraries=" + fn_string + ".libraries" + ", support_code=" + fn_string + ".support_code" + ", extra_compile_args=" + str(compile_args) + ")"
    return exec_string
                 

global_support_code = """
#include <atlas/cblas.h>
#include <atlas/clapack.h>

/*
int clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb);
int clapack_dgetrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv);
int clapack_dgetrs
   (const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const double *A, const int lda,
    const int *ipiv, double *B, const int ldb);
int clapack_dgetri(const enum CBLAS_ORDER Order, const int N, double *A,
                   const int lda, const int *ipiv);
int clapack_dposv(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                  const int N, const int NRHS, double *A, const int lda,
                  double *B, const int ldb);
int clapack_dpotrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dpotrs(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const double *A, const int lda,
                   double *B, const int ldb);
int clapack_dpotri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dlauum(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dtrtri(const enum ATLAS_ORDER Order,const enum ATLAS_UPLO Uplo,
                  const enum ATLAS_DIAG Diag,const int N, double *A, const int lda);
*/
"""

#ITIS = collections.namedtuple('ITIS', ['valid', 'type', 'ndims', 'shape'])
LIST_OF_NDARRAY = "LIST_OF_NDARRAY"
LIST_OF_MATRIX = "LIST_OF_MATRIX"

class ITIS(object):
    def __init__(self, _valid, _type, _ndims, _shape, _base_type):
        self.valid = _valid
        self.type = _type
        self.ndims = _ndims
        self.shape = _shape
        self.base_type = _base_type
    def __repr__(self):
        return ("ITIS(valid=" + str(self.valid) + 
                ", type=" + str(self.type) + 
                ", ndims=" + str(self.ndims) + 
                ", shape=" + str(self.shape) + 
                ", base_type=" + str(self.base_type) + ")")
    def consistent_type(self):
        _type = np.array(self.type)
        if np.all([self.type[i] == np.ndarray for i in range(len(self.type))]):
            return np.ndarray
        if np.all([self.type[i] == np.matrix for i in range(len(self.type))]):
            return np.matrix
        elif all(_type==list):
            return list
        elif (self.type[0] == list) and np.all([self.type[i] == np.ndarray for i in range(1,len(self.type))]):
            return LIST_OF_NDARRAY
        elif (self.type[0] == list) and np.all([self.type[i] == np.matrix for i in range(1,len(self.type))]):
            return LIST_OF_MATRIX
        else:
            return None

def whatisit(arr):
    if type(arr) == np.ndarray:
        if arr.dtype == object:
            itis = ITIS(False, [np.ndarray], len(arr.shape), arr.shape, arr.dtype)
            return itis
        return ITIS(True, [np.ndarray], len(arr.shape), arr.shape, arr.dtype)
    elif type(arr) == np.matrix:
        if arr.dtype == object:
            itis = ITIS(False, [np.matrix], len(arr.shape), arr.shape, arr.dtype)
            return itis
        return ITIS(True, [np.matrix], len(arr.shape), arr.shape, arr.dtype)
    elif type(arr) == list:
        itis = ITIS(True, [list], 1, (len(arr),), None)
        sub_arr = whatisit(arr[0])
        itis.valid &= sub_arr.valid
        itis.type += sub_arr.type
        itis.ndims += sub_arr.ndims
        itis.shape += sub_arr.shape
        itis.base_type = sub_arr.base_type
        return itis
    else:
        itis = ITIS(True, [], 0, (), type(arr))
        return itis


def isvalidvector(x):
    x_is = whatisit(x)
    if not (x_is.consistent_type==np.ndarray):
        return False
    if not x.flags.c_continuous:
        return False
    if not (len(x.shape)==2):
        return False
    return True
    
def assert_valid_vector(x, varname="unknown"):
    x_is = whatisit(x)
    assert (x_is.consistent_type()==np.ndarray), varname + " must be of type ndarray"
    assert x.flags.c_contiguous, varname + " must be C-order contiguous"
    assert (len(x.shape)==2), varname + " must be 2 dimensional"
    
def assert_valid_matrix(x, varname="unknown"):
    x_is = whatisit(x)
    assert (x_is.consistent_type()==np.ndarray), varname + " must be of type ndarray"
    assert x.flags.c_contiguous, varname + " must be C-order contiguous"
    assert (len(x.shape)==3), varname + " must be 3 dimensional"
    
    
def isnestedstruct(itis_info):
    if not itis_info.valid:
        return True
    
    types_len = len(itis_info.type)
    if np.all([itis_info.type[i] == np.ndarray for i in range(types_len)]):
        return False
    elif itis_info.ndims > 1:
        return True
    else:
        return False


def _np_generate(np_function, itis):
    consistent_type = itis.consistent_type()
    if consistent_type == list:
        y = np_function(itis.shape, dtype=itis.base_type).tolist()
    elif consistent_type == np.ndarray:
        y = np_function(itis.shape, dtype=itis.base_type)
    elif consistent_type == LIST_OF_NDARRAY:
        y = [np_function(itis.shape[1:], dtype=itis.base_type) 
             for i in range(itis.shape[0])]
    else:
        y = None
    return y
    
def empty(itis):
    return _np_generate(np.empty, itis)

def zeros(itis):
    return _np_generate(np.zeros, itis)

def ones(itis):
    return _np_generate(np.ones, itis)

    
def _py_copy(x):
    itis = whatisit(x)
    if isnestedstruct(itis):
        y = copy.deepcopy(x)
    else:
        if itis.type[0] == np.ndarray:
            y = x.copy()
        else:
            y = copy.copy(x)
        
    return y


def fast_copy(x):
    itis = whatisit(x)
    consistent_type = itis.consistent_type()
    if consistent_type == list and (itis.ndims <=3):
        if itis.ndims == 1:
            y = copy.copy(x)
        else:
            y = np.empty(itis.shape, dtype=itis.base_type).tolist()
            exec blas_weaver("lcopy")
    elif consistent_type == np.ndarray:
        y = x.copy()
    elif consistent_type == LIST_OF_NDARRAY:
        y = [x[i].copy() for i in range(len(x))]
    else:
        y = _py_copy(x)
    return y
    

def ddot(x, y):
    if DEBUG:
        assert_valid_vector(x, "x")
        assert_valid_vector(y, "y")
        assert (x.shape[0] in [1, y.shape[0]]) and (y.shape[0] in [1, x.shape[0]]), NUM_ELEM_MISMATCH
        assert (x.shape[1] == y.shape[1]), V_V_DIM_MISMATCH
    xt_dot_y = np.zeros(max([x.shape[0], y.shape[0]]), dtype=float)
    exec blas_weaver("ddot")
    return xt_dot_y
    
    
def dnrm2(x):
    if DEBUG:
        assert_valid_vector(x, "x")
    
    nrm2 = np.zeros(x.shape[0], dtype=float)
    exec blas_weaver("dnrm2")
    return nrm2


def dasum(x):
    if DEBUG:
        assert_valid_vector(x)
    
    asum = np.zeros(x.shape[0], dtype=float)
    exec blas_weaver("dasum")
    return asum
    
    
def idamax(x):
    if DEBUG:
        assert_valid_vector(x, "x")
    
    max_idx = np.empty(x.shape[0], dtype=int)
    exec blas_weaver("idamax")
    return max_idx
    
    
def daxpy(alpha, x, y=None):
    if DEBUG:
        assert_valid_vector(x, "x")
    
    fn_return_val = None
    if y == None:
        y = np.zeros(x.shape)
        fn_return_val = y
    if DEBUG:
        assert_valid_vector(y, "y")
        assert x.shape == y.shape, NUM_ELEM_MISMATCH+V_V_DIM_MISMATCH
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    
    if DEBUG:
        assert len(alpha) in [1, x.shape[0]], V_V_DIM_MISMATCH
    
    exec blas_weaver("daxpy")
    return fn_return_val
    
    
def dscal(alpha, x):
    if DEBUG:
        assert_valid_vector(x)
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    
    if DEBUG:
        assert len(alpha) in (1, x.shape[0]), V_V_DIM_MISMATCH
    
    exec blas_weaver("dscal")
    
    
def dcopy(x, y):
    if DEBUG:
        assert_valid_vector(x, "x")
        assert_valid_vector(y, "y")
        assert x.shape==y.shape, NUM_ELEM_MISMATCH+V_V_DIM_MISMATCH
    exec blas_weaver("dcopy")
    
    
def _dcopy_(x, y, vec_len=None, x_offset=0, y_offset=0):
    num_x = np.prod(x.shape)
    num_y = np.prod(y.shape)
    if vec_len==None:
        vec_len = num_x
    assert x_offset+vec_len < num_x and y_offset+vec_len < num_y, NUM_ELEM_MISMATCH
    exec blas_weaver("_dcopy_")
    
##
## LEVEL 2 BLAS
##

#def dgemv(A, x, alpha=1.0, y=None, beta=1.0, TRANSPOSE_A=False):
def dgemv(A, x, TRANSPOSE_A=False, alpha=1.0, beta=1.0, y=None):
    """
    dgemv --
    y = alpha*A*x + beta*y OR 
    y = alpha*A**T*x + beta*y
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(x, "x")
    fn_return_val = None
    if y==None:
        if not TRANSPOSE_A:
            y = np.zeros(((max([A.shape[0],x.shape[0]]),) + (A.shape[1],)))
        else:
            y = np.zeros(((max([A.shape[0],x.shape[0]]),) + (A.shape[2],)))
        fn_return_val = y
    if DEBUG:
        assert_valid_vector(y, "y")
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if np.isscalar(beta):
        beta = np.array([beta], dtype=float)
    if DEBUG:
        assert len(alpha) in [1, x.shape[0], A.shape[0]], NUM_ELEM_MISMATCH
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), NUM_ELEM_MISMATCH
        if not TRANSPOSE_A:
            assert A.shape[2] == x.shape[1], M_V_DIM_MISMATCH
        else:
            assert A.shape[1] == x.shape[1], M_V_DIM_MISMATCH
        assert len(beta) in [1, y.shape[0]], NUM_ELEM_MISMATCH
    exec blas_weaver("dgemv")
    return fn_return_val
    
    
def dtrmv(UPLO, A, x, TRANSPOSE_A=False):
    """
    dtrmv -- 
    x := A*x  if TRANSPOSE_A=False or   
    x := A**T*x  if TRANSPOSE_A=True
    A is nxn triangular
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(x, "x")
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), NUM_ELEM_MISMATCH
        assert A.shape[2] == x.shape[1], M_V_DIM_MISMATCH
    
    exec blas_weaver("dtrmv")
    

def dtrsv(UPLO, A, x, TRANSPOSE_A=False):
    """
    dtrsv -- Solve A*x = b,   or   A**T*x = b, A is nxn triangular
    x = A^{-1}*b or x = A^{-T}b
    """
    if DEBUG:
        assert_valid_matrix(A)
        assert_valid_vector(x)
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), NUM_ELEM_MISMATCH
        if TRANSPOSE_A:
            assert A.shape[1] == x.shape[1], M_V_DIM_MISMATCH
        else:
            assert A.shape[2] == x.shape[1], M_V_DIM_MISMATCH
    
    exec blas_weaver("dtrsv")
    

def dsymv(UPLO, A, x, alpha=1.0, beta=1.0, y=None):
    """
    dsymv -- y = alpha*A*x + beta*y, A is symmetric
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(x, "x")
    fn_return_val = None
    if y==None:
        y = np.zeros(((max([A.shape[0],x.shape[0]]),) + (x.shape[1],)))
        fn_return_val = y
    if DEBUG:
        assert_valid_vector(y, "y")
        assert y.shape[1] == x.shape[1], V_V_DIM_MISMATCH
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if np.isscalar(beta):
        beta = np.array([beta], dtype=float)
    if DEBUG:
        assert len(alpha) in [1, x.shape[0], A.shape[0]], NUM_ELEM_MISMATCH
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), NUM_ELEM_MISMATCH
        assert A.shape[2] == x.shape[1], M_V_DIM_MISMATCH
        assert len(beta) in [1, y.shape[0]], NUM_ELEM_MISMATCH
    
    exec blas_weaver("dsymv")
    return fn_return_val
    
    
def dger(x, y, alpha=1.0, A=None):
    """
    dger -- rank 1 operation, A = alpha*x*y**T + A
    """
    if DEBUG:
        assert_valid_vector(x, "x")
        assert_valid_vector(y, "y")
    
    fn_return_val = None
    if A==None:
        A = np.zeros((max([x.shape[0], y.shape[0]]), x.shape[1], y.shape[1]), dtype=float)
        fn_return_val = A
    
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert A.shape == (max([x.shape[0], y.shape[0]]), x.shape[1], y.shape[1]), NUM_ELEM_MISMATCH+M_V_DIM_MISMATCH
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        #assert_valid_vector(alpha, "alpha")
        assert len(alpha) in [1, x.shape[0], y.shape[0]], NUM_ELEM_MISMATCH
        assert (x.shape[0]==y.shape[0]) or (x.shape[0]==1) or (y.shape[0]==1), NUM_ELEM_MISMATCH
    
    exec blas_weaver("dger")
    return fn_return_val
    
    
def dsyr(UPLO, x, alpha=1.0, A=None):
    """
    dsyr -- symmetric rank 1 operation
    A = alpha*x*x**T + A,  A is nxn symmetric
    """
    if DEBUG:
        assert_valid_vector(x, "x")
    
    fn_return_val = None
    if A==None:
        A = np.zeros(x.shape + (x.shape[1],), dtype=float)
        fn_return_val = A
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        assert_valid_matrix(A)
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1), NUM_ELEM_MISMATCH
        assert len(alpha) in [1, x.shape[0]], NUM_ELEM_MISMATCH
        assert A.shape[1]==A.shape[2]==x.shape[1], M_V_DIM_MISMATCH
    
    exec blas_weaver("dsyr")
    return fn_return_val


##
## LEVEL 3 BLAS
##

#def dgemm(A, B, alpha=1.0, C=None, beta=1.0, TRANSPOSE_A=False, TRANSPOSE_B=False):
def dgemm(A, B, TRANSPOSE_A=False, TRANSPOSE_B=False, alpha=1.0, beta=1.0, C=None):
    """
    dgemm -- matrix-matrix operation, 
    C := alpha*op( A )*op( B ) + beta*C,
    where  op( X ) is one of op( X ) = X   or   op( X ) = X**T
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_matrix(B, "B")
        assert A.shape[0] in [1, B.shape[0]] and B.shape[0] in [1, A.shape[0]], NUM_ELEM_MISMATCH
    
    A_dims = list(A.shape[1:])
    B_dims = list(B.shape[1:])
    if TRANSPOSE_A:
        A_dims.reverse()
    if TRANSPOSE_B:
        B_dims.reverse()
            
    fn_return_val = None
    if C==None:
        C = np.zeros((max([A.shape[0], B.shape[0]]), A_dims[0], B_dims[1]), dtype=float)
        fn_return_val = C
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if np.isscalar(beta):
        beta = np.array([beta], dtype=float)
        
    if DEBUG:
        assert_valid_matrix(C)
        if A.shape[0] == B.shape[0]==1:
            assert C.shape[1:] == (A_dims[0], B_dims[1]), NUM_ELEM_MISMATCH
        else:
            assert C.shape == (max([A.shape[0], B.shape[0]]), A_dims[0], B_dims[1]), M_M_DIM_MISMATCH
        assert A_dims[1]==B_dims[0], M_M_DIM_MISMATCH
        assert len(alpha) in [1, A.shape[0], B.shape[0]], NUM_ELEM_MISMATCH
        assert len(beta) in [1, C.shape[0]], NUM_ELEM_MISMATCH
    
    exec blas_weaver("dgemm")
    return fn_return_val


def dsymm(A, B, UPLO, SIDE='l', alpha=1.0, beta=1.0, C=None):
    """
    dsymm -- matrix-matrix operation, A is symmetric, B and C are mxn
    C := alpha*A*B + beta*C, if SIDE='l' or
    C := alpha*B*A + beta*C, if SIDE='r'
    UPLO = 'l' or 'u'
    """
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        assert SIDE in ['l', 'L', 'r', 'R'], "SIDE must be one of ['l', 'L', 'r', 'R']"
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'L', 'u', 'U']"
        assert_valid_matrix(A, "A")
        assert_valid_matrix(B, "B")
        assert A.shape[0] in [1, B.shape[0]] and B.shape[0] in [1, A.shape[0]], NUM_ELEM_MISMATCH
        assert len(alpha) in [1, A.shape[0], B.shape[0]], NUM_ELEM_MISMATCH
            
    A_dims = list(A.shape[1:])
    B_dims = list(B.shape[1:])
    
    if np.isscalar(beta):
        beta = np.array([beta], dtype=float)
    fn_return_val = None
    if C==None:
        if SIDE=='l':
            C = np.zeros((max([A.shape[0], B.shape[0]]), A_dims[0], B_dims[1]), dtype=float)
        else:
            C = np.zeros((max([A.shape[0], B.shape[0]]), B_dims[0], A_dims[1]), dtype=float)
        fn_return_val = C
        
    if DEBUG:
        assert_valid_matrix(C)
        if SIDE=='l':
            assert C.shape == (max([A.shape[0], B.shape[0]]), A_dims[0], B_dims[1]), NUM_ELEM_MISMATCH+M_M_DIM_MISMATCH
        else:
            assert C.shape == (max([A.shape[0], B.shape[0]]), B_dims[0], A_dims[1]), NUM_ELEM_MISMATCH+M_M_DIM_MISMATCH
        assert len(beta) in [1, C.shape[0]], NUM_ELEM_MISMATCH
    
    exec blas_weaver("dsymm")
    return fn_return_val


def dsyrk(UPLO, A, TRANSPOSE_A=False, alpha=1.0, beta=1.0, C=None):
    """
    dsyrk -- symmetric rank k operation, C is symmetric
    C := alpha*A*A**T + beta*C  if TRANSPOSE_A=False or
    C := alpha*A**T*A + beta*C  if TRANSPOSE_A=True
    """
    A_dims = list(A.shape[1:])
    if TRANSPOSE_A:
        A_dims.reverse()
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    
    if DEBUG:
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'L', 'u', 'U']"
        assert_valid_matrix(A, "A")
        assert alpha.shape[0] in [1, A.shape[0]], NUM_ELEM_MISMATCH
    
    fn_return_val = None
    if C==None:
        C = np.zeros((A.shape[0], A_dims[0], A_dims[1]), dtype=float)
        fn_return_val = C
    if np.isscalar(beta):
        beta = np.array([beta], dtype=float)
        
    if DEBUG:
        assert_valid_matrix(C)
        assert C.shape == (A.shape[0], A_dims[0], A_dims[0]), M_M_DIM_MISMATCH
        assert beta.shape[0] in [1, C.shape[0]], NUM_ELEM_MISMATCH
    
    exec blas_weaver("dsyrk")
    return fn_return_val


##
## LINEAR ALGEBRA
##

# LU decomposition
def dgetrf(A, INPLACE=False):
    if DEBUG:
        assert_valid_matrix(A, "A")
    ipiv = np.zeros(A.shape[0:2], dtype=int)
    signum = np.zeros(A.shape[0])
    if not INPLACE:
        A = A.copy()
        fn_return_val = A, ipiv, signum
    else:
        fn_return_val = ipiv, signum
    
    exec blas_weaver("dgetrf")
    return fn_return_val


# Solve Ax=B using LU decomposition
def dgetrs(LU, ipiv, b, x=None):
    if DEBUG:
        assert_valid_matrix(LU, "LU")
        assert_valid_vector(ipiv, "ipiv")
        assert_valid_vector(b, "b")
        assert ipiv.shape[0] == LU.shape[0], NUM_ELEM_MISMATCH
        assert LU.shape[1] == ipiv.shape[1], M_M_DIM_MISMATCH
        assert LU.shape[0] in [1, b.shape[0]] and b.shape[0] in [1, LU.shape[0]], NUM_ELEM_MISMATCH
        assert LU.shape[2] == b.shape[1], M_V_DIM_MISMATCH
    
    fn_return_val = None
    if x==None:
        x = np.zeros((max([LU.shape[0], b.shape[0]]), b.shape[1]), dtype=float)
        fn_return_val = x
    if DEBUG:
        assert_valid_vector(x, "x")
        assert x.shape == (max([LU.shape[0], b.shape[0]]), b.shape[1]), NUM_ELEM_MISMATCH+V_V_DIM_MISMATCH
    
    exec blas_weaver("dgetrs")
    return fn_return_val


# Solve Ax=B using LU decomposition in place
def dgetrsx(LU, ipiv, b):
    if DEBUG:
        assert_valid_matrix(LU, "LU")
        assert_valid_vector(ipiv, "ipiv")
        assert_valid_vector(b, "b")
        assert ipiv.shape[0] == LU.shape[0], NUM_ELEM_MISMATCH
        assert LU.shape[1] == ipiv.shape[1], M_M_DIM_MISMATCH
        assert LU.shape[0] in [1, b.shape[0]], NUM_ELEM_MISMATCH
        assert LU.shape[2] == b.shape[1], M_V_DIM_MISMATCH
    
    exec blas_weaver("dgetrsx")
    

def dgetri(LU, ipiv):
    if DEBUG:
        assert_valid_matrix(LU, "LU")
        assert_valid_vector(ipiv, "ipiv")
        assert LU.shape[0] == ipiv.shape[0], NUM_ELEM_MISMATCH
        assert LU.shape[1] == ipiv.shape[1], M_M_DIM_MISMATCH
    invA = np.zeros(LU.shape, dtype=float)
    exec blas_weaver("dgetri")
    return invA


# Solve Ax = B
def dgesv(A, b, INPLACE=False, OVERWRITE_A=False):
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(b, "b")
        if INPLACE:
            assert A.shape[0] in [1, b.shape[0]]
    fn_return_val = ()
    if not OVERWRITE_A:
        A = A.copy()
        fn_return_val += (A,)
    else:
        fn_return_val += (None,)
    ipiv, signum = dgetrf(A, INPLACE=True)
    fn_return_val += (ipiv, signum)
    if not INPLACE:
        x = dgetrs(A, ipiv, b)
        fn_return_val += (x,)
    else:
        dgetrsx(A, ipiv, b)
        fn_return_val += (None,)
    return fn_return_val
    

def dgetrdet(LU, signum):
    if DEBUG:
        assert_valid_matrix(LU)
        assert signum.shape[0] == LU.shape[0], NUM_ELEM_MISMATCH
    det_vec = np.zeros(LU.shape[0])
    exec blas_weaver("dgetrdet")
    return det_vec
    
    
# Cholesky decomposition
def dpotrf(A, INPLACE=False):
    if DEBUG:
        assert_valid_matrix(A, "A")
    fn_return_val = None
    if not INPLACE:
        A = A.copy()
        fn_return_val = A
    exec blas_weaver("dpotrf")
    return fn_return_val


# Solve using Cholesky decomposition
def dpotrs(cholA, b, x=None):
    if DEBUG:
        assert_valid_matrix(cholA, "cholA")
        assert_valid_vector(b, "b")
        assert cholA.shape[0]==b.shape[0] or cholA.shape[0]==1 or b.shape[0]==1, NUM_ELEM_MISMATCH
        assert cholA.shape[2] == b.shape[1], M_V_DIM_MISMATCH
    
    fn_return_val = None
    if x==None:
        x = np.zeros((max([cholA.shape[0], b.shape[0]]), b.shape[1]), dtype=float)
        fn_return_val = x
    if DEBUG:
        assert_valid_vector(x, "x")
        assert x.shape == (max([cholA.shape[0], b.shape[0]]), b.shape[1]), NUM_ELEM_MISMATCH+V_V_DIM_MISMATCH
    
    exec blas_weaver("dpotrs")
    return fn_return_val


# Solve Ax=B using LU decomposition in place
def dpotrsx(cholA, b):
    if DEBUG:
        assert_valid_matrix(cholA, "cholA")
        assert_valid_vector(b, "b")
        assert cholA.shape[0] in [1, b.shape[0]], NUM_ELEM_MISMATCH
        assert cholA.shape[2] == b.shape[1], M_V_DIM_MISMATCH
    
    exec blas_weaver("dpotrsx")


# Compute inverse using Cholesky factorisation from dpotrf
def dpotri(A, INPLACE=False):
    if DEBUG:
        assert_valid_matrix(A)
    fn_return_val = None
    if not INPLACE:
        A = A.copy()
        fn_return_val = A
    exec blas_weaver("dpotri")
    return fn_return_val
    

# Solve for positive definite matrix
def dposv(A, b, INPLACE=False, OVERWRITE_A=False):
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(b, "b")
        if INPLACE:
            assert A.shape[0] in [1, b.shape[0]]
    fn_return_val = ()
    if not OVERWRITE_A:
        A = A.copy()
        fn_return_val += (A,)
    else:
        fn_return_val += (None,)
    dpotrf(A, INPLACE=True)
    
    if not INPLACE:
        x = dpotrs(A, b)
        fn_return_val += (x,)
    else:
        dpotrsx(A, b)
        fn_return_val += (None,)
    return fn_return_val
    

# Compute inverse of a triangular matrix using dtrtri
def dtrtri(A, UPLO, INPLACE=False):
    if DEBUG:
        assert_valid_matrix(A)
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'L', 'u', 'U']"
    fn_return_val = None
    if not INPLACE:
        A = A.copy()
        fn_return_val = A
    exec blas_weaver("dtrtri")
    return fn_return_val
    
    
def inverse(A, FACTORISATION, INPLACE=False):
    assert FACTORISATION[0] in ["c", "C", "l", "L"], "Factorisation must be either 'c' - cholesky or 'l' - LU"
    if not INPLACE:
        A = A.copy()
    if FACTORISATION[0] in ["c", "C"]:
        dpotrf(A, True)
        dpotri(A, True)
    else:
        ipiv, signum = dgetrf(A, True)
        dgetri(A, ipiv)
    return A
    
            
def symmetrise(A, UPLO):
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert A.shape[1]==A.shape[2], "A must be symmetric"
        assert type(UPLO)
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'u', 'L', 'U']"
    weave.inline(blas_tools_c_code.symmetrise.code, 
                 blas_tools_c_code.symmetrise.python_vars, 
                 libraries=blas_tools_c_code.symmetrise.libraries,
                 support_code=blas_tools_c_code.symmetrise.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    

def mahalanobis(x, P, y):
    residual = x.copy()
    P_copy = P.copy()
    daxpy(-1.0, y, residual)
    _, p_residual = dposv(P_copy, residual, OVERWRITE_A=True)
    return np.power(ddot(residual, p_residual), 0.5)
    
    
def merge_states(wt, x, P):
    merged_wt = wt.sum()
    merged_x = np.sum(daxpy(wt, x), 0)/merged_wt
    residual = x.copy()
    daxpy(-1.0, np.array([merged_x]), residual)
    # Convert the residual to a column vector
    residual.shape += (1,)
    P_copy = P.copy()
    merged_P = sum(dsyrk('l', residual, True, 1.0, wt, P_copy), 0)/merged_wt
    return merged_wt, merged_x, merged_P
    
    
    
def dgemv_old(A, x, alpha=1.0, beta=0.0, y=None, TRANSPOSE_A=False):
    A_is = whatisit(A)
    x_is = whatisit(x)
    A_consistent_type = A_is.consistent_type()
    x_consistent_type = x_is.consistent_type()
    
    assert A_consistent_type == x_consistent_type, "A and x must have consistent types"
    if not (y == None):
        y_is = whatisit(y)
        y_consistent_type = y_is.consistent_type
        assert A_consistent_type == y_consistent_type, "A, x and y must have consistent types"
    
    assert A_consistent_type in [np.ndarray, list], "Valid type must be used"
    y_is = copy.deepcopy(x_is)
    if not (A_is.shape[0] == x_is.shape[0]):
        y_is.shape = ((max([A_is.shape[0],x_is.shape[0]]),) + x_is.shape[1:])
    
    x_shape = x_is.shape
    A_shape = A_is.shape
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    assert type(alpha) == np.ndarray, "alpha must be of type scalar float or float ndarray"
    assert len(alpha) in [1, x_shape[0]], "alpha must be a scalar or have same length as x"
    
    y = zeros(y_is)
    assert A.flags.c_contiguous or A.flags.f_contiguous, "blas_tools.dgemv may only be used with contiguous data"
    C_CONTIGUOUS = A.flags.c_contiguous
    weave.inline(blas_tools_c_code.npdgemv.code, 
                 blas_tools_c_code.npdgemv.python_vars, 
                 libraries=blas_tools_c_code.npdgemv.libraries,
                 support_code=blas_tools_c_code.npdgemv.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return y
    """
    if A_consistent_type == list:
        y = zeros(y_is)
        weave.inline(blas_tools_c_code.ldgemv.code, 
                     blas_tools_c_code.ldgemv.python_vars, 
                     libraries=blas_tools_c_code.ldgemv.libraries,
                     support_code=blas_tools_c_code.ldgemv.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
        
    elif A_consistent_type == np.ndarray:
        y = zeros(y_is)
        assert A.flags.c_contiguous or A.flags.f_contiguous, "blas_tools.dgemv may only be used with contiguous data"
        C_CONTIGUOUS = A.flags.c_contiguous
        weave.inline(blas_tools_c_code.npdgemv.code, 
                     blas_tools_c_code.npdgemv.python_vars, 
                     libraries=blas_tools_c_code.npdgemv.libraries,
                     support_code=blas_tools_c_code.npdgemv.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
                     
    
    elif A_consistent_type in [LIST_OF_NDARRAY, LIST_OF_MATRIX]:
        assert A.flags.c_contiguous or A.flags.f_contiguous, "blas_tools.dgemv may only be used with contiguous data"
        
        y_is.consistent_type = np.ndarray
        y = zeros(y_is)
        x = np.array(x)
        weave.inline(blas_tools_c_code.npdgemv.code, 
                     blas_tools_c_code.npdgemv.python_vars, 
                     libraries=blas_tools_c_code.npdgemv.libraries,
                     support_code=blas_tools_c_code.npdgemv.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
        y = [y[i] for i in range(len(y))]
    
    return y
    """
    
    
def test_ddot(num_elements=1000, num_dims=4):
    print "Computing dot product..."
    x = np.random.rand(num_elements, num_dims)
    y = np.random.rand(num_elements, num_dims)
    # Using numpy:
    np_xt_dot_y = np.sum(x*y, 1)
    # Using blas
    b_xt_dot_y = ddot(x, y)
    max_err = max(np.abs(np_xt_dot_y-b_xt_dot_y))
    print "Max error = ", max_err
    
def test_dnrm2(num_elements=1000, num_dims=4):
    print "Computing dot product..."
    x = np.random.rand(num_elements, num_dims)
    # Using numpy:
    np_norm2 = np.array([np.linalg.norm(x[i]) for i in range(num_elements)])
    # Using blas
    b_norm2 = dnrm2(x)
    max_err = max(np.abs(np_norm2-b_norm2))
    print "Max error = ", max_err
    
def test_dasum(num_elements=1000, num_dims=4):
    print "Computing dot product..."
    x = np.random.rand(num_elements, num_dims)
    # Using numpy:
    np_asum = np.sum(np.abs(x), 1)
    # Using blas
    b_asum = dasum(x)
    max_err = max(abs(np.abs(np_asum-b_asum)))
    print "Max error = ", max_err
    
def test_idamax(num_elements=1000, num_dims=4):
    print "Computing dot product..."
    x = np.random.rand(num_elements, num_dims)
    # Using numpy:
    np_idx = x.argmax(1)
    # Using blas
    b_idx = idamax(x)
    max_err = max(abs(np.abs(np_idx-b_idx+1)))
    print "Max error = ", max_err
    
def test_daxpy(num_elements=1000, num_dims=4):
    print "Computing dot product..."
    a = np.random.rand(num_elements)
    x = np.random.rand(num_elements, num_dims)
    y = np.random.rand(num_elements, num_dims)
    # Using numpy:
    np_axpy = np.array([a[i]*x[i]+y[i] for i in range(num_elements)]).flatten()
    # Using blas
    daxpy(a, x, y)
    max_err = max(abs(np.abs(np_axpy-y.flatten())))
    print "Max error = ", max_err
    #assert max_err < 1e-12, "Error exceeds 1e-12"
    
def test_dscal(num_elements=1000, num_dims=4):
    print "Computing dot product..."
    a = np.random.rand(num_elements)
    x = np.random.rand(num_elements, num_dims)
    # Using numpy:
    np_scal = np.array([a[i]*x[i] for i in range(num_elements)]).flatten()
    # Using blas
    dscal(a, x)
    max_err = max(abs(np.abs(np_scal-x.flatten())))
    print "Max error = ", max_err

def test_dgemv(num_elements=1000, num_dims=4, num_rows=4):
    print "Case 1a: num_P == num_x"
    x = np.random.rand(num_elements, num_dims)
    P = np.random.rand(num_elements, num_rows, num_dims)
    # Using numpy only
    np_result = np.array([np.dot(P[i], x[i]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x))
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    print "Maximum error = ", max_err1, " for np_arrays"
    
    print "Case 1b: num_P == num_x with transpose"
    x = np.random.rand(num_elements, num_dims)
    P = np.random.rand(num_elements, num_dims, num_rows)
    # Using numpy only
    np_result = np.array([np.dot(P[i].T, x[i]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x), TRANSPOSE_A=True)
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    print "Maximum error = ", max_err1, " for np_arrays"
    
    print "Case 2: num_P == 1"
    x = np.random.rand(num_elements, num_dims)
    P = np.random.rand(1, num_rows, num_dims)
    # Using numpy only
    np_result = np.array([np.dot(P[0], x[i]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x))
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    print "Maximum error = ", max_err1, " for np_arrays"
    
    print "Case 3: num_x == 1"
    x = np.random.rand(1, num_dims)
    P = np.random.rand(num_elements, num_rows, num_dims)
    # Using numpy only
    np_result = np.array([np.dot(P[i], x[0]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x))
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    print "Maximum error = ", max_err1, " for np_arrays"
    
def test_dtrmv(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dtrsv(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dsymv(num_elements=1000, num_dims=4):
    alpha = np.random.rand(num_elements)
    beta = np.random.rand(num_elements)
    A = np.zeros((num_elements, num_dims, num_dims))
    dgemm(np.array([np.eye(num_dims)]), np.array([np.eye(num_dims)]), C=A)
    x = np.random.rand(num_elements, num_dims)
    dger(x, x, A=A)
    y_orig = np.random.rand(num_elements, num_dims)
    y = y_orig.copy()
    # Using numpy only
    np_result = np.array([alpha[i]*np.dot(A[i], x[i])+beta[i]*y[i] for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    dsymv('l', A, x, alpha, beta, y)
    max_err1 = np.max(np.abs(np_result - y.squeeze()))
    print "Maximum error = ", max_err1, " for np_arrays"


def test_dger(num_elements=1000, dims_x=4, dims_y=2):
    x = np.random.rand(num_elements, dims_x)
    y = np.random.rand(num_elements, dims_y)
    A = np.random.rand(num_elements, dims_x, dims_y)
    alpha = np.random.rand(num_elements)
    np_dger = np.array([np.array(alpha[i]*np.dot(np.matrix(x[i]).T, np.matrix(y[i]))) + A[i] for i in range(num_elements)])
    dger(x, y, alpha, A)
    max_err = np.max(np.abs(np_dger.squeeze()-A.squeeze()))
    print "Maximum error = ", max_err
    
def test_dsyr(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgemm(num_elements=1000, rows_A=4, cols_A=2, cols_B=6):
    alpha = np.random.rand(num_elements)
    A = np.random.rand(num_elements, rows_A, cols_A)
    B = np.random.rand(num_elements, cols_A, cols_B)
    beta = np.random.rand(num_elements)
    C = np.random.rand(num_elements, rows_A, cols_B)
    
    
    np_dgemm = np.array([np.array(alpha[i]*np.dot(A[i], B[i])) + beta[i]*C[i] for i in range(num_elements)])
    dgemm(A, B, alpha=alpha, beta=beta, C=C)
    max_err = np.max(np.abs(np_dgemm.squeeze()-C.squeeze()))
    print "Maximum error = ", max_err
    
def test_dsymm(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dsyrk(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgetrf(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgetrs(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgetrsx(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgetri(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgesv(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dgetrdet(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dpotrf(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dpotrs(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dpotrsx(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dpotri(num_elements=1000, num_dims=4, num_rows=4):
    pass

def test_dposv(num_elements=1000, num_dims=4, num_rows=4):
    pass
