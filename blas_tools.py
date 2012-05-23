# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:08:44 2012

@author: snagappa
"""

import numpy as np
import copy
from scipy import weave
import blas_tools_c_code
import phdmisctools
#import collections


default_dtype = np.float32

def array(*args, **kw):
    if 'dtype' not in kw.keys():
        return np.array(*args, dtype=default_dtype, **kw)
    else:
        return np.array(*args, **kw) 

DEBUG = False


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
            weave.inline(blas_tools_c_code.lcopy.code, 
                     blas_tools_c_code.lcopy.python_vars, 
                     libraries=blas_tools_c_code.lcopy.libraries,
                     support_code=blas_tools_c_code.lcopy.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
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
        assert (x.shape[0] in [1, y.shape[0]]) and (y.shape[0] in [1, x.shape[0]]), "number of elements in x and y must match"
        assert (x.shape[1] == y.shape[1]), "vectors must have equal lengths"
    xt_dot_y = np.zeros(max([x.shape[0], y.shape[0]]), dtype=float)
    weave.inline(blas_tools_c_code.ddot.code,
                 blas_tools_c_code.ddot.python_vars, 
                 libraries=blas_tools_c_code.ddot.libraries,
                 support_code=blas_tools_c_code.ddot.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return xt_dot_y
    
    
def dnrm2(x):
    if DEBUG:
        assert_valid_vector(x, "x")
    
    nrm2 = np.zeros(x.shape[0], dtype=float)
    weave.inline(blas_tools_c_code.dnrm2.code,
                 blas_tools_c_code.dnrm2.python_vars, 
                 libraries=blas_tools_c_code.dnrm2.libraries,
                 support_code=blas_tools_c_code.dnrm2.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return nrm2


def dasum(x):
    if DEBUG:
        assert_valid_vector(x)
    
    asum = np.zeros(x.shape[0], dtype=float)
    weave.inline(blas_tools_c_code.dasum.code,
                 blas_tools_c_code.dasum.python_vars, 
                 libraries=blas_tools_c_code.dasum.libraries,
                 support_code=blas_tools_c_code.dasum.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return asum
    
    
def idamax(x):
    if DEBUG:
        assert_valid_vector(x, "x")
    
    max_idx = np.empty(x.shape[0], dtype=int)
    weave.inline(blas_tools_c_code.idamax.code,
                 blas_tools_c_code.idamax.python_vars, 
                 libraries=blas_tools_c_code.idamax.libraries,
                 support_code=blas_tools_c_code.idamax.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
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
        assert x.shape == y.shape, "x and y must have the same shape"
    
    if np.isscalar(alpha):
        alpha = np.array([alpha])
    alpha = alpha.astype(float)
    
    if DEBUG:
        assert len(alpha) in [1, x.shape[0]], "alpha must be a scalar or a numpy array of same length as x"
    
    weave.inline(blas_tools_c_code.daxpy.code,
                 blas_tools_c_code.daxpy.python_vars, 
                 libraries=blas_tools_c_code.daxpy.libraries,
                 support_code=blas_tools_c_code.daxpy.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return fn_return_val
    
    
def dscal(alpha, x):
    if DEBUG:
        assert_valid_vector(x)
    
    if np.isscalar(alpha):
        alpha = np.array([alpha])
    alpha = alpha.astype(float)
    
    if DEBUG:
        assert len(alpha) in (1, x.shape[0]), "alpha must be a scalar or a numpy array of same length as x"
    
    weave.inline(blas_tools_c_code.dscal.code,
                 blas_tools_c_code.dscal.python_vars, 
                 libraries=blas_tools_c_code.dscal.libraries,
                 support_code=blas_tools_c_code.dscal.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    

def dgemv(A, x, alpha=1.0, y=None, beta=1.0, TRANSPOSE_A=False):
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(x, "x")
    fn_return_val = None
    if y==None:
        y = np.zeros(((max([A.shape[0],x.shape[0]]),) + x.shape[1:]))
        fn_return_val = y
    if DEBUG:
        assert_valid_vector(y, "y")
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if np.isscalar(beta):
        beta = np.array([beta], dtype=float)
    if DEBUG:
        assert_valid_vector(alpha, "alpha")
        assert_valid_vector(beta, "beta")
        assert len(alpha) in [1, x.shape[0]], "alpha must be a scalar or have same length as x"
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), "incompatible sizes for A and x specified"
        assert A.shape[2] == x.shape[1], "x must have as many elements as columns in A"
        assert len(beta) in [1, y.shape[0]], "beta must be a scalar or have same length as y"
    C_CONTIGUOUS = A.flags.c_contiguous
    weave.inline(blas_tools_c_code.dgemv.code, 
                 blas_tools_c_code.dgemv.python_vars, 
                 libraries=blas_tools_c_code.dgemv.libraries,
                 support_code=blas_tools_c_code.dgemv.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return fn_return_val
    
    
def dger(x, y, alpha=1.0, A=None):
    if DEBUG:
        assert_valid_vector(x, "x")
        assert_valid_vector(y, "y")
    
    fn_return_val = None
    if A==None:
        A = np.zeros((max([x.shape[0], y.shape[0]]), x.shape[1], y.shape[1]), dtype=float)
        fn_return_val = A
    
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert A.shape == (max([x.shape[0], y.shape[0]]), x.shape[1], y.shape[1]), "x,y are incompatible with A"
    
    if np.isscalar(alpha):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        assert_valid_vector(alpha, "alpha")
        assert len(alpha) in [1, x.shape[0], y.shape[0]], "alpha must be a scalar or have same length as x"
        assert (x.shape[0]==y.shape[0]) or (x.shape[0]==1) or (y.shape[0]==1), "incompatible sizes for x and y specified"
    weave.inline(blas_tools_c_code.dger.code, 
                 blas_tools_c_code.dger.python_vars, 
                 libraries=blas_tools_c_code.dger.libraries,
                 support_code=blas_tools_c_code.dger.support_code,
                 extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
    return fn_return_val
    
    
    
def dgemm(A, B, alpha=1.0, C=None, beta=1.0, TRANSPOSE_A=False, TRANSPOSE_B=False):
    assert_valid_matrix(A, "A")
    assert_valid_matrix(B, "B")
    
    fn_return_val = None
    if C==None:
        C_rows = A.shape[2] if TRANSPOSE_A else A.shape[1]
        C_cols = B.shape[1] if TRANSPOSE_B else B.shape[2]
        C = np.zeros((max([A.shape[0], B.shape[0]]), C_rows, C_cols), dtype=float)
        fn_return_val = C
    C_is = whatisit(C)
    assert C_is.consistent_type()==np.ndarray, "C must by of type ndarray"
    assert C.flags.c_contiguous, "blas_tools.dgemm may only be used with C-order contiguous data"
    
    test_rows = A.shape[2] if TRANSPOSE_A else A.shape[1]
    test_cols = B.shape[1] if TRANSPOSE_B else B.shape[2]
    assert C_is.shape == (max([A.shape[0], B.shape[0]]), test_rows, test_cols), "A,B are incompatible with C"
    
    return fn_return_val



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
    print "Case 1: num_P == num_x"
    x = np.random.rand(num_elements, num_dims)
    P = np.random.rand(num_elements, num_dims, num_rows)
    # Using numpy only
    np_result = np.array([np.dot(P[i], x[i]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x))
    # Using dgemv with lists
    list_dgemv_result = np.array(dgemv(np.array(P).tolist(), np.array(x).tolist()))
    
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    max_err2 = np.max(np.abs(np_result - list_dgemv_result))
    
    print "Maximum error = ", max_err1, " for np_arrays"
    print "Maximum error = ", max_err2, " for lists"
    
    
    print "Case 2: num_P == 1"
    x = np.random.rand(num_elements, num_dims)
    P = np.random.rand(1, num_dims, num_rows)
    # Using numpy only
    np_result = np.array([np.dot(P[0], x[i]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x))
    # Using dgemv with lists
    list_dgemv_result = np.array(dgemv(np.array(P).tolist(), np.array(x).tolist()))
    
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    max_err2 = np.max(np.abs(np_result - list_dgemv_result))
    
    print "Maximum error = ", max_err1, " for np_arrays"
    print "Maximum error = ", max_err2, " for lists"
    
    
    print "Case 3: num_x == 1"
    x = np.random.rand(1, num_dims)
    P = np.random.rand(num_elements, num_dims, num_rows)
    # Using numpy only
    np_result = np.array([np.dot(P[i], x[0]) for i in range(num_elements)]).squeeze()
    # Using dgemv with numpy style arrays
    np_dgemv_result = dgemv(np.array(P), np.array(x))
    # Using dgemv with lists
    list_dgemv_result = np.array(dgemv(np.array(P).tolist(), np.array(x).tolist()))
    
    max_err1 = np.max(np.abs(np_result - np_dgemv_result))
    max_err2 = np.max(np.abs(np_result - list_dgemv_result))
    
    print "Maximum error = ", max_err1, " for np_arrays"
    print "Maximum error = ", max_err2, " for lists"
    