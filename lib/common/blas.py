#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       blas_tools.py
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

import numpy as np
import copy
from scipy import weave
import __blas_c_code__ as __c_code__
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

def SET_DEBUG(bool_value):
    DEBUG = bool_value
    if not DEBUG:
        print "Warning, blas debugging is disabled."
    return

__blas_fns_list__ = ["ddot", "dnrm2", "dasum", "idamax", "daxpy", "dscal", 
                     "dcopy", "dgemv", "dtrmv", "dtrsv", "dsymv", "dger", 
                     "dsyr", "dgemm", "dsymm", "dsyrk", "dgetrf", "dgetrs", 
                     "dgetrsx", "dgetri", "dgetrdet", "dpotrf", "dpotrs", 
                     "dpotrsx", "dpotri", "dtrtri", "symmetrise", "mktril", 
                     "mktriu"]

def blas_weaver(subroutine_string):
    subroutine = getattr(__c_code__, subroutine_string)
    compile_args = getattr(__c_code__, "EXTRA_COMPILE_ARGS", [])
    compile_args += getattr(subroutine, "extra_compile_args", [])
    
    fn_string = "__c_code__."+subroutine_string
    exec_string = ("weave.inline(" + 
        fn_string+".code, " + 
        fn_string+".python_vars" + 
        ", libraries=" + fn_string+".libraries" + 
        ", support_code=" + fn_string+".support_code" + 
        ", extra_compile_args=" + str(compile_args) + 
        ", verbose=1" + ")" )
    return exec_string

blas_exec_cmd = dict()                 
for blas_fn in __blas_fns_list__:
    blas_exec_cmd[blas_fn] = blas_weaver(blas_fn)
del blas_fn


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


def _np_generate_(np_function, itis):
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
    return _np_generate_(np.empty, itis)

def zeros(itis):
    return _np_generate_(np.zeros, itis)

def ones(itis):
    return _np_generate_(np.ones, itis)

    
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
    """Returns the dot product of two vectors.
    
    Given two vectors x and y, return transpose(x)*y
    
    Parameters
    ----------
    *x, y*: two dimensional ndarray, shape (Mx, N) and (My, N), Mx==My or \
            Mx==1 or My==1, where each row corresponds to a single \
            vector. Elements of x and y must have the same dimensions N. 
    
    Returns
    -------
    *z:* ndarray corresponding to the dot product of the elements of x and y.
    
    """
    if DEBUG:
        assert_valid_vector(x, "x")
        assert_valid_vector(y, "y")
        assert (x.shape[0] in [1, y.shape[0]]) and (y.shape[0] in [1, x.shape[0]]), NUM_ELEM_MISMATCH
        assert (x.shape[1] == y.shape[1]), V_V_DIM_MISMATCH
    xt_dot_y = np.zeros(max([x.shape[0], y.shape[0]]), dtype=float)
    exec blas_exec_cmd["ddot"]
    return xt_dot_y
    
    
def dnrm2(x):
    """
    Returns the L2 norm of a vector x.
    """
    if DEBUG:
        assert_valid_vector(x, "x")
    
    nrm2 = np.zeros(x.shape[0], dtype=float)
    exec blas_exec_cmd["dnrm2"]
    return nrm2


def dasum(x):
    """
    Returns the sum of elements in a vector x.
    """
    if DEBUG:
        assert_valid_vector(x)
    
    asum = np.zeros(x.shape[0], dtype=float)
    exec blas_exec_cmd["dasum"]
    return asum
    
    
def idamax(x):
    """
    Returns the index corresponding to the maximum value in a vector x.
    """
    if DEBUG:
        assert_valid_vector(x, "x")
    
    max_idx = np.empty(x.shape[0], dtype=int)
    exec blas_exec_cmd["idamax"]
    return max_idx
    
    
def daxpy(alpha, x, y=None):
    """
    Evaluates y = a*x + y
    If y is not specified, the function returns the value a*x as a new vector.
    """
    if DEBUG:
        assert_valid_vector(x, "x")
    
    fn_return_val = None
    if y == None:
        y = np.zeros(x.shape)
        fn_return_val = y
    if DEBUG:
        assert_valid_vector(y, "y")
        assert (x.shape[0]==y.shape[0]) or (x.shape[0]==1), NUM_ELEM_MISMATCH
        assert (x.shape[1]==y.shape[1]), V_V_DIM_MISMATCH
    
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    
    if DEBUG:
        assert alpha.shape[0] in [1, x.shape[0]], V_V_DIM_MISMATCH
    
    exec blas_exec_cmd["daxpy"]
    return fn_return_val
    
    
def dscal(alpha, x):
    """
    Multiply the vector x by alpha in-place.
    """
    if DEBUG:
        assert_valid_vector(x)
    
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    
    if DEBUG:
        assert len(alpha) in (1, x.shape[0]), V_V_DIM_MISMATCH
    
    exec blas_exec_cmd["dscal"]
    
    
def dcopy(x, y):
    """
    Performs the operation y=x in-place.
    """
    if DEBUG:
        assert_valid_vector(x, "x")
        assert_valid_vector(y, "y")
        assert x.shape==y.shape, NUM_ELEM_MISMATCH+V_V_DIM_MISMATCH
    exec blas_exec_cmd["dcopy"]
    
    
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
def dgemv(A, x, TRANSPOSE_A=False, alpha=np.array([1.0]), 
          beta=np.array([1.0]), y=None):
    """
    Performs in-place matrix vector multiplication
    dgemv --
    y = alpha*A*x + beta*y OR 
    y = alpha*A**T*x + beta*y if TRANSPOSE_A==True.
    If y is not specified, a new vector is returned.
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
    
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    if not isinstance(beta, np.ndarray):
        beta = np.array([beta], dtype=float)
    if DEBUG:
        assert len(alpha) in [1, x.shape[0], A.shape[0]], NUM_ELEM_MISMATCH
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), NUM_ELEM_MISMATCH
        if not TRANSPOSE_A:
            assert A.shape[2] == x.shape[1], M_V_DIM_MISMATCH
        else:
            assert A.shape[1] == x.shape[1], M_V_DIM_MISMATCH
        assert len(beta) in [1, y.shape[0]], NUM_ELEM_MISMATCH
    exec blas_exec_cmd["dgemv"]
    return fn_return_val
    
    
def dtrmv(UPLO, A, x, TRANSPOSE_A=False):
    """
    Performs in-place matrix vector multiplication for triangular matrix A.
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
    
    exec blas_exec_cmd["dtrmv"]
    

def dtrsv(UPLO, A, x, TRANSPOSE_A=False):
    """
    Solve the set of linear equations Ax=b in-place.
    On entry, x contains b. This is overwritten with the values for x.
    dtrsv -- Solve A*x = b if TRANSPOSE_A=False   or
    A**T*x = b if TRANSPOSE_A=True.
    A is nxn triangular
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
    
    exec blas_exec_cmd["dtrsv"]
    

def dsymv(UPLO, A, x, alpha=np.array([1.0]), beta=np.array([1.0]), y=None):
    """
    Performs in-place matrix vector multiplication for the symmetric matrix A.
    dsymv -- y = alpha*A*x + beta*y, A is symmetric
    if y is not specified, a new vector is returned.
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
    
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    if not isinstance(beta, np.ndarray):
        beta = np.array([beta], dtype=float)
    if DEBUG:
        assert len(alpha) in [1, x.shape[0], A.shape[0]], NUM_ELEM_MISMATCH
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1) or (A.shape[0]==1), NUM_ELEM_MISMATCH
        assert A.shape[2] == x.shape[1], M_V_DIM_MISMATCH
        assert len(beta) in [1, y.shape[0]], NUM_ELEM_MISMATCH
    
    exec blas_exec_cmd["dsymv"]
    return fn_return_val
    
    
def dger(x, y, alpha=np.array([1.0]), A=None):
    """
    Perform the general rank 1 operation in-place.
    dger -- rank 1 operation, A = alpha*x*y**T + A
    if A is not specified, it is assumed to be 0 and a new matrix is returned.
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
    
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        #assert_valid_vector(alpha, "alpha")
        assert len(alpha) in [1, x.shape[0], y.shape[0]], NUM_ELEM_MISMATCH
        assert (x.shape[0]==y.shape[0]) or (x.shape[0]==1) or (y.shape[0]==1), NUM_ELEM_MISMATCH
    
    exec blas_exec_cmd["dger"]
    return fn_return_val
    
    
def dsyr(UPLO, x, alpha=np.array([1.0]), A=None):
    """
    Perform the symmetric rank 1 operation in-place.
    dsyr -- symmetric rank 1 operation
    A = alpha*x*x**T + A,  A is nxn symmetric
    If A is not specified, a new matrix is returned.
    """
    if DEBUG:
        assert_valid_vector(x, "x")
    
    fn_return_val = None
    if A==None:
        A = np.zeros(x.shape + (x.shape[1],), dtype=float)
        fn_return_val = A
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        assert_valid_matrix(A)
        assert (x.shape[0]==A.shape[0]) or (x.shape[0]==1), NUM_ELEM_MISMATCH
        assert len(alpha) in [1, x.shape[0]], NUM_ELEM_MISMATCH
        assert A.shape[1]==A.shape[2]==x.shape[1], M_V_DIM_MISMATCH
    
    exec blas_exec_cmd["dsyr"]
    return fn_return_val


##
## LEVEL 3 BLAS
##

#def dgemm(A, B, alpha=1.0, C=None, beta=1.0, TRANSPOSE_A=False, TRANSPOSE_B=False):
def dgemm(A, B, TRANSPOSE_A=False, TRANSPOSE_B=False, alpha=np.array([1.0]), 
          beta=np.array([1.0]), C=None):
    """
    Performs general matrix matrix multiplication in-place
    dgemm -- matrix-matrix operation, 
    C := alpha*op( A )*op( B ) + beta*C,
    where  op( X ) is one of op( X ) = X   or   op( X ) = X**T
    If C is not specified, a new matrix is returned.
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_matrix(B, "B")
        assert A.shape[0] == B.shape[0] or A.shape[0]==1 or B.shape[0] == 1, NUM_ELEM_MISMATCH
        #assert A.shape[0] in [1, B.shape[0]] and B.shape[0] in [1, A.shape[0]], NUM_ELEM_MISMATCH
    
    A_dims = A.shape[1:]
    B_dims = B.shape[1:]
    if TRANSPOSE_A:
        A_dims = (A_dims[1], A_dims[0])
    if TRANSPOSE_B:
        B_dims = (B_dims[1], B_dims[0])
            
    fn_return_val = None
    if C==None:
        C = np.zeros((max([A.shape[0], B.shape[0]]), A_dims[0], B_dims[1]), dtype=float)
        fn_return_val = C
    
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    if not isinstance(beta, np.ndarray):
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
    
    exec blas_exec_cmd["dgemm"]
    #weave.inline(__c_code__.dgemm.code, __c_code__.dgemm.python_vars, 
    #             libraries=__c_code__.dgemm.libraries, 
    #             support_code=__c_code__.dgemm.support_code, 
    #             extra_compile_args=['-O3 -g -fopenmp'])
    return fn_return_val


def dsymm(A, B, UPLO, SIDE='l', alpha=np.array([1.0]), 
          beta=np.array([1.0]), C=None):
    """
    Performs in-place matrix matrix multiplication for symmetric matrix A.
    dsymm -- matrix-matrix operation, A is symmetric, B and C are mxn
    C := alpha*A*B + beta*C, if SIDE='l' or
    C := alpha*B*A + beta*C, if SIDE='r'
    UPLO = 'l' or 'u'
    if C is not specified, a new matrix is returned.
    """
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    if DEBUG:
        assert SIDE in ['l', 'L', 'r', 'R'], "SIDE must be one of ['l', 'L', 'r', 'R']"
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'L', 'u', 'U']"
        assert_valid_matrix(A, "A")
        assert_valid_matrix(B, "B")
        assert A.shape[0] in [1, B.shape[0]] and B.shape[0] in [1, A.shape[0]], NUM_ELEM_MISMATCH
        assert len(alpha) in [1, A.shape[0], B.shape[0]], NUM_ELEM_MISMATCH
            
    A_dims = A.shape[1:]
    B_dims = B.shape[1:]
    
    if not isinstance(beta, np.ndarray):
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
    
    exec blas_exec_cmd["dsymm"]
    return fn_return_val


def dsyrk(UPLO, A, TRANSPOSE_A=False, alpha=np.array([1.0]), 
          beta=np.array([1.0]), C=None):
    """
    Perform in-place symmetric rank k operation.
    dsyrk -- symmetric rank k operation, C is symmetric
    C := alpha*A*A**T + beta*C  if TRANSPOSE_A=False or
    C := alpha*A**T*A + beta*C  if TRANSPOSE_A=True
    If C is not specified, a new matrix is returned.
    """
    A_dims = A.shape[1:]
    if TRANSPOSE_A:
        A_dims = (A_dims[1], A_dims[0])
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha], dtype=float)
    
    if DEBUG:
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'L', 'u', 'U']"
        assert_valid_matrix(A, "A")
        assert alpha.shape[0] in [1, A.shape[0]], NUM_ELEM_MISMATCH
    
    fn_return_val = None
    if C==None:
        C = np.zeros((A.shape[0], A_dims[0], A_dims[1]), dtype=float)
        fn_return_val = C
    if not isinstance(beta, np.ndarray):
        beta = np.array([beta], dtype=float)
        
    if DEBUG:
        assert_valid_matrix(C)
        assert C.shape == (A.shape[0], A_dims[0], A_dims[0]), M_M_DIM_MISMATCH
        assert beta.shape[0] in [1, C.shape[0]], NUM_ELEM_MISMATCH
    
    exec blas_exec_cmd["dsyrk"]
    return fn_return_val


##
## LINEAR ALGEBRA
##

# LU decomposition
def dgetrf(A, INPLACE=False):
    """
    Performs LU decomposition on the matrix A.
    Returns (LU, ipiv, signum) if INPLACE=False, otherwise (ipiv, signum)
    ipiv contains the pivot points for use by dgetrs(), dgetri() and signum is
    used to compute the determinant by dgetrdet().
    
    The matrix A is decomposed as A = L*U and both L and U are stored in the 
    same matrix. L is obtained from the strictly lower part of the matrix and 
    placing 1s on the diagonal. U is obtained by taking the upper part of the
    matrix (including the diagonal). If INPLACE=False, a new matrix is 
    returned with the decomposition.
    
    The GSL is used to perform the decomposition. See the documentation for 
    further details.
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
    ipiv = np.zeros(A.shape[0:2], dtype=int)
    signum = np.zeros(A.shape[0])
    if not INPLACE:
        A = A.copy()
        fn_return_val = A, ipiv, signum
    else:
        fn_return_val = ipiv, signum
    
    exec blas_exec_cmd["dgetrf"]
    return fn_return_val


# Solve Ax=B using LU decomposition
def dgetrs(LU, ipiv, b, x=None):
    """
    Solves the set of linear equations Ax=b using the LU decomposition 
    generated by dgetrf()
    if x is not specified, a new vector is returned.
    """
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
    
    exec blas_exec_cmd["dgetrs"]
    return fn_return_val


# Solve Ax=B using LU decomposition in place
def dgetrsx(LU, ipiv, b):
    """
    Solves the set of linear equations Ax=b in-place using the LU decomposition
    generated by dgetrf().
    On exit, the values in b are overwritten.
    """
    if DEBUG:
        assert_valid_matrix(LU, "LU")
        assert_valid_vector(ipiv, "ipiv")
        assert_valid_vector(b, "b")
        assert ipiv.shape[0] == LU.shape[0], NUM_ELEM_MISMATCH
        assert LU.shape[1] == ipiv.shape[1], M_M_DIM_MISMATCH
        assert LU.shape[0] in [1, b.shape[0]], NUM_ELEM_MISMATCH
        assert LU.shape[2] == b.shape[1], M_V_DIM_MISMATCH
    
    exec blas_exec_cmd["dgetrsx"]
    

def dgetri(LU, ipiv):
    """
    Computes the inverse of a matrix A in-place using the LU decomposition
    generated from dgetrf().
    """
    if DEBUG:
        assert_valid_matrix(LU, "LU")
        assert_valid_vector(ipiv, "ipiv")
        assert LU.shape[0] == ipiv.shape[0], NUM_ELEM_MISMATCH
        assert LU.shape[1] == ipiv.shape[1], M_M_DIM_MISMATCH
    invA = np.zeros(LU.shape, dtype=float)
    exec blas_exec_cmd["dgetri"]
    return invA


# Solve Ax = B
def dgesv(A, b, INPLACE=False, OVERWRITE_A=False):
    """
    Solves the set of linear equations Ax=b.
    Returns (x, A, ipiv, signum)
    If INPLACE=True, then b will be overwritten with the result.
    If OVERWRITE_A=True, then A will be overwritten with the LU decomposition.
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(b, "b")
        if INPLACE:
            assert A.shape[0] in [1, b.shape[0]]
    if not OVERWRITE_A:
        A = A.copy()
    ipiv, signum = dgetrf(A, INPLACE=True)
    if not INPLACE:
        x = dgetrs(A, ipiv, b)
    else:
        dgetrsx(A, ipiv, b)
        x = b
    fn_return_val = (x, A, ipiv, signum)
    return fn_return_val
    

def dgetrdet(LU, signum):
    """
    Returns the determinant for matrix A given the LU decomposition and signum.
    """
    if DEBUG:
        assert_valid_matrix(LU)
        assert signum.shape[0] == LU.shape[0], NUM_ELEM_MISMATCH
    det_vec = np.zeros(LU.shape[0])
    exec blas_exec_cmd["dgetrdet"]
    return det_vec
    
    
# Cholesky decomposition
def dpotrf(A, INPLACE=False):
    """
    Computes the Cholesky factorisation of a matrix A. The upper triangle of
    the factorisation contains the transpose of the lower triangle. If only the
    lower triangle is required, this must be performed on the result.
    if INPLACE=False, then a new matrix with the symmetrised Cholesky 
    decomposition is returned, otherwise A is overwritten.
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
    fn_return_val = None
    if not INPLACE:
        A = A.copy()
        fn_return_val = A
    error_occurred = 0
    exec blas_exec_cmd["dpotrf"]
    assert error_occurred==0, "Error occurred while computing Cholesky factorisation."
    return fn_return_val


# Solve using Cholesky decomposition
def dpotrs(cholA, b, x=None):
    """
    Solve the set of linear equations Ax=b using the cholesky decomposition
    generated by dpotrf().
    If x is not specified, a new vector will be returned.
    """
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
    error_occurred = 0
    exec blas_exec_cmd["dpotrs"]
    assert error_occurred==0, "Error occurred while computing Cholesky factorisation."
    return fn_return_val


# Solve Ax=B using LU decomposition in place
def dpotrsx(cholA, b):
    """
    Solve the set of linear equations Ax=b in-place using the Cholesky 
    decomposition generated by dportf().
    On exit, the vector b is overwritten with the result.
    """
    if DEBUG:
        assert_valid_matrix(cholA, "cholA")
        assert_valid_vector(b, "b")
        assert cholA.shape[0] in [1, b.shape[0]], NUM_ELEM_MISMATCH
        assert cholA.shape[2] == b.shape[1], M_V_DIM_MISMATCH
    error_occurred = 0
    exec blas_exec_cmd["dpotrsx"]
    assert error_occurred==0, "Error occurred while computing Cholesky factorisation."


# Compute inverse using Cholesky factorisation from dpotrf
def dpotri(A, INPLACE=False):
    """
    Compute the inverse of the matrix A given its Cholesky decomposition 
    computed by dpotrf()
    If INPLACE=False, a new matrix with the inverse is returned, otherwise
    A is overwritten with its inverse.
    """
    if DEBUG:
        assert_valid_matrix(A)
    fn_return_val = None
    if not INPLACE:
        A = A.copy()
        fn_return_val = A
    error_occurred = 0
    exec blas_exec_cmd["dpotri"]
    assert error_occurred==0, "Error occurred while computing Cholesky factorisation."
    return fn_return_val
    

# Solve for positive definite matrix
def dposv(A, b, INPLACE=False, OVERWRITE_A=False):
    """
    Solve the set of linear equations for the symmetric matrix A and vector b
    
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert_valid_vector(b, "b")
        if INPLACE:
            assert A.shape[0] in [1, b.shape[0]]
    if not OVERWRITE_A:
        A = A.copy()
    dpotrf(A, INPLACE=True)
    
    if not INPLACE:
        x = dpotrs(A, b)
    else:
        dpotrsx(A, b)
        x = b
    fn_return_val = (x, A)
    return fn_return_val
    

# Compute inverse of a triangular matrix using dtrtri
def dtrtri(A, UPLO, INPLACE=False):
    """
    Computes the inverse of a triangular matrix.
    If INPLACE=False, a new matrix with the inverse is returned, otherwise A
    is overwritten.
    """
    if DEBUG:
        assert_valid_matrix(A)
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'L', 'u', 'U']"
    fn_return_val = None
    if not INPLACE:
        A = A.copy()
        fn_return_val = A
    error_occurred = 0
    exec blas_exec_cmd["dtrtri"]
    assert error_occurred==0, "Error occurred while computing Cholesky factorisation."
    return fn_return_val
    
    
def inverse(A, FACTORISATION, INPLACE=False):
    """
    Compute the inverse of a matrix using (C)holesky of (L)U decomposition.
    If INPLACE=False, a new matrix with the inverse is returned, otherwise A
    is overwritten.
    """
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
    """
    Given an upper or lower triangular matrix, make the matrix symmetric by 
    copying the values to the lower or upper triangle. This operation is 
    performed in-place.
    """
    if DEBUG:
        assert_valid_matrix(A, "A")
        assert A.shape[1]==A.shape[2], "A must be symmetric"
        assert type(UPLO)
        assert UPLO in ['l', 'L', 'u', 'U'], "UPLO must be one of ['l', 'u', 'L', 'U']"
    exec blas_exec_cmd["symmetrise"]
    #weave.inline(blas_c_code.symmetrise.code, 
    #             blas_c_code.symmetrise.python_vars, 
    #             libraries=blas_c_code.symmetrise.libraries,
    #             support_code=blas_c_code.symmetrise.support_code,
    #             extra_compile_args=blas_c_code.EXTRA_COMPILE_ARGS)
    
def mktril(A):
    """
    Given a matrix A, zero the strictly upper triangular portion of the matrix.
    """
    #if DEBUG:
    #    assert_valid_matrix(A, "A")
    #exec blas_exec_cmd["mktril"]
    A_dims = A.shape[-2:]
    mask = np.tril(np.ones(A_dims))
    A *= mask
    
def mktriu(A):
    """
    Given a matrix A, zero the strictly upper triangular portion of the matrix.
    """
    #if DEBUG:
    #    assert_valid_matrix(A, "A")
    #exec blas_exec_cmd["mktriu"]
    A_dims = A.shape[-2:]
    mask = np.triu(np.ones(A_dims))
    A *= mask


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
    x = np.random.rand(num_elements, num_dims)
    A = np.random.rand(num_elements, num_dims, num_dims)
    dsyr('l', x, np.random.rand(1), A)
    symmetrise(A, 'l')


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



def blas_init():
    A = np.random.rand(1, 2, 2) + 10*np.eye(2)
    B = np.random.rand(1, 2, 2) + 10*np.eye(2)
    C = np.random.rand(1, 2, 2) + 10*np.eye(2)
    x = np.random.rand(1, 2)
    y = np.random.rand(1, 2)
    UPLO = 'l'
    TR_A = False
    TR_B = False
    SIDE = 'l'
    alpha = np.array([1.0])
    beta = np.array([1.0])
    
    ddot(x, y)
    dnrm2(x)
    dasum(x)
    idamax(x)
    daxpy(alpha, x, y)
    dscal(alpha, x)
    dcopy(x, y)
    dgemv(A, x, TR_A, alpha, beta, y)
    dtrmv(UPLO, A, x, TR_A)
    dtrsv(UPLO, A, x, TR_A)
    dsymv(UPLO, A, x, alpha, beta, y)
    dger(x, y, alpha, A)
    dsyr(UPLO, x, alpha, A)
    dgemm(A, B, TR_A, TR_B, alpha, beta, C)
    dsymm(A, B, UPLO, SIDE, alpha, beta, C)
    dsyrk(UPLO, A, TR_A, alpha, beta, C)
    LU, ipiv, signum = dgetrf(A)
    dgetrs(LU, ipiv, y, x)
    dgetrsx(LU, ipiv, y)
    dgetri(LU, ipiv)
    dgesv(A, y)
    dgetrdet(LU, signum)
    cholA = dpotrf(A)
    dpotrs(cholA, y, x)
    dpotrsx(cholA, y)
    dpotri(A)
    dtrtri(A, UPLO, INPLACE=False)
    symmetrise(A, UPLO)
    mktril(A)
    mktriu(A)
    del A, B, C, x, y, UPLO, TR_A, TR_B, SIDE, alpha, beta, LU, ipiv, signum

print "initialising blas wrapper"
blas_init()
