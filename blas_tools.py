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
        elif all(_type==list):
            return list
        elif (self.type[0] == list) and np.all([self.type[i] == np.ndarray for i in range(1,len(self.type))]):
            return LIST_OF_NDARRAY
        else:
            return None

def whatisit(arr):
    if type(arr) is np.ndarray:
        if arr.dtype == object:
            itis = ITIS(False, [np.ndarray], len(arr.shape), arr.shape, arr.dtype)
            return itis
        
        return ITIS(True, [np.ndarray], len(arr.shape), arr.shape, arr.dtype)
    elif type(arr) is list:
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
    

def dgemv(A, x, alpha=1.0, beta=0.0, y=None, TRANSPOSE_A=False):
    A_is = whatisit(A)
    x_is = whatisit(x)
    A_consistent_type = A_is.consistent_type()
    x_consistent_type = x_is.consistent_type()
    
    assert A_consistent_type == x_consistent_type, "A and x must have consistent types"
    if not (y == None):
        y_is = whatisit(y)
        y_consistent_type = y_is.consistent_type
        assert A_consistent_type == y_consistent_type, "A, x and y must have consistent types"
    
    assert A_consistent_type in [np.ndarray, list, LIST_OF_NDARRAY], "Valid type must be used"
    y_is = copy.deepcopy(x_is)
    if not (A_is.shape[0] == x_is.shape[0]):
        y_is.shape = ((max([A_is.shape[0],x_is.shape[0]]),) + x_is.shape[1:])
    
    x_shape = x_is.shape
    A_shape = A_is.shape
    
    if A_consistent_type == list:
        y = zeros(y_is)
        weave.inline(blas_tools_c_code.ldgemv.code, 
                     blas_tools_c_code.ldgemv.python_vars, 
                     libraries=blas_tools_c_code.ldgemv.libraries,
                     support_code=blas_tools_c_code.ldgemv.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
        
    elif A_consistent_type == np.ndarray:
        y = zeros(y_is)
        weave.inline(blas_tools_c_code.npdgemv.code, 
                     blas_tools_c_code.npdgemv.python_vars, 
                     libraries=blas_tools_c_code.npdgemv.libraries,
                     support_code=blas_tools_c_code.npdgemv.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
                     
    elif A_consistent_type == LIST_OF_NDARRAY:
        y_is.consistent_type = np.ndarray
        y = zeros(y_is)
        weave.inline(blas_tools_c_code.npdgemv.code, 
                     blas_tools_c_code.npdgemv.python_vars, 
                     libraries=blas_tools_c_code.npdgemv.libraries,
                     support_code=blas_tools_c_code.npdgemv.support_code,
                     extra_compile_args=blas_tools_c_code.EXTRA_COMPILE_ARGS)
        y = [y[i] for i in range(len(y))]
    return y
    
    
def test_dgemv(num_elements=1000, num_dims=4):
    print "Case 1: num_P == num_x"
    x, mu, P = phdmisctools.test_data(num_elements, num_dims)
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
    x, mu, P = phdmisctools.test_data(num_elements, num_dims)
    P = [P[0]]
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
    x, mu, P = phdmisctools.test_data(num_elements, num_dims)
    x = [x[0]]
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
    