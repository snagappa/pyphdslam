#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       phdmisctools.py
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
from scipy import weave
import python_c_code
reload(python_c_code)


COMPILER = 'gcc'
FORCE_RECOMPILE = 0
EXTRA_COMPILE_ARGS = ['-O2']
VERBOSE = 1


def dummyfunc(*args): pass


def test_data(sz_n=1, sz_x=4):
    x = [np.random.rand(sz_x) for i in range(sz_n)]
    mu = [np.random.randn(sz_x) for i in range(sz_n)]
    sigma = []
    for i in range(sz_n):
        r = np.random.rand(sz_x)
        sigma.append(np.matrix(r).T*r + (np.eye(sz_x)*(1+np.random.rand(1))))
    #sigma = [np.matrix(r).T*r + (np.eye(sz_x)*(1+np.random.rand())) for r in np.random.rand(sz_x) for i in range(sz_n)]
    return x, mu, sigma


def get_resample_index(weights, nparticles=-1):
    try:
        resampled_indices = get_resample2(weights, nparticles)
        return resampled_indices
    except (RuntimeError, TypeError, NameError):
        weights = np.array(weights, dtype=np.float64)
        normfac = weights.sum()
        weights = weights/normfac
        wt_cdf = weights.cumsum()
        
        if nparticles==-1:
            nparticles = len(weights)
        
        resampled_indices = list(np.empty(nparticles))
        #resampled_indices = [0]*nparticles
        
        float_nparticles = np.float64(nparticles)
        u1 = np.random.uniform()/float_nparticles
        uj = u1 + np.arange(nparticles)/float_nparticles
        array_cur_loc = 0;
        for j in np.arange(nparticles):
            while wt_cdf[array_cur_loc] < uj[j]:
                array_cur_loc += 1
            resampled_indices[j] = array_cur_loc
        return resampled_indices


def get_resample1(weights, nparticles=-1):
    weights = np.array(weights, dtype=np.float64)
    normfac = weights.sum()
    weights = weights/normfac
    wt_cdf = weights.cumsum()
    
    if nparticles==-1:
        nparticles = len(weights)
    
    resampled_indices = np.empty(nparticles, dtype=np.int)
    
    u1 = np.random.uniform()/nparticles
    
    ccode = python_c_code.resample_code1()
    weave.inline(ccode.code, ccode.python_vars, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    return resampled_indices.tolist()
    
    
def get_resample2(weights, nparticles=-1):
    weights = np.array(weights, dtype=np.float64)
    len_weights = len(weights)
    
    if nparticles==-1:
        nparticles = len(weights)
    
    resampled_indices = np.empty(nparticles, dtype=np.int)
    wt_cdf = np.empty(weights.shape, dtype=np.float64)
    u1 = np.random.uniform()/nparticles
    
    ccode = python_c_code.resample_code2()
    weave.inline(ccode.code, ccode.python_vars, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    return resampled_indices.tolist()
    


def logmvnpdf_generic(x, mu, sigma):
    num_x = len(x)
    num_mu = len(mu)
    num_sigma = len(sigma)
    
    # Compute the residual
    if num_x == 1:
        Residual = [x[0]-mu[i] for i in range(num_mu)]
        num_residuals = num_mu
    elif num_mu == 1:
        Residual = [x[i]-mu[0] for i in range(num_x)]
        num_residuals = num_x
    else:
        Residual = [x[i]-mu[i] for i in range(num_x)]
        num_residuals = num_x
        
    if num_sigma == 1:
        k = np.float64(sigma[0].shape[0])
        sigma_inv = sigma[0].getI()
        log2pidet = -(k/2)*np.log(2*np.pi)-0.5*np.log(np.linalg.det(sigma[0]))
        # Evaluate the log-likelihood
        #LogLikelihood = [log2pidet - 0.5*Residual[i].T*sigma_inv*Residual[i] for i in range(num_residuals)]
        LogLikelihood = np.array([log2pidet - 0.5*np.ravel(np.dot(Residual[i], np.dot(sigma_inv, Residual[i]).T))[0] for i in range(num_residuals)])
    else:
        sigma_inv = [sigma[i].getI() for i in range(num_sigma)]
        log2pi = np.log(2*np.pi)
        log2pidet = [-(np.float64(sigma[i].shape[0])/2.0)*log2pi-0.5*np.log(np.linalg.det(sigma[i])) for i in range(num_sigma)]
        # Evaluate the log-likelihood
        if num_residuals == 1:
            #LogLikelihood = [log2pidet[i] - 0.5*Residual[0].T*sigma_inv[i]*Residual[0] for i in range(num_sigma)]
            LogLikelihood = np.array([log2pidet[i] - 0.5*np.ravel(np.dot(Residual[0], np.dot(sigma_inv[i], Residual[0]).T)) for i in range(num_sigma)])
                                         
        else:
            #LogLikelihood = [log2pidet[i] - 0.5*Residual[i].T*sigma_inv[i]*Residual[i] for i in range(num_sigma)]
            LogLikelihood = np.array([log2pidet[i] - 0.5*np.ravel(np.dot(Residual[i], np.dot(sigma_inv[i], Residual[i]).T)) for i in range(num_sigma)])
            
    
    return LogLikelihood.flatten()
    


def log_mvnpdf(x, mu, sigma):
    num_x = len(x)
    num_mu = len(mu)
    num_sigma = len(sigma)
    state_dimensions = len(mu[0])
    
    #residuals, num_residuals = _compute_residuals_(x, mu)
    residuals = [np.empty(state_dimensions, dtype=float).tolist() for i in range(np.max([num_mu, num_x]))]
    num_residuals = [0]
    
    log_likelihood = np.empty(np.max([num_x, num_mu, num_sigma]), dtype=float)
    sigmainv = np.empty((state_dimensions,state_dimensions)).tolist()
    determinant = [0]
    a_inv_b = np.empty(state_dimensions).tolist()
    pi = np.pi
    
    ccode = python_c_code.log_mvnpdf()
    weave.inline(ccode.code, ccode.python_vars, 
                 support_code=ccode.support_code, libraries=ccode.libs, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    return log_likelihood#, residuals
    

def mvnpdf(x, mu, sigma, LOGPDF=False):
    num_x = len(x)
    num_mu = len(mu)
    num_sigma = len(sigma)
    state_dimensions = len(mu[0])
    
    #residuals, num_residuals = _compute_residuals_(x, mu)
    residuals = [np.empty(state_dimensions, dtype=float).tolist() for i in range(np.max([num_mu, num_x]))]
    num_residuals = [0]
    
    likelihood = np.empty(np.max([num_x, num_mu, num_sigma]), dtype=float)
    sigmainv = np.empty((state_dimensions,state_dimensions)).tolist()
    determinant = [0]
    a_inv_b = np.empty(state_dimensions).tolist()
    pi = np.pi
    
    ccode = python_c_code.mvnpdf()
    weave.inline(ccode.code, ccode.python_vars, 
                 support_code=ccode.support_code, libraries=ccode.libs, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    return likelihood
    
    
    
def _compute_residuals_(x, mu):
    num_x = len(x)
    num_mu = len(mu)
    state_dimensions = len(mu[0])
    if num_x == 1:
        residuals = [np.empty(state_dimensions, dtype=np.float64).tolist() for i in range(num_mu)]
    else:
        residuals = [np.empty(state_dimensions, dtype=np.float64).tolist() for i in range(num_x)]
    num_residuals = [0];
    
    ccode = python_c_code.compute_residuals()
    weave.inline(ccode.code, ccode.python_vars, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    return residuals, num_residuals[0]
    
    

def imul(vec, mul_fac):
    vec_len = len(vec)
    ccode = python_c_code.np_ndarray_imul()
    weave.inline(ccode.code, ccode.python_vars,
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    
    
def mahalanobis(x, P, y):
    num_x = len(x)
    num_y = len(y)
    num_P = len(P)
    
    state_dimensions = len(x[0])
    
    residuals = [np.empty(state_dimensions, dtype=float).tolist() for i in range(np.max([num_x, num_y]))]
    num_residuals = [0]
    
    mahalanobis_dist = np.empty(np.max([num_x, num_y, num_P]), dtype=float)
    sigmainv = np.empty((state_dimensions,state_dimensions)).tolist()
    a_inv_b = np.empty(state_dimensions).tolist()
    
    ccode = python_c_code.mahalanobis()
    weave.inline(ccode.code, ccode.python_vars, 
                 support_code=ccode.support_code, libraries=ccode.libs, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    return mahalanobis_dist


def cholesky(Amat):
    #A = [Amat]
    #invA = [np.matrix(np.empty(Amat.shape))]
    A = Amat.tolist()
    cholA = np.zeros(Amat.shape).tolist()
    dims = Amat.shape[0]
    
    ccode = python_c_code.cholesky()
    weave.inline(ccode.code, ccode.python_vars, 
                 support_code=ccode.support_code, libraries=ccode.libs, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
                 #type_converters=weave.converters.blitz)
    return cholA
    
    
def batch_cholesky(A):
    num_A = len(A)
    dims = len(A[0])
    A = np.array(A).tolist()
    cholA = np.zeros((num_A,dims,dims)).tolist()
    
    ccode = python_c_code.batch_cholesky()
    weave.inline(ccode.code, ccode.python_vars, 
                 support_code=ccode.support_code, libraries=ccode.libs, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    
    return cholA
    
    
def inverse(A):
    num_A = len(A)
    dims = len(A[0])
    A = np.array(A).tolist()
    invA = np.empty((num_A,dims,dims)).tolist()
    
    ccode = python_c_code.inverse()
    weave.inline(ccode.code, ccode.python_vars, 
                 support_code=ccode.support_code, libraries=ccode.libs, 
                 compiler=COMPILER, force=FORCE_RECOMPILE, 
                 extra_compile_args=EXTRA_COMPILE_ARGS, verbose=VERBOSE)
    
    return invA
    
    
def m_times_x(matrix_m, vector_x):
    pass


def xt_times_m(vector_x, matrix_m):
    pass


def m_times_m(matrix_m, matrix_n):
    pass


def x_times_xt(vector_x, vector_y):
    pass


def xt_times_x(vector_x, vector_y):
    pass

    
def merge_states(wt, x, P):
    num_x = len(x)
    merged_wt = wt.sum()
    merged_x = np.sum([x[i]*wt[i] for i in range(num_x)], 0)/merged_wt
    residuals, num_residuals = _compute_residuals_(x, [merged_x])
    merged_P = sum([wt[i]*(P[i] + np.dot(np.matrix(residuals[i]).T, np.matrix(residuals[i]))) for i in range(num_x)], 0)/merged_wt
    return merged_wt, merged_x, merged_P
    
    
def delete_from_list(x, indices):
    #if type(indices) == type(list()):
    #    indices.sort(reverse=True)
    #else:
    #    indices = np.sort(indices)[::-1]
    indices.sort(reverse=True)
    [x.__delitem__(idx) for idx in indices]
    
    
def kalman_update(x, P, H, R, z=None):
    num_x = len(x)
    if len(H) == 1:
        h_idx = [0]*num_x
    else:
        h_idx = range(num_x)
    if len(R) == 1:
        r_idx = [0]*num_x
    else:
        r_idx = range(num_x)
        
    kalman_info = lambda:0
    # Evaluate inverse and determinant using Cholesky decomposition
    sqrt_S = [np.linalg.cholesky(H[h_idx[i]]*P[i]*H[h_idx[i]].T + R[r_idx[i]]) for i in range(num_x)]
    inv_sqrt_S = [sqrt_S[i].getI() for i in range(num_x)]
    
    det_S = [np.diag(sqrt_S[i]).prod()**2 for i in range(num_x)]
    inv_S = [inv_sqrt_S[i].T*inv_sqrt_S[i] for i in range(num_x)]
    
    # Kalman gain
    kalman_gain = [P[i]*H[h_idx[i]].T*inv_S[i] for i in range(num_x)]
    
    # Predicted observations
    pred_z = [np.dot(H[h_idx[i]],x[i]).A[0] for i in range(num_x)]
    
    # Update to new state if observations were received
    if not (z is None):
        residuals = _compute_residuals_(z, pred_z)
        #[z - pred_z[i] for i in range(num_x)]
        x_upd = [x[i] + np.dot(kalman_gain[i], residuals[i]).A[0] for i in range(num_x)]
    else:
        x_upd = x
        
    # Update covariance
    P_upd = [P[i] - (kalman_gain[i]*H[h_idx[i]]*P[i]) for i in range(num_x)]
    
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.pred_z = pred_z
    kalman_info.kalman_gain = kalman_gain
    
    return x_upd, P_upd, kalman_info
# Code taken from:
#   http://www.sagemath.org/doc/numerical_sage/weave.html
def weave_solve(a,b):
    n = len(a[0])
    x = np.array([0]*n,dtype=float)
    determinant = [1.0]
    
    support_code="""
    #include <stdio.h>
    extern "C" {
    //void dgesv_(int *size, int *flag,double* data,int*size,
    //            int*perm,double*vec,int*size,int*ok);
    //void dgesv_(int *, int *,double* ,int*,
    //            int*,double*,int*,int*);
    
    // Solve a linear ssystem of equations ax=b
    int dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, 
               double *b, int *ldb, int *info);
    
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
    
    // Solve a linear system of equations ax=b using LU decomposition of a
    int dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, 
                int *ipiv, double *b, int *ldb, int *info);
    }
    """

    code="""
        int i,j;
        double* a_c;
        double* b_c;
        int size;
        int flag;
        int* p;
        int ok;
        size=n;
        flag=1;
        char trans = 'N';
        int nrhs = 1;
        
        a_c= (double *)malloc(sizeof(double)*n*n);
        b_c= (double *)malloc(sizeof(double)*n);
        p = (int*)malloc(sizeof(int)*n);
        for(i=0;i<n;i++)
           {
           b_c[i]=b[i];
           for(j=0;j<n;j++)
             a_c[i*n+j]=a[i][j];
           }
           
        // Perform the LU decomposition
        dgetrf_(&size, &size, a_c, &size, p, &ok);
        
        // Determinant is the product of the diaganol
        double _det = 1.0;
        for(i=0;i<n;i++) {
            _det *= a_c[i*n+i];
            }
        determinant[0] = (double)_det;
        //std::cout << "det=" << determinant << std::endl;
        
        // Call the solve function
        //dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
        dgetrs_(&trans, &size, &nrhs, a_c, &size, p, b_c, &size, &ok);
        
        for(i=0;i<n;i++)
           x(i)=b_c[i];
        free(a_c);
        free(b_c);
        free(p);
    """
    
    libs=['lapack','blas']
    #dirs=['/media/sdb1/sage-2.6.linux32bit-i686-Linux']
    vars = ['a','b','x','n', 'determinant']
    #weave.inline(code,vars,support_code=support_code,libraries=libs,library_dirs=dirs,  \
    weave.inline(code,vars,support_code=support_code,libraries=libs,  \
    type_converters=weave.converters.blitz,compiler='gcc')
    return x, determinant
    
    original_code="""
        int i,j;
        double* a_c;
        double* b_c;
        int size;
        int flag;
        int* p;
        int ok;
        size=n;
        flag=1;
        a_c= (double *)malloc(sizeof(double)*n*n);
        b_c= (double *)malloc(sizeof(double)*n);
        p = (int*)malloc(sizeof(int)*n);
        for(i=0;i<n;i++)
           {
           b_c[i]=b[i];
           for(j=0;j<n;j++)
             a_c[i*n+j]=a[i][j];
           }
        dgesv_(&size,&flag,a_c,&size,p,b_c,&size,&ok);
        
        for(i=0;i<n;i++)
           x(i)=b_c[i];
        free(a_c);
        free(b_c);
        free(p);
    """
    
    

lapack_matrix_inverse = """
#include <cstdio>

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}

void inverse(double* A, int N)
{
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete IPIV;
    delete WORK;
}

int main(){

    double A [2*2] = {
        1,2,
        3,4
    };

    inverse(A, 2);

    printf("%f %f\n", A[0], A[1]);
    printf("%f %f\n", A[2], A[3]);

    return 0;
}
"""


lapack_example_code = """
    Understanding LAPACK calls in C   with a simple example
    // LAPACK test code
     
    #include<iostream>
    #include<vector>
     
    using namespace std;
    extern "C" void dgetrs(char *TRANS, int *N, int *NRHS, double *A,
                          int *LDA, int *IPIV, double *B, int *LDB, int *INFO );
     
    int main()
    {
        char trans = 'N';
        int dim = 2;    
        int nrhs = 1;
        int LDA = dim;
        int LDB = dim;
        int info;
     
        vector<double> a, b;
     
        a.push_back(1);
        a.push_back(1);
        a.push_back(1);
        a.push_back(-1);
     
        b.push_back(2);
        b.push_back(0);
     
        int ipiv[3];
     
     
        dgetrs(&trans, &dim, &nrhs, & *a.begin(), &LDA, ipiv, & *b.begin(), &LDB, &info);
     
     
        std::cout << "solution is:";    
        std::cout << "[" << b[0] << ", " << b[1] << ", " << "]" << std::endl;
        std::cout << "Info = " << info << std::endl;
     
        return(0);
    }
           
    // LAPACK test code
     
    #include <iostream>
    #include <vector>
    #include <Accelerate/Accelerate.h>
     
    using namespace std;
     
    int main()
    {
        char trans = 'N';
        int dim = 2;    
        int nrhs = 1;
        int LDA = dim;
        int LDB = dim;
        int info;
     
        vector<double> a, b;
     
        a.push_back(1);
        a.push_back(1);
        a.push_back(1);
        a.push_back(-1);
     
        b.push_back(2);
        b.push_back(0);
     
        int ipiv[3];
     
        dgetrf_(&dim, &dim, &*a.begin(), &LDA, ipiv, &info);
        dgetrs_(&trans, &dim, &nrhs, & *a.begin(), &LDA, ipiv, & *b.begin(), &LDB, &info);
     
     
        std::cout << "solution is:";    
        std::cout << "[" << b[0] << ", " << b[1] << ", " << "]" << std::endl;
        std::cout << "Info = " << info << std::endl;
     
        return(0);
    }

"""