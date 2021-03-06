#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       python_c_code.py
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
#   Some code copyright of Gunter Winkler, Konstantin Kutzkow (2005)

global_support_code = """
#include <stdio.h>

// For Cholesky decomposition
#include <cassert>
#include <limits>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "/home/snagappa/projects/python/phdslam/cholesky.hpp"

extern "C" {

// Solve a linear ssystem of equations ax=b
int dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, 
           double *b, int *ldb, int *info);

// LU decomoposition of a general matrix
void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

// Solve a linear system of equations ax=b using LU decomposition of a
int dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, 
            int *ipiv, double *b, int *ldb, int *info);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}


void solve_ax_eq_b(py::indexed_ref a, py::indexed_ref b, py::list x, int n, py::list determinant) {
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
    dgetrs_(&trans, &size, &nrhs, a_c, &size, p, b_c, &size, &ok);
    
    for(i=0;i<n;i++)
       x[i]=b_c[i];
    free(a_c);
    free(b_c);
    free(p);
}


void solve_ax_eq_b(py::indexed_ref a, py::indexed_ref b, py::list x, int n) {
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
    
    // Call the solve function
    dgetrs_(&trans, &size, &nrhs, a_c, &size, p, b_c, &size, &ok);
    
    for(i=0;i<n;i++)
       x[i]=b_c[i];
    free(a_c);
    free(b_c);
    free(p);
}


void get_inverse(py::indexed_ref a, int n, py::list determinant, py::list ainv) {
    int i,j;
    double* a_c;
    int size = n;
    char trans = 'N';
    int nrhs = 1;
    int *IPIV = new int[n+1];
    int LWORK = n*n;
    double *WORK = new double[LWORK];
    int INFO;

    a_c = (double *)malloc(sizeof(double)*n*n);
    //p = (int*)malloc(sizeof(int)*n);
    for(i=0;i<n;i++)
       {
       for(j=0;j<n;j++)
         a_c[i*n+j]=a[i][j];
       }
       
    // Perform the LU decomposition
    dgetrf_(&size, &size, a_c, &size, IPIV, &INFO);
    
    // Determinant is the product of the diaganol
    double _det = 1.0;
    for(i=0;i<n;i++) {
        _det *= a_c[i*n+i];
        }
    determinant[0] = (double)_det;
    
    // Compute the inverse using the LU decomposition
    dgetri_(&size, a_c ,&size, IPIV, WORK, &LWORK, &INFO);
    
    // Copy the inverse to the list
    for(i=0;i<n;i++)
       {
       for(j=0;j<n;j++)
         ainv[i][j]=a_c[i*n+j];
       }
    
    free(a_c);
    delete IPIV;
    delete WORK;
}


void get_inverse(py::indexed_ref a, int n, py::list ainv) {
    int i,j;
    double* a_c;
    int size = n;
    char trans = 'N';
    int nrhs = 1;
    int *IPIV = new int[n+1];
    int LWORK = n*n;
    double *WORK = new double[LWORK];
    int INFO;

    a_c = (double *)malloc(sizeof(double)*n*n);
    //p = (int*)malloc(sizeof(int)*n);
    for(i=0;i<n;i++)
       {
       for(j=0;j<n;j++)
         a_c[i*n+j]=a[i][j];
       }
       
    // Perform the LU decomposition
    dgetrf_(&size, &size, a_c, &size, IPIV, &INFO);
    
    // Compute the inverse using the LU decomposition
    dgetri_(&size, a_c ,&size, IPIV, WORK, &LWORK, &INFO);
    
    // Copy the inverse to the list
    for(i=0;i<n;i++)
       {
       for(j=0;j<n;j++)
         ainv[i][j]=a_c[i*n+j];
       }
    
    free(a_c);
    delete IPIV;
    delete WORK;
}


void get_inverse(py::indexed_ref a, int n, py::indexed_ref ainv) {
    int i,j;
    double* a_c;
    int size = n;
    char trans = 'N';
    int nrhs = 1;
    int *IPIV = new int[n+1];
    int LWORK = n*n;
    double *WORK = new double[LWORK];
    int INFO;

    a_c = (double *)malloc(sizeof(double)*n*n);
    //p = (int*)malloc(sizeof(int)*n);
    for(i=0;i<n;i++)
       {
       for(j=0;j<n;j++)
         a_c[i*n+j]=a[i][j];
       }
       
    // Perform the LU decomposition
    dgetrf_(&size, &size, a_c, &size, IPIV, &INFO);
    
    // Compute the inverse using the LU decomposition
    dgetri_(&size, a_c ,&size, IPIV, WORK, &LWORK, &INFO);
    
    // Copy the inverse to the list
    for(i=0;i<n;i++)
       {
       for(j=0;j<n;j++)
         ainv[i][j]=a_c[i*n+j];
       }
    
    free(a_c);
    delete IPIV;
    delete WORK;
}


void compute_residuals(py::list x, py::list mu, py::list residuals, py::list num_residuals) {
    int num_x = x.length();
    int num_mu = mu.length();
    
    int state_dimensions = mu[0].length();
    int icnt, jcnt;
    
    // Compute the residuals
    if (num_x == 1) {
        for (icnt=0; icnt<num_mu; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals[icnt][jcnt] = (double)x[0][jcnt]-(double)mu[icnt][jcnt];
            }
        }
        num_residuals[0] = num_mu;
    }
    else if (num_mu == 1) {
        for (icnt=0; icnt<num_x; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals[icnt][jcnt] = (double)x[icnt][jcnt]-(double)mu[0][jcnt];
            }
        }
        num_residuals[0] = num_x;
    }
    else {
        for (icnt=0; icnt<num_x; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals[icnt][jcnt] = (double)x[icnt][jcnt]-(double)mu[icnt][jcnt];
            }
        }
        num_residuals[0] = num_x;
    }
}

int do_cholesky(py::indexed_ref pyA, int dims, py::indexed_ref cholA) {
    //copyright            : (C) 2005 by Gunter Winkler, Konstantin Kutzkow
    //email                : guwi17@gmx.de
    
    size_t size = dims;
    
    typedef double DBL;
    typedef ublas::row_major  ORI;
    // use dense matrix
    ublas::matrix<DBL, ORI> A (size, size);
    ublas::matrix<DBL, ORI> L (size, size);
    
    A = ublas::zero_matrix<DBL>(size, size);
    L = ublas::zero_matrix<DBL>(size, size);
    
    for(int i=0;i<size;i++) {
        for(int j=0;j<size;j++)
            A(i, j)=pyA[i][j];
    }
    
    size_t res = cholesky_decompose(A, L);
    
    for(int i=0;i<size;i++) {
        for(int j=0;j<size;j++)
            cholA[i][j]=L(i, j);
    }
    
    return(res);
}
"""


class c_code:
    def __call__(self):
        return python_vars, code, support_code, libs
    support_code = ""
    code = ""
    python_vars = ""
    libs = ""
    

class resample_code1(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['nparticles', 'wt_cdf', 'u1', 'resampled_indices']
    code = """
    int j = 0;
    int array_cur_loc = 0;
    double uj, d_u1, float_nparticles;
    //d_u1 = (double)u1;
    float_nparticles = (double)nparticles;
    for (; j<nparticles; j++) {
        uj = u1 + (double)j/nparticles;
        while (wt_cdf[array_cur_loc] < uj) {
            array_cur_loc += 1;
        }
        resampled_indices[j] = array_cur_loc;
    }
    """
    

class resample_code2(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['nparticles', 'weights', 'len_weights', 'wt_cdf', 'u1', 
                   'resampled_indices']
    code = """
    double normfac = 0;
    int j = 0;
    int array_cur_loc = 0;
    double uj, d_u1;

    // Normalise the weights first
    for (j=0; j<len_weights; j++) {
        normfac += weights[j];
    }
    
    weights[0] = weights[0]/normfac;
    wt_cdf[0] = weights[0];
    for (j=1; j<len_weights; j++) {
        weights[j] = weights[j]/normfac;
        wt_cdf[j] = wt_cdf[j-1] + weights[j];
    }
    
    for (j=0; j<nparticles; j++) {
        uj = u1 + (double)j/nparticles;
        while (wt_cdf[array_cur_loc] < uj) {
            array_cur_loc += 1;
        }
        resampled_indices[j] = array_cur_loc;
    }
    """
    
    
    
class compute_residuals(c_code):
    def __call__(self):
        return (self.python_vars, self.code, self.support_code, self.libs)
    python_vars = ['x', 'mu', 'residuals', 'num_residuals']
    code = """
    int num_x = x.length();
    int num_mu = mu.length();
    
    int state_dimensions = mu[0].length();
    int icnt, jcnt;
    
    // Compute the residuals
    if (num_x == 1) {
        for (icnt=0; icnt<num_mu; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals[icnt][jcnt] = (double)x[0][jcnt]-(double)mu[icnt][jcnt];
            }
        }
        num_residuals[0] = num_mu;
    }
    else if (num_mu == 1) {
        for (icnt=0; icnt<num_x; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals[icnt][jcnt] = (double)x[icnt][jcnt]-(double)mu[0][jcnt];
            }
        }
        num_residuals[0] = num_x;
    }
    else {
        for (icnt=0; icnt<num_x; icnt++) {
            for (jcnt=0; jcnt<state_dimensions; jcnt++) {
                residuals[icnt][jcnt] = (double)x[icnt][jcnt]-(double)mu[icnt][jcnt];
            }
        }
        num_residuals[0] = num_x;
    }
    """
    support_code = ""
    libs=[]
    
    
class np_ndarray_imul(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['vec', 'mul_fac', 'vec_len']
    code = """
    for (int i=0; i<vec_len;i++) {
        vec[i] *= mul_fac;
    }
    """
    
    
class log_mvnpdf(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['log_likelihood', 'x', 'mu', 'sigma', 'sigmainv', 'a_inv_b',
                   'pi', 'residuals', 'num_residuals', 'determinant']
    libs=['lapack','blas']
    support_code = global_support_code
    
    code = """
    int num_x = x.length();
    int num_mu = mu.length();
    int num_sigma = sigma.length();
    
    int state_dimensions = mu[0].length();
    int icnt, jcnt, st_dim_cntx, st_dim_cnty;
    
    compute_residuals(x, mu, residuals, num_residuals);
    
    
    if (num_sigma == 1) {
        // One sigma, many residuals
        get_inverse(sigma[0], state_dimensions, determinant, sigmainv);
        
        double log2pidet = -((double)state_dimensions/2)*log(2*(double)pi)-0.5*log((double)determinant[0]);
        double inner_tmp_result, outer_result;
        
        for (icnt = 0; icnt < (int)num_residuals[0]; icnt++) {
            outer_result = 0;
            for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                inner_tmp_result = 0;
                for (st_dim_cnty=0; st_dim_cnty<state_dimensions; st_dim_cnty++) {
                    inner_tmp_result += (double)sigmainv[st_dim_cntx][st_dim_cnty]*(double)residuals[icnt][st_dim_cnty];
                }
                outer_result += (double)residuals[icnt][st_dim_cntx]*inner_tmp_result;
            }
            log_likelihood[icnt] = log2pidet - 0.5*outer_result;
        }
    }
    
    else {
        double log2pi_const = -((double)state_dimensions/2.0)*log(2*(double)pi);
        double outer_result;
        
        if (num_residuals == 1) {
            // Many sigma, one residual
            for (int sigma_cnt=0; sigma_cnt<num_sigma; sigma_cnt++) {
                outer_result = 0;
                solve_ax_eq_b(sigma[sigma_cnt], residuals[0], a_inv_b, state_dimensions, determinant);
                for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                    outer_result += (double)residuals[0][st_dim_cntx]*(double)a_inv_b[st_dim_cntx];
                }
                log_likelihood[sigma_cnt] = log2pi_const-0.5*log((double)determinant[0]) - 0.5*outer_result;
            }
        }
        else {
            // Many sigma, many residuals
            for (int sigma_cnt=0; sigma_cnt<num_sigma; sigma_cnt++) {
                outer_result = 0;
                solve_ax_eq_b(sigma[sigma_cnt], residuals[sigma_cnt], a_inv_b, state_dimensions, determinant);
                for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                    outer_result += (double)residuals[sigma_cnt][st_dim_cntx]*(double)a_inv_b[st_dim_cntx];
                }
                log_likelihood[sigma_cnt] = log2pi_const-0.5*log((double)determinant[0]) - 0.5*outer_result;
            }
        }
    }
    
    """
    

class mvnpdf(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['likelihood', 'x', 'mu', 'sigma', 'sigmainv', 'a_inv_b',
                   'pi', 'residuals', 'num_residuals', 'determinant', 'LOGPDF']
    libs=['lapack','blas']
    support_code = global_support_code
    code = """
    int num_x = x.length();
    int num_mu = mu.length();
    int num_sigma = sigma.length();
    
    int state_dimensions = mu[0].length();
    int icnt, jcnt, st_dim_cntx, st_dim_cnty;
    
    compute_residuals(x, mu, residuals, num_residuals);
    
    
    if (num_sigma == 1) {
        // One sigma, many residuals
        get_inverse(sigma[0], state_dimensions, determinant, sigmainv);
        
        double _2pidet = pow(2*(double)pi, -((double)state_dimensions/2.0))/sqrt((double)determinant[0]);
        double inner_tmp_result, outer_result;
        
        for (icnt = 0; icnt < (int)num_residuals[0]; icnt++) {
            outer_result = 0;
            for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                inner_tmp_result = 0;
                for (st_dim_cnty=0; st_dim_cnty<state_dimensions; st_dim_cnty++) {
                    inner_tmp_result += (double)sigmainv[st_dim_cntx][st_dim_cnty]*(double)residuals[icnt][st_dim_cnty];
                }
                outer_result += (double)residuals[icnt][st_dim_cntx]*inner_tmp_result;
            }
            if ((int)LOGPDF)
                likelihood[icnt] = log(_2pidet) - 0.5*outer_result;
            else
                likelihood[icnt] = _2pidet * exp(-0.5*outer_result);
        }
    }
    
    else {
        double _2pi_const = pow(2*(double)pi, -((double)state_dimensions/2.0));
        double outer_result;
        
        if (num_residuals == 1) {
            // Many sigma, one residual
            for (int sigma_cnt=0; sigma_cnt<num_sigma; sigma_cnt++) {
                outer_result = 0;
                solve_ax_eq_b(sigma[sigma_cnt], residuals[0], a_inv_b, state_dimensions, determinant);
                for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                    outer_result += (double)residuals[0][st_dim_cntx]*(double)a_inv_b[st_dim_cntx];
                }
                if ((int)LOGPDF)
                    likelihood[sigma_cnt] = log(_2pi_const)-0.5*log((double)determinant[0]) - 0.5*outer_result;
                else
                    likelihood[sigma_cnt] = _2pi_const/sqrt((double)determinant[0]) * exp(-0.5*outer_result);
            }
        }
        else {
            // Many sigma, many residuals
            for (int sigma_cnt=0; sigma_cnt<num_sigma; sigma_cnt++) {
                outer_result = 0;
                solve_ax_eq_b(sigma[sigma_cnt], residuals[sigma_cnt], a_inv_b, state_dimensions, determinant);
                for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                    outer_result += (double)residuals[sigma_cnt][st_dim_cntx]*(double)a_inv_b[st_dim_cntx];
                }
                if ((int)LOGPDF)
                    likelihood[sigma_cnt] = log(_2pi_const)-0.5*log((double)determinant[0]) - 0.5*outer_result;
                else
                    likelihood[sigma_cnt] = _2pi_const/sqrt((double)determinant[0])*exp(-0.5*outer_result);
            }
        }
    }
    
    """


class mahalanobis(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['mahalanobis_dist', 'x', 'y', 'P', 'sigmainv', 'a_inv_b',
                   'residuals', 'num_residuals']
    libs=['lapack','blas']
    support_code = global_support_code
    code = """
    int num_x = x.length();
    int num_y = y.length();
    int num_P = P.length();
    
    int state_dimensions = x[0].length();
    int icnt, jcnt, st_dim_cntx, st_dim_cnty;
    
    compute_residuals(x, y, residuals, num_residuals);
    
    if (num_P == 1) {
        // One sigma, many residuals
        get_inverse(P[0], state_dimensions, sigmainv);
        
        double inner_tmp_result, outer_result;
        
        for (icnt = 0; icnt < (int)num_residuals[0]; icnt++) {
            outer_result = 0;
            for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                inner_tmp_result = 0;
                for (st_dim_cnty=0; st_dim_cnty<state_dimensions; st_dim_cnty++) {
                    inner_tmp_result += (double)sigmainv[st_dim_cntx][st_dim_cnty]*(double)residuals[icnt][st_dim_cnty];
                }
                outer_result += (double)residuals[icnt][st_dim_cntx]*inner_tmp_result;
            }
            mahalanobis_dist[icnt] = sqrt(outer_result);
        }
    }
    
    else {
        double outer_result;
        
        if (num_residuals == 1) {
            // Many sigma, one residual
            for (int sigma_cnt=0; sigma_cnt<num_P; sigma_cnt++) {
                outer_result = 0;
                solve_ax_eq_b(P[sigma_cnt], residuals[0], a_inv_b, state_dimensions);
                for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                    outer_result += (double)residuals[0][st_dim_cntx]*(double)a_inv_b[st_dim_cntx];
                }
                mahalanobis_dist[sigma_cnt] = sqrt(outer_result);
            }
        }
        else {
            // Many sigma, many residuals
            for (int sigma_cnt=0; sigma_cnt<num_P; sigma_cnt++) {
                outer_result = 0;
                solve_ax_eq_b(P[sigma_cnt], residuals[sigma_cnt], a_inv_b, state_dimensions);
                for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                    outer_result += (double)residuals[sigma_cnt][st_dim_cntx]*(double)a_inv_b[st_dim_cntx];
                }
                mahalanobis_dist[sigma_cnt] = sqrt(outer_result);
            }
        }
    }
    """


class inverse(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'invA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    code = """
    int num_A = A.length();
    int state_dimensions = (int)dims;
    int i;
    
    for (i=0; i<num_A; i++) {
        get_inverse(A[i], state_dimensions, invA[i]);
    }
    """
    

class cholesky(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    code = """
    //do_cholesky(A[0], dims, cholA[0]);
    
    //copyright            : (C) 2005 by Gunter Winkler, Konstantin Kutzkow
    //email                : guwi17@gmx.de
    
    size_t size = dims;
    

    typedef double DBL;
    typedef ublas::row_major  ORI;
    // use dense matrix
    ublas::matrix<DBL, ORI> ublasA (size, size);
    ublas::matrix<DBL, ORI> L (size, size);

    ublasA = ublas::zero_matrix<DBL>(size, size);
    L = ublas::zero_matrix<DBL>(size, size);

    for(int i=0;i<size;i++) {
        for(int j=0;j<size;j++)
            ublasA(i, j)=A[i][j];
    }
    
    size_t res = cholesky_decompose(ublasA, L);
    
    for(int i=0;i<size;i++) {
        for(int j=0;j<size;j++)
            cholA[i][j]=L(i, j);
    }
    
    //return(res);
    """
    

class batch_cholesky(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    code = """
    int num_A = A.length();
    int state_dimensions = (int)dims;
    int i;
    
    for (i=0; i<num_A; i++) {
        do_cholesky(A[i], state_dimensions, cholA[i]);
    }
    
    """
    
    
class m_times_x(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    #(matrix_m, vector_x)
    code = """
    //(matrix_m * vector_x)
    """
    
    

class xt_times_m(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    #(matrix_m, vector_x)
    code = """
    //(vector_x' * matrix_m)
    """
    


class m_times_m(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    #(matrix_m, matrix_n)
    code = """
    //(matrix_m, matrix_n)
    """
    


class x_times_xt(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    #(vector_x, vector_y)
    code = """
    //(vector_x, vector_y)
    """
    
    


class xt_times_x(c_code):
    def __call__(self):
        return python_vars, code, support_code, libs
    python_vars = ['A', 'cholA', 'dims']
    libs=['lapack','blas']
    support_code = global_support_code
    code = """
    //(vector_x, vector_y)
    """
        
    
log_mvnpdf_code_orig = """
int num_x = x.length();
int num_mu = mu.length();
int num_sigma = sigma.length();
int num_residuals;

int state_dimensions = mu[0].length();
int icnt, jcnt, st_dim_cntx, st_dim_cnty;

// Compute the residuals
if (num_x == 1) {
    for (icnt=0; icnt<num_mu; icnt++) {
        for (jcnt=0; jcnt<state_dimensions; jcnt++) {
            residual[icnt][jcnt] = (double)x[0][jcnt]-(double)mu[icnt][jcnt];
        }
    }
    num_residuals = num_mu;
}
else if (num_mu == 1) {
    for (icnt=0; icnt<num_x; icnt++) {
        for (jcnt=0; jcnt<state_dimensions; jcnt++) {
            residual[icnt][jcnt] = (double)x[icnt][jcnt]-(double)mu[0][jcnt];
        }
    }
    num_residuals = num_x;
}
else {
    for (icnt=0; icnt<num_x; icnt++) {
        for (jcnt=0; jcnt<state_dimensions; jcnt++) {
            residual[icnt][jcnt] = (double)x[icnt][jcnt]-(double)mu[icnt][jcnt];
        }
    }
    num_residuals = num_x;
}


if (num_sigma == 1) {
    // We have sigma_det and sigma_inv computed in python
    double log2pidet = -((double)state_dimensions/2)*log(2*(double)pi)-0.5*log((double)sigma_det[0]);
    double inner_tmp_result, outer_result;
    
    for (icnt = 0; icnt < num_residuals; icnt++) {
        outer_result = 0;
        for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
            inner_tmp_result = 0;
            for (st_dim_cnty=0; st_dim_cnty<state_dimensions; st_dim_cnty++) {
                inner_tmp_result += (double)sigma_inv[0][st_dim_cntx][st_dim_cnty]*(double)residual[icnt][st_dim_cnty];
            }
            outer_result += (double)residual[icnt][st_dim_cntx]*inner_tmp_result;
        }
        log_likelihood[icnt] = log2pidet - 0.5*outer_result;
    }
}
else {
    double log2pi_const = -((double)state_dimensions/2.0)*log(2*(double)pi);
    double inner_tmp_result, outer_result;
    
    if (num_residuals == 1) {
        for (int sigma_cnt=0; sigma_cnt<num_sigma; sigma_cnt++) {
            outer_result = 0;
            for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                inner_tmp_result = 0;
                for (st_dim_cnty=0; st_dim_cnty<state_dimensions; st_dim_cnty++) {
                    inner_tmp_result += (double)sigma_inv[sigma_cnt][st_dim_cntx][st_dim_cnty]*(double)residual[0][st_dim_cnty];
                }
                outer_result += (double)residual[0][st_dim_cntx]*inner_tmp_result;
            }
            log_likelihood[sigma_cnt] = log2pi_const-0.5*log((double)sigma_det[sigma_cnt]) - 0.5*outer_result;
        }
    }
    else {
        for (int sigma_cnt=0; sigma_cnt<num_sigma; sigma_cnt++) {
            outer_result = 0;
            for (st_dim_cntx=0; st_dim_cntx<state_dimensions; st_dim_cntx++) {
                inner_tmp_result = 0;
                for (st_dim_cnty=0; st_dim_cnty<state_dimensions; st_dim_cnty++) {
                    inner_tmp_result += (double)sigma_inv[sigma_cnt][st_dim_cntx][st_dim_cnty]*(double)residual[sigma_cnt][st_dim_cnty];
                }
                outer_result += (double)residual[sigma_cnt][st_dim_cntx]*inner_tmp_result;
            }
            log_likelihood[sigma_cnt] = log2pi_const-0.5*log((double)sigma_det[sigma_cnt]) - 0.5*outer_result;
        }
    }
}

"""
