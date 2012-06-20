#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       misctools.py
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

import blas_tools as blas
import numpy as np
from scipy import weave


def mahalanobis(x, P, y):
    """
    Compute the Mahalanobis distance given x, P and y as 
    sqrt((x-y)'*inv(P)*(x-y)).
    """
    residual = x.copy()
    P_copy = P.copy()
    blas.daxpy(-1.0, y, residual)
    _, p_residual = blas.dposv(P_copy, residual, OVERWRITE_A=True)
    return np.power(blas.ddot(residual, p_residual), 0.5)
    
    
def merge_states(wt, x, P):
    """
    Compute the weighted mean and covariance from a (numpy) list of states and
    covariances with their weights.
    """
    merged_wt = wt.sum()
    merged_x = np.sum(blas.daxpy(wt, x), 0)/merged_wt
    residual = x.copy()
    blas.daxpy(-1.0, np.array([merged_x]), residual)
    # Convert the residual to a column vector
    residual.shape += (1,)
    P_copy = P.copy()
    merged_P = sum(blas.dsyrk('l', residual, True, 1.0, wt, P_copy), 0)/merged_wt
    return merged_wt, merged_x, merged_P
    
    
def get_resample_index(weights, nparticles=-1):
    weights = weights/weights.sum()
    
    if nparticles==-1:
        nparticles = weights.shape[0]
    
    resampled_indices = np.empty(nparticles, dtype=int)
    wt_cdf = np.empty(weights.shape, dtype=float)
    u1 = np.random.uniform()/nparticles
    
    python_vars = ['nparticles', 'weights', 'wt_cdf', 'u1', 'resampled_indices']
    code = """
    double normfac = 0;
    int j = 0;
    int array_cur_loc = 0;
    double uj, d_u1;

    wt_cdf[0] = weights[0];
    for (j=1; j<len_weights; j++)
        wt_cdf[j] = wt_cdf[j-1] + weights[j];
    
    for (j=0; j<nparticles; j++) {
        uj = u1 + (double)j/(double)nparticles;
        while (wt_cdf[array_cur_loc] < uj) {
            array_cur_loc++;
        }
        resampled_indices[j] = array_cur_loc;
    }
    """
    weave.inline(code, python_vars, extra_compile_args=["-O3"])
    return resampled_indices
    
    
def mvnpdf(x, mu, sigma):
    # Compute the residuals
    residual = x.copy()
    if residual.shape[0] == 1:
        residual = np.repeat(residual, mu.shape[0], 0)
    blas.daxpy(-1.0, mu, residual)
    chol_sigma = blas.dpotrf(sigma)
    # Compute the determinant
    diag_vec = np.array([np.diag(chol_sigma[i]) 
                        for i in range(chol_sigma.shape[0])])
    det_sigma = diag_vec.prod(1)**2
    
    # If same number of sigma and residuals, or only residual and many sigma
    if sigma.shape[0] == residual.shape[0] or residual.shape[0] == 1:
        inv_sigma_times_residual = blas.dpotrs(chol_sigma, residual)
        exp_term = blas.ddot(residual, inv_sigma_times_residual)
        
    # Otherwise, we have only one sigma - compute the inverse once
    else:
        # Compute the inverse of the square root
        inv_sqrt_sigma = blas.dtrtri(chol_sigma, 'l')
        exp_term = np.power(blas.dgemv(inv_sqrt_sigma,residual), 2).sum(axis=1)
    
    pdf = np.exp(-0.5*exp_term)/np.sqrt(det_sigma*(2*np.pi)**x.shape[1])
    return pdf
    
    
def sample_mn_cv(x, wt=None):
    if wt==None:
        wt = 1.0/x.shape[0]*np.ones(x.shape[0])
    else:
        wt /= wt.sum()
    
    mean_x = x.copy()
    # Scale the state by the associated weight
    blas.dscal(wt, mean_x)
    # and take the sum
    mean_x = mean_x.sum(axis=0)
    
    residuals = x.copy()
    blas.daxpy(-1.0, np.array([mean_x]), residuals)
    cov_x = np.array([blas.dsyr('l', residuals, wt).sum(axis=0)/(1-(wt**2).sum())])
    blas.symmetrise(cov_x, 'l')
    return mean_x, cov_x[0]