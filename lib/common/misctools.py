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

import blas
import numpy as np
from scipy import weave
from operator import mul, add
import code

def gen_retain_idx(array_len, del_idx):
    if del_idx is None or len(del_idx) == 0:
        return range(array_len)
    retain_idx = [i for i in range(array_len) if i not in del_idx]
    return retain_idx

def mahalanobis(x, P, y):
    """
    Compute the Mahalanobis distance given x, P and y as 
    sqrt((x-y)'*inv(P)*(x-y)).
    """
    residual = x-y
    if P.shape[0] == 1:
        p_times_residual = np.linalg.solve(P[0], residual.T).T
    else:
        p_times_residual, _ = blas.dposv(P, residual, OVERWRITE_A=False)
    
    #blas_result = np.power(blas.ddot(residual, p_times_residual), 0.5)
    return (residual*p_times_residual).sum(1)**0.5
    
def approximate_mahalanobis(x, P, y):
    # Compute the mahalanobis distance using the diagonal of the matrix P
    assert P.shape[1] == P.shape[2], "P must be a square matrix"
    select_diag_idx = xrange(P.shape[1])
    residual = x-y
    diag_P = P[:, select_diag_idx, select_diag_idx]
    p_times_residual = (1.0/diag_P)*residual
    return (residual*p_times_residual).sum(1)**0.5

def merge_states(wt, x, P):
    """
    Compute the weighted mean and covariance from a (numpy) list of states and
    covariances with their weights.
    """
    merged_wt = wt.sum()
    #merged_x = np.sum(blas.daxpy(wt, x), 0)/merged_wt
    merged_x = (wt[:, np.newaxis]*x).sum(0)/merged_wt
    """
    residual = x.copy()
    blas.daxpy(-1.0, np.array([merged_x]), residual)
    """
    residual = x - merged_x
    # method 1:
    # Convert the residual to a column vector
    #residual.shape += (1,)
    #P_copy = P.copy()
    #blas.dsyr('l', residual, 1.0, P_copy)
    #merged_P = np.array([(wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt], order='C')
    #blas.symmetrise(merged_P, 'l')
    
    # method 2:
    #P_copy = P + blas.dger(residual, residual)
    
    # method 3:
    P_copy = P + [residual[np.newaxis,i].T * residual[np.newaxis,i] 
        for i in xrange(residual.shape[0])]
    
    #merged_P = np.array([(wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt], order='C')
    merged_P = (wt[:,np.newaxis,np.newaxis]*P_copy).sum(axis=0)/merged_wt
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
    for (j=1; j<Nweights[0]; j++)
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
    #if x.shape[0] == 1:
    #    residual = np.repeat(x, mu.shape[0], 0)
    #else:
    #    residual = x.copy(order='c')
    #blas.daxpy(-1.0, mu, residual)
    residual = x-mu
    #if x.shape[0] == 1:
    #    x = np.repeat(x, mu.shape[0], 0)
    #residual = x-mu
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
    
    pdf = np.exp(-0.5*exp_term)/np.sqrt(det_sigma*(2*np.pi)**residual.shape[1])
    return pdf
    
    
def sample_mn_cv(x, wt=None, SYMMETRISE=False):
    if wt.shape[0] == 1:
        return x[0].copy(), np.zeros((x.shape[1], x.shape[1]))
    if wt==None:
        wt = 1.0/x.shape[0]*np.ones(x.shape[0])
    else:
        wt /= wt.sum()
    
    #mean_x = x.copy()
    # Scale the state by the associated weight
    #blas.dscal(wt, mean_x)
    # and take the sum
    #mean_x = mean_x.sum(axis=0)
    
    #residuals = x.copy()
    #blas.daxpy(-1.0, np.array([mean_x]), residuals)
    
    #mean_x = np.apply_along_axis(mul, 0, x, wt).sum(axis=0)
    mean_x = (wt[:,np.newaxis]*x).sum(axis=0)
    residuals = x - mean_x
    cov_x = np.array([blas.dsyr('l', residuals, wt).sum(axis=0)/(1-(wt**2).sum())])
    if SYMMETRISE:
        blas.symmetrise(cov_x, 'l')
    return mean_x, cov_x[0]




###############################################################################
###############################################################################
#                                                                             #
# Copyright (C) 2010 Edward d'Auvergne                                        #
#                                                                             #
# This file is part of the program relax (http://www.nmr-relax.com).          #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                             #
###############################################################################

# Module docstring.
"""Module for transforming between different coordinate systems."""

def cartesian_to_spherical(vector):
    """Convert the Cartesian vector [x; y; z] to spherical coordinates [r; theta; phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    #r = norm(vector)
    r = np.sqrt((vector**2).sum(axis=0))

    # Unit vector.
    unit = vector / r

    # The polar angle.
    theta = np.arccos(unit[2])

    # The azimuth.
    phi = np.arctan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return np.vstack((r, theta, phi))


def spherical_to_cartesian(spherical_vect, cart_vect):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """

    # Trig alias.
    sin_theta = np.sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * np.cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * np.sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * np.cos(spherical_vect[1])
