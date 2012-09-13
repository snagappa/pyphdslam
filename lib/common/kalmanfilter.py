# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:14:38 2012

@author: snagappa
"""

from lib.common import blas
import numpy as np

#class kalmanfilter(object): pass
__all__ = []

def kf_predict(state, covariance, F, Q, B=None, u=None):
    pred_state = blas.dgemv(F, state)
    if (not B==None) and (not u==None):
        blas.dgemv(B, u, y=pred_state)
    # Repeat Q n times and return as the predicted covariance
    pred_cov = np.repeat(np.array([Q]), state.shape[0], 0)
    blas.dgemm(F, blas.dgemm(covariance, F, TRANSPOSE_B=True), C=pred_cov)
    return pred_state, pred_cov
    

def kf_update(state, covariance, H, R, z=None, INPLACE=True):
    kalman_info = lambda:0
    assert z == None or len(z.shape) == 1, "z must be a single observations, \
    not an array of observations"
    
    if INPLACE:
        upd_state = state
    else:
        upd_state = state.copy()
    # Update the covariance and generate the Kalman gain, etc.
    upd_covariance, kalman_info = kf_update_cov(covariance, H, R, INPLACE)
    
    # Observation from current state
    pred_z = blas.dgemv(H, state)
    if not (z==None):
        upd_state, residuals = kf_update_x(upd_state, pred_z, z, 
                                           kalman_info.kalman_gain, INPLACE=True)
    else:
        residuals = np.empty(0)
    
    kalman_info.pred_z = pred_z
    kalman_info.residuals = residuals
    
    return upd_state, upd_covariance, kalman_info
    
    
def kf_update_x(x, pred_z, z, kalman_gain, INPLACE=True):
    assert len(z.shape) == 1, "z must be a single observations, \
    not an array of observations"
    if INPLACE:
        upd_state = x
    else:
        upd_state = x.copy()
    
    #residuals = np.repeat([z], pred_z.shape[0], 0)
    #blas.daxpy(-1, pred_z, residuals)
    residuals = z - pred_z
    # Update the state
    blas.dgemv(kalman_gain, residuals, y=upd_state)
    
    return upd_state, residuals
    

def np_kf_update_x(x, pred_z, z, kalman_gain, INPLACE=True):
    assert len(z.shape) == 1, "z must be a single observations, \
    not an array of observations"
    if INPLACE:
        upd_state = x
    else:
        upd_state = x.copy()
    
    #residuals = np.repeat([z], pred_z.shape[0], 0)
    #blas.daxpy(-1, pred_z, residuals)
    residuals = z - pred_z
    # Update the state
    upd_state += np.dot(kalman_gain, residuals.T).T
    
    return upd_state, residuals
    

def kf_predict_cov(covariance, F, Q):
    # Repeat Q n times and return as the predicted covariance
    #pred_cov = np.repeat(np.array([Q]), covariance.shape[0], 0)
    #blas.dgemm(F, blas.dgemm(covariance, F, TRANSPOSE_B=True), C=pred_cov)
    pred_cov = blas.dgemm(F, blas.dgemm(covariance, F, TRANSPOSE_B=True)) + Q
    return pred_cov
    
def np_kf_update_cov(covariance, H, R, INPLACE=True):
    kalman_info = lambda:0
    
    if INPLACE:
        upd_covariance = covariance
        covariance_copy = covariance.copy()
    else:
        upd_covariance = covariance.copy()
        covariance_copy = covariance
    
    # Comput PH^T
    p_ht = np.dot(covariance, H.T)
    # Compute HPH^T + R
    hp_ht_pR = np.dot(H, p_ht) + R
    
    # Compute the Cholesky decomposition
    chol_S = np.linalg.cholesky(hp_ht_pR)
    # Compute the determinant
    det_S = (np.diag(chol_S).prod())**2
    # Compute the inverse of the square root
    inv_sqrt_S = np.array(np.linalg.inv(chol_S), order='C')
    # and the inverse
    inv_S = np.dot(inv_sqrt_S.T, inv_sqrt_S)
    
    # Kalman gain
    kalman_gain = np.dot(p_ht, inv_S)
    
    # Update the covariance
    k_h = np.dot(kalman_gain, H)
    upd_covariance -= np.dot(k_h, covariance_copy)
    
    kalman_info.S = hp_ht_pR
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.kalman_gain = kalman_gain
    
    return upd_covariance, kalman_info
    
    
def kf_update_cov(covariance, H, R, INPLACE=True):
    kalman_info = lambda:0
    
    if INPLACE:
        upd_covariance = covariance
        covariance_copy = covariance.copy()
    else:
        upd_covariance = covariance.copy()
        covariance_copy = covariance
    
    # Store R
    #chol_S = np.repeat(R, covariance.shape[0], 0)
    # Compute PH^T
    p_ht = blas.dgemm(covariance, H, TRANSPOSE_B=True)
    # Compute HPH^T + R
    #blas.dgemm(H, p_ht, C=chol_S)
    hp_ht_pR = blas.dgemm(H, p_ht) + R
    # Compute the Cholesky decomposition
    chol_S = blas.dpotrf(hp_ht_pR, False)
    # Select the lower triangle (set the upper triangle to zero)
    blas.mktril(chol_S)
    # Compute the determinant
    diag_vec = np.array([np.diag(chol_S[i]) for i in range(chol_S.shape[0])])
    det_S = diag_vec.prod(1)**2
    # Compute the inverse of the square root
    inv_sqrt_S = blas.dtrtri(chol_S, 'l')
    # Compute the inverse using dsyrk
    inv_S = blas.dsyrk('l', inv_sqrt_S, TRANSPOSE_A=True)
    # Symmetrise the matrix since only the lower triangle is stored
    blas.symmetrise(inv_S, 'l')
    #blas.dpotri(op_S, True)
    # inv_S = op_S
    
    # Kalman gain
    kalman_gain = blas.dgemm(p_ht, inv_S)
    
    # Update the covariance
    k_h = blas.dgemm(kalman_gain, H)
    blas.dgemm(k_h, covariance_copy, alpha=-1.0, C=upd_covariance)
    
    kalman_info.S = hp_ht_pR
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.kalman_gain = kalman_gain
    
    return upd_covariance, kalman_info
    

def ukf_predict(states, covs, ctrl_input, proc_noise, predict_fn, 
                delta_t, parameters, _alpha=1e-3, _beta=0, _kappa=0):
    # Time update
    covs += proc_noise# + 1e-3*np.eye(covs.shape[0])
    sigma_x, wt_mn, wt_cv = sigma_pts(states, covs, _alpha, _beta, _kappa)
    
    sigma_x_pred = predict_fn(sigma_x, ctrl_input, delta_t, 
                                                parameters)
    
    # Predicted state is weighted mean of predicted sigma points
    pred_state = sigma_x_pred.copy()
    blas.dscal(wt_mn, pred_state)
    pred_state = pred_state.sum(axis=0)
    
    # Predicted covariance is weighted mean of sigma covariance + proc_noise
    pred_cov = _sigma_cov_(sigma_x_pred, pred_state, wt_cv, 0)
    return pred_state, pred_cov
    
def sigma_pts(x, x_cov, _alpha=1e-3, _beta=2, _kappa=0):
    # State dimensions
    _L = x.shape[0]
    # UKF parameters
    _lambda = _alpha**2 * (_L+_kappa) - _L
    _gamma = np.sqrt(_L + _lambda)
    
    # Square root of scaled covariance matrix
    sqrt_cov = _gamma*np.linalg.cholesky(x_cov)
    
    # Array of the sigma points
    sigma_x = np.vstack((x, x-sqrt_cov.T, x+sqrt_cov.T))
    
    # Array of the weights for each sigma point
    wt_mn = 0.5*np.ones(1+2*_L)/(_L+_lambda)
    wt_mn[0] = _lambda/(_L+_lambda)
    wt_cv = wt_mn.copy()
    wt_cv[0] += (1 - _alpha**2 + _beta)
    return sigma_x, wt_mn, wt_cv

def _sigma_cov_(sigma_x, x_hat, wt_cv, proc_noise):
    residuals = sigma_x - x_hat
    sigma_cov = np.array([ (blas.dsyr('l', residuals, wt_cv)).sum(axis=0) ])
    blas.symmetrise(sigma_cov, 'l')
    sigma_cov = sigma_cov[0] + proc_noise

