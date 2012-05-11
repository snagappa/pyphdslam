#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       gmphdfilter.py
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
from phdfilter import PHD, fn_params
import phdmisctools
import copy

def placeholder():
    return (lambda:0)


class GMPHD(PHD):
    def __init__(self, markov_predict_fn, obs_fn, likelihood_fn,
                 state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn,
                 estimate_fn,
                 phd_parameters={"max_terms":100,
                                 "elim_threshold":1e-4,
                                 "merge_threshold":4}):
        super(GMPHD, self).__init__(markov_predict_fn, obs_fn, likelihood_fn,
                                    state_update_fn, clutter_fn, birth_fn,
                                    ps_fn, pd_fn, estimate_fn, phd_parameters)
        
    
    def phdGenerateBirth(self, observation_set):
        birth_states, birth_weights = \
            self.birth_fn.handle(observation_set, self.birth_fn.parameters)
        return birth_states, birth_weights
        
        
    def phdUpdate(self, observation_set):
        num_x = len(self.states)
        num_observations = len(observation_set)
        if num_observations:
            z_dim = len(observation_set[0])
        else:
            z_dim = 0
        
        detection_probability = self.pd_fn.handle(self.states, 
                                                  self.pd_fn.parameters)
        clutter_pdf = [self.clutter_fn.handle(_observation_, 
                                              self.clutter_fn.parameters) \
                       for _observation_ in observation_set]
        
        # Account for missed detection
        self._states_ = copy.deepcopy(self.states)
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # Scale the weights by detection probability -- same for all detection terms
        self.weights.__imul__(detection_probability)
        
        # Split x and P out from the combined state vector
        x = [self.states[i][0] for i in range(num_x)]
        P = [self.states[i][1] for i in range(num_x)]
        
        # Part of the Kalman update is common to all observation-updates
        x, P, kalman_info = kalman_update(x, P, 
                                          self.obs_fn.parameters.H, 
                                          self.obs_fn.parameters.R)
        
        # We need to update the states and find the updated weights
        for (_observation_, obs_count) in zip(observation_set, range(num_observations)):
            new_x = copy.deepcopy(x)
            # Apply the Kalman update to get the new state - update in-place
            # and return the residuals
            residuals = kalman_update_x(new_x, kalman_info.pred_z, 
                                        _observation_, kalman_info.kalman_gain)
            
            # Calculate the weight of the Gaussians for this observation
            # Calculate term in the exponent
            x_pdf = [np.exp(-0.5*(np.dot(kalman_info.inv_sqrt_S[i], 
                                         residuals[i]).A[0]**2).sum())/ \
                    np.sqrt(kalman_info.det_S[i]*(2*np.pi)^z_dim) 
                    for i in range(num_x)]
            new_weight = self.weights*x_pdf
            # Normalise the weights
            new_weight.__idiv__(clutter_pdf(obs_count) + new_weight.sum())
            
            # Create new state with new_x and P to add to _states_
            self._states_ +=[[new_x[i], copy.copy(P[i])] for i in range(num_x)]
            self._weights_ += [new_weight]
            
        self._weights_ = np.concatenate(self._weights_)
        
    
    def phdPrune(self):
        if (self.phd_parameters['elim_threshold'] <= 0):
            return
        retain_indices = np.where(self._weights_ >= self.phd_parameters['elim_threshold'])
        pruned_states = [self._states_[ri] for ri in retain_indices]
        pruned_weights = self._weights_[retain_indices]
        
        if (len(pruned_states) > self.phd_parameters['max_terms']):
            inds = np.flipud(self._weights_.argsort())
            inds = inds[self.phd_parameters['max_terms']:]
            phdmisctools.delete_from_list(pruned_states, inds)
            pruned_weights = np.delete(pruned_weights, inds)
        
        self._states_ = pruned_states
        self._weights_ = pruned_weights
        
    
    def phdMerge(self):
        if (self.phd_parameters['merge_threshold'] <= 0):
            return
        
        result_wt_list = []
        result_state_list = []
        
        num_remaining_components = len(self._weights_)
        while num_remaining_components:
            max_wt_index = self._weights_.argmax()
            x_list = [tmp_states_[0] for tmp_states_ in self._states_]
            P_list = [tmp_states_[1] for tmp_states_ in self._states_]
            
            mahalanobis_dist = phdmisctools.mahalanobis([self._states_[max_wt_index][0]], 
                                                        [self._states_[max_wt_index][1]], 
                                                         x_list)
            merge_list_indices = np.where(mahalanobis_dist <= self.phd_parameters['merge_threshold'])
            
            x_merge_list = [x_list[i] for i in merge_list_indices]
            P_merge_list = [P_list[i] for i in merge_list_indices]
            merged_wt, merged_x, merged_P = \
                phdmisctools.merge_states(self._weights_[merge_list_indices], 
                                          x_merge_list,
                                          P_merge_list)
            result_wt_list += [merged_wt]
            result_state_list += [[merged_x, merged_P]]
            
            phdmisctools.delete_from_list(self._states_, merge_list_indices)
            self._weights_ = np.delete(self._weights_, merge_list_indices)
            num_remaining_components = len(self._weights_)
        
        self._states_ = result_state_list
        self._weights_ = np.array(result_wt_list)
        
    
    def phdFlattenUpdate(self):
        pass
    
    
    def phdEstimate(self):
        total_intensity = sum(self._weights_)
        num_targets = int(round(total_intensity))
        
        inds = np.flipud(self._weights_.argsort())
        inds = inds[0:num_targets]
        est_state = [copy.copy(self._states_[i]) for i in inds]
        est_weight = self._weights_[inds]
        
        # Create empty structure to store state and weight
        estimate = lambda:0
        estimate.state = est_state
        estimate.weight = est_weight
        return estimate
    

def kalman_predict(x, P, F, Q):
    num_x = len(x)
    if len(F) == 1:
        f_idx = [0]*num_x
    else:
        f_idx = range(num_x)
    if len(Q) == 1:
        q_idx = [0]*num_x
    else:
        q_idx = range(num_x)
        
    # Predict state
    x_pred = [np.dot(F[f_idx[i]], x[i]).A[0] for i in range(num_x)]
    
    # Predict covariance
    P_pred = [F[f_idx[i]]*P[i]*F[f_idx[i]].T + Q[q_idx[i]] for i in range(num_x)]
    
    return x_pred, P_pred


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
        residuals = phdmisctools._compute_residuals_(z, pred_z)
        #[z - pred_z[i] for i in range(num_x)]
        x_upd = [x[i] + np.dot(kalman_gain[i], residuals[i]).A[0] for i in range(num_x)]
    else:
        x_upd = x
        
    # Update covariance
    P_upd = [P[i] - (kalman_gain[i]*H[h_idx[i]]*P[i])]
    
    kalman_info.inv_sqrt_S = inv_sqrt_S
    kalman_info.det_S = det_S
    kalman_info.pred_z = pred_z
    kalman_info.kalman_gain = kalman_gain
    
    return x_upd, P_upd, kalman_info
    

def kalman_update_x(x, zhat, z, kalman_gain):
    num_x = len(x)
    residuals = phdmisctools._compute_residuals_(z, zhat)
    #residuals = [z - zhat[i] for i in range(num_x)]
    x_upd = [x[i] + np.dot(kalman_gain[i], residuals[i]).A[0] for i in range(num_x)]
    return x_upd, residuals
    

def markov_predict(state, parameters):
    num_x = len(state)
    x = [jointstate[0] for jointstate in state]
    P = [jointstate[1] for jointstate in state]
    x_pred, P_pred = kalman_predict(x, P, parameters.F, parameters.Q)
    state = [[x[i], P[i]] for i in range(num_x)]
    return state
    
    
def uniform_clutter(z, parameters):
    return [parameters.intensity/np.prod(np.diff(parameters.range))]*len(z)
    

def measurement_birth(state, z, parameters):
    # Convert the observation to state space
    x = parameters.obs2state(z)
    
    # Couple each with (observation) noise
    P = [copy.copy(parameters.R) for count in range(len(z))]
    
    birth_states = [[x[i], P[i]] for i in range(len(z))]
    birth_weights = parameters.intensity*range(len(z))
    return birth_states, birth_weights
    

def constant_survival(state, parameters):
    return [parameters.ps]*len(state)
    

def constant_detection(state, parameters):
    return [parameters.pd]*len(state)


def default_constant_position_model(dims=2):
    markov_predict_fn_handle = markov_predict
    markov_predict_fn_parameters = placeholder()
    markov_predict_fn_parameters.F = [np.matrix(np.eye(dims))]
    markov_predict_fn_parameters.Q = [np.matrix(np.eye(dims))]
    markov_predict_fn = fn_params(markov_predict_fn_handle, markov_predict_fn_parameters)
    
    obs_fn_handle = None
    obs_fn_parameters = placeholder()
    obs_fn_parameters.H = [np.matrix(np.eye(dims))]
    obs_fn_parameters.R = [np.matrix(np.eye(dims))]
    obs_fn = fn_params(obs_fn_handle, obs_fn_parameters)
    
    # Likelihood function - not used here
    likelihood_fn = fn_params()
    
    return (markov_predict_fn, obs_fn, likelihood_fn)
    
    
def default_phd_parameters():
    state_update_fn = fn_params()
    
    # Clutter function
    clutter_fn_handle = uniform_clutter
    clutter_fn_parameters = placeholder()
    clutter_fn_parameters.intensity = 2
    clutter_fn_parameters.range = [[-1, 1], [-1, 1]]
    clutter_fn = fn_params(clutter_fn_handle, clutter_fn_parameters)
    
    # Birth function
    birth_fn_handle = measurement_birth
    birth_fn_parameters = placeholder()
    birth_fn_parameters.intensity = 0.01
    birth_fn = fn_params(birth_fn_handle, birth_fn_parameters)
    
    # Survival/detection probability
    ps_fn_handle = constant_survival
    ps_fn_parameters = placeholder()
    ps_fn_parameters.ps = 1
    ps_fn = fn_params(ps_fn_handle, ps_fn_parameters)
    pd_fn_handle = constant_detection
    pd_fn_parameters = placeholder()
    pd_fn_parameters = 0.98
    pd_fn = fn_params(pd_fn_handle, pd_fn_parameters)
    
    # Use default estimator
    estimate_fn = fn_params()
    
    return (state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn)
    
    
def default_gm_parameters():
    phd_parameters = {"max_terms":100, 
                      "elim_threshold":1e-4, 
                      "merge_threshold":4}
    return phd_parameters
    
    
def gmphdfilter(observations):
    markov_predict_fn, obs_fn, likelihood_fn = default_constant_position_model(dims=2)
    state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn = \
        default_phd_parameters()
    
    phd_parameters={"max_terms":100, 
                    "elim_threshold":1e-4, 
                    "merge_threshold":4}
    
    gmphdobj = GMPHD(markov_predict_fn, obs_fn, likelihood_fn, state_update_fn,
                     clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn, 
                     phd_parameters)
    
    estimates = []
    for obs in observations:
        gmphdobj.phdPredict()
        gmphdobj.phdUpdate(obs)
        estimates += [gmphdobj.phdEstimate()]
        gmphdobj.phdPrune()
        gmphdobj.phdMerge()
        birth_states, birth_weights = gmphdobj.phdGenerateBirth(obs)
        gmphdobj.phdAppendBirth(birth_states, birth_weights)
        
    return estimates
    
