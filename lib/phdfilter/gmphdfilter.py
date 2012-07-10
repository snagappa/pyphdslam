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
from phdfilter import PHD, fn_params, PARAMETERS
#import copy
import collections
from lib.common import blas
from lib.common import misctools
from lib.common import kalmanfilter

#def placeholder():
#    return (lambda:0)

GMSAMPLE = collections.namedtuple("GMSAMPLE", "state covariance")

class GMSTATES(object):
    """
    Creates an object to hold a set of Gaussian states with some state and 
    covariance.
    """
    def __init__(self, num_states, ndims=0):
        self._state_ = np.zeros((num_states, ndims))
        self._covariance_ = np.zeros((num_states, ndims, ndims))
        
    def set(self, state, covariance, CREATE_NEW_COPY=False):
        if CREATE_NEW_COPY:
            self._state_ = state.copy()
            self._covariance_ = covariance.copy()
        else:
            self._state_ = state
            self._covariance_ = covariance
    
    def members(self):
        return self._state_, self._covariance_
        
    def state(self):
        return self._state_
        
    def covariance(self):
        return self._covariance_
        
    def append(self, new_state):
        if self._state_.shape[0] == 0 or self._state_.shape[1] == 0:
            self._state_ = new_state.state().copy()
            self._covariance_ = new_state.covariance().copy()
        else:
            self._state_ = np.append(self._state_, new_state.state())
            self._covariance_ = np.append(self._covariance_, 
                                          new_state.covariance())
        
    def copy(self):
        state_copy = self.__class__(0)
        state_copy._state_ = self._state_.copy()
        state_copy._covariance = self._covariance_
        return state_copy
    
    def select(self, idx_vector, INPLACE=True):
        fn_return_val = None
        if not INPLACE:
            state_copy = self.__class__(0)
        else:
            state_copy = self
            fn_return_val = state_copy
        
        state_copy._state_ = self._state_[idx_vector]
        state_copy._covariance_ = self._covariance_[idx_vector]
        return fn_return_val
        
    
    def delete(self, idx_vector, INPLACE=True):
        fn_return_val = None
        if not INPLACE:
            state_copy = self.__class__(0)
        else:
            state_copy = self
            fn_return_val = state_copy
        
        state_copy._state_ = np.delete(self._state_, idx_vector, 0)
        state_copy._covariance_ = np.delete(self._covariance_, idx_vector, 0)
        return fn_return_val
        
    
    def __getitem__(self, index):
        return GMSAMPLE(self._state_[index].copy(), 
                        self._covariance_[index].copy())
    
    def __setitem__(self, key, item):
        self._state_[key] = item[0]
        self._covariance_[key] = item[1]
        
    def __len__(self):
        return self._state_.shape[0]
    


class GMPHD(PHD):
    """
    Creates an object to evaluate the PHD of a multi-object distribution using
    a Gaussian mixture.
    """
    def __init__(self, *args, **kwargs):
        super(GMPHD, self).__init__(*args, **kwargs)
        self.states = GMSTATES(0)
        self._states_ = GMSTATES(0)
        self.parameters = PARAMETERS()
        if len(args) or len(kwargs):
            self.init(*args, **kwargs)
            
    #def phdGenerateBirth(self, observation_set):
    #    birth_states, birth_weights = \
        #        self.birth_fn.handle(observation_set, self.birth_fn.parameters)
    #    return birth_states, birth_weights
        
    def init(self, markov_predict_fn, obs_fn, likelihood_fn, 
             state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn,
             estimate_fn, phd_parameters={"max_terms":100,
                                          "elim_threshold":1e-4,
                                          "merge_threshold":4}):
        super(GMPHD, self).init(markov_predict_fn, obs_fn, likelihood_fn,
                                    state_update_fn, clutter_fn, birth_fn,
                                    ps_fn, pd_fn, estimate_fn, phd_parameters)
    
    def phdUpdate(self, observation_set):
        # Container for slam parent update
        slam_info = PARAMETERS()
        num_observations = len(observation_set)
        if num_observations:
            z_dim = observation_set.shape[1]
        else:
            z_dim = 0
        
        detection_probability = self.parameters.pd_fn.handle(self.states, 
                                            self.parameters.pd_fn.parameters)
        #clutter_pdf = [self.clutter_fn.handle(_observation_, 
        #                                      self.clutter_fn.parameters) \
        #               for _observation_ in observation_set]
        clutter_pdf = self.parameters.clutter_fn.handle(observation_set, 
                                        self.parameters.clutter_fn.parameters)
        # Account for missed detection
        self._states_ = self.states.copy()
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # Scale the weights by detection probability 
        #   -- same for all detection terms
        self.weights.__imul__(detection_probability)
        
        # SLAM,  step 1:
        slam_info.exp_sum__pd_predwt = np.exp(-self.weights.sum())
        
        # Split x and P out from the combined state vector
        detected_states = self.states[detection_probability > 0.1]
        x = detected_states.state
        P = detected_states.covariance
        
        # Part of the Kalman update is common to all observation-updates
        x, P, kalman_info = kalmanfilter.kf_update(x, P, 
                            np.array([self.parameters.obs_fn.parameters.H]), 
                            np.array([self.parameters.obs_fn.parameters.R]), 
                            None, INPLACE=True)#USE_NP=0)
        
        # SLAM, prep for step 2:
        slam_info.sum__clutter_with_pd_updwt = np.zeros(num_observations)
        # Container for the updated states
        new_gmstate = self.states.__class__(0)
        # We need to update the states and find the updated weights
        for (_observation_, obs_count) in zip(observation_set, 
                                              range(num_observations)):
            #new_x = copy.deepcopy(x)
            # Apply the Kalman update to get the new state - update in-place
            # and return the residuals
            new_x, residuals = kalmanfilter.kf_update_x(x, kalman_info.pred_z, 
                                        _observation_, kalman_info.kalman_gain,
                                        INPLACE=False)
            
            # Calculate the weight of the Gaussians for this observation
            # Calculate term in the exponent
            x_pdf = np.exp(-0.5*np.power(
                blas.dgemv(kalman_info.inv_sqrt_S, residuals), 2).sum(axis=1))/ \
                np.sqrt(kalman_info.det_S*(2*np.pi)**z_dim) 
            
            new_weight = self.weights*x_pdf
            # Normalise the weights
            normalisation_factor = clutter_pdf[obs_count] + new_weight.sum()
            new_weight.__idiv__(normalisation_factor)
            # SLAM, step 2:
            slam_info.sum__clutter_with_pd_updwt[obs_count] = \
                                                        normalisation_factor
            
            # Create new state with new_x and P to add to _states_
            new_gmstate.set(new_x, P)
            self._states_.append(new_gmstate)
            self._weights_ += [new_weight]
            
        self._weights_ = np.concatenate(self._weights_)
        # SLAM, finalise:
        slam_info.likelihood = (slam_info.exp_sum__pd_predwt * 
                                slam_info.sum__clutter_with_pd_predwt.prod())
        return slam_info
    
    def phdPrune(self):
        if (self.parameters.phd_parameters['elim_threshold'] <= 0):
            return
        retain_indices = np.where((self._weights_ >= 
                          self.parameters.phd_parameters['elim_threshold']))[0]
        pruned_states = self._states_.select(retain_indices, INPLACE=False)
        pruned_weights = self._weights_[retain_indices]
        
        if (len(pruned_states) > self.parameters.phd_parameters['max_terms']):
            inds = np.flipud(pruned_weights.argsort())
            inds = inds[self.parameters.phd_parameters['max_terms']:]
            pruned_states.delete(inds, INPLACE=True)
            pruned_weights = np.delete(pruned_weights, inds)
        
        self._states_.set( *pruned_states.members() )
        self._weights_ = pruned_weights
        
    
    def phdMerge(self):
        if (self.parameters.phd_parameters['merge_threshold'] <= 0):
            return
        
        result_wt_list = []
        result_state_list = []
        result_covariance_list = []
        
        num_remaining_components = len(self._weights_)
        while num_remaining_components:
            max_wt_index = self._weights_.argmax()
            
            max_wt_state = self._states_[max_wt_index]
            mahalanobis_dist = misctools.mahalanobis(max_wt_state.state, 
                                                      max_wt_state.covariance, 
                                                      self._states_.state())
            merge_list_indices = np.where(mahalanobis_dist <= 
                          self.parameters.phd_parameters['merge_threshold'])[0]
            merge_list_states = self._states_[merge_list_indices] 
                                    #.select(merge_list_indices, INPLACE=False)
            merged_wt, merged_x, merged_P = misctools.merge_states(
                                            self._weights_[merge_list_indices], 
                                            merge_list_states.state,
                                            merge_list_states.covariance)
                
            result_wt_list += [merged_wt]
            result_state_list += [merged_x]
            result_covariance_list += [merged_P]
            
            #phdmisctools.delete_from_list(
            self._states_.delete(merge_list_indices)
            self._weights_ = np.delete(self._weights_, merge_list_indices)
            num_remaining_components = len(self._weights_)
        
        self._states_.set(np.array(result_state_list), 
                          np.array(result_covariance_list))
        self._weights_ = np.array(result_wt_list)
        
    
    def phdFlattenUpdate(self):
        self.states = self._states_
        self.weights = self._weights_
        self._states_ = self._states_.__class(0)
        self._weights_ = np.empty(0, dtype=float)
    
    
    def phdEstimate(self):
        total_intensity = sum(self._weights_)
        num_targets = int(round(total_intensity))
        
        inds = np.flipud(self._weights_.argsort())
        inds = inds[0:num_targets]
        est_state = self._states_[inds]
        est_weight = self._weights_[inds]
        
        # Create empty structure to store state and weight
        estimate = lambda:0
        estimate.state = est_state
        estimate.weight = est_weight
        return estimate
        
        
    def phdIterate(self, observations):
        # Predict existing states
        self.phdPredict()
        # Update existing states
        self.phdUpdate(observations)
        # Generate estimates
        estimates = self.phdEstimate()
        # Prune low weight Gaussian components
        self.phdPrune()
        # Create birth terms from measurements
        birth_states, birth_weights = self.phdGenerateBirth(observations)
        # Append birth terms to Gaussian mixture
        self.phdAppendBirth(birth_states, birth_weights)
        # Merge components
        self.phdMerge()
        # End of iteration call
        self.phdFlattenUpdate()
        return estimates
    
    
##############################################################################

def markov_predict(state, parameters):
    x_pred, P_pred = kalmanfilter.kf_predict(state.state(), state.covariance(), 
                                             np.array([parameters.F]),
                                             np.array([parameters.Q]))
    state.set(x_pred, P_pred)
    return state
    
    
def uniform_clutter(z, parameters):
    return [float(parameters.intensity)/
            np.prod(np.diff(parameters.range))]*len(z)
    

def measurement_birth(z, parameters):
    # Convert the observation to state space
    x = parameters.obs2state(z)
    # Couple each with (observation) noise
    P = np.array([parameters.R]).repeat(len(z), 0)
    birth_states = GMSTATES(0)
    birth_states.set(x, P)
    birth_weights = parameters.intensity*range(len(z))
    return birth_states, birth_weights
    

def constant_survival(state, parameters):
    return parameters.ps*np.ones(len(state))
    

def constant_detection(state, parameters):
    return parameters.pd*np.ones(len(state))


def default_constant_position_model(dims=3):
    markov_predict_fn_handle = markov_predict
    markov_predict_fn_parameters = PARAMETERS()
    markov_predict_fn_parameters.F = np.eye(dims)
    markov_predict_fn_parameters.Q = np.eye(dims)
    markov_predict_fn = fn_params(markov_predict_fn_handle, 
                                  markov_predict_fn_parameters)
    
    obs_fn_handle = None
    obs_fn_parameters = PARAMETERS()
    obs_fn_parameters.H = np.eye(dims)
    obs_fn_parameters.R = np.eye(dims)
    obs_fn = fn_params(obs_fn_handle, obs_fn_parameters)
    
    # Likelihood function - not used here
    likelihood_fn = fn_params()
    
    return (markov_predict_fn, obs_fn, likelihood_fn)
    
    
def default_phd_parameters():
    state_update_fn = fn_params()
    
    # Clutter function
    clutter_fn_handle = uniform_clutter
    clutter_fn_parameters = PARAMETERS()
    clutter_fn_parameters.intensity = 2
    clutter_fn_parameters.range = [[-1, 1], [-1, 1]]
    clutter_fn = fn_params(clutter_fn_handle, clutter_fn_parameters)
    
    # Birth function
    birth_fn_handle = measurement_birth
    birth_fn_parameters = PARAMETERS()
    birth_fn_parameters.intensity = 0.01
    birth_fn = fn_params(birth_fn_handle, birth_fn_parameters)
    
    # Survival/detection probability
    ps_fn_handle = constant_survival
    ps_fn_parameters = PARAMETERS()
    ps_fn_parameters.ps = 1
    ps_fn = fn_params(ps_fn_handle, ps_fn_parameters)
    pd_fn_handle = constant_detection
    pd_fn_parameters = PARAMETERS()
    pd_fn_parameters.pd = 0.98
    pd_fn = fn_params(pd_fn_handle, pd_fn_parameters)
    
    # Use default estimator
    estimate_fn = fn_params()
    
    return (state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn)
    
    
def default_gm_parameters():
    phd_parameters = {"max_terms":100, 
                      "elim_threshold":1e-4, 
                      "merge_threshold":4}
    return phd_parameters
    
    
def get_default_gmphd_obj_args():
    return tuple(list(default_constant_position_model(dims=2)) +
            list(default_phd_parameters()) + 
            list([default_gm_parameters()]))
    
    
def default_gmphd_obj(dims=2):
    markov_predict_fn, obs_fn, likelihood_fn = (
                                       default_constant_position_model(dims))
    state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn = \
        default_phd_parameters()
    phd_parameters = default_gm_parameters()
    gmphdobj = GMPHD(markov_predict_fn, obs_fn, likelihood_fn, state_update_fn,
                     clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn, 
                     phd_parameters)
    return gmphdobj
    
    
def gmphdfilter(observations):
    markov_predict_fn, obs_fn, likelihood_fn = (
                                       default_constant_position_model(dims=2))
    state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn, estimate_fn = \
        default_phd_parameters()
    
    phd_parameters = {"max_terms":100, 
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
        gmphdobj.phdFlattenUpdate()
        birth_states, birth_weights = gmphdobj.phdGenerateBirth(obs)
        gmphdobj.phdAppendBirth(birth_states, birth_weights)
        
    return estimates
    
