#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       phdfilter.py
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

"""This module includes abstractions of the PHD filter birth, prediction and
update. No assumptions are made about the state, Markov prediction or
evaluation of the likelihood function, except that all fields must be stored as
numpy.ndarray. The use of numpy.matrix is discouraged, but this representation
may be used internally in some functions.

The PHD (internal) representation is in the form of a list for states and a
numpy.array for the weights. The user must specify the formulation of the
state, possibly as a tuple when mean and covariance are used to represent
components of the PHD.

The predict and update functions will attempt to change values in place and a 
copy must be made prior to the function call if the previous states/weights
are required."""


import numpy as np
import copy
from misctools import get_resample_index

class PARAMETERS(object): pass

def fn_params(handle=None, parameters=None):
    f_p = PARAMETERS()
    f_p.handle = handle
    f_p.parameters = parameters
    return f_p


class STATES(object):
    """
    Creates an object to store the states of particles.
    """
    def __init__(self, num_states, ndims=0):
        self._state_ = np.zeros((num_states, ndims))
        
    def set(self, state, CREATE_NEW_COPY=False):
        if CREATE_NEW_COPY:
            self._state_ = state.copy()
        else:
            self._state_ = state
    
    def members(self):
        return (self._state_,)
        
    def state(self):
        return self._state_
        
    def append(self, new_state):
        if self._state_.shape[0] == 0 or self._state_.shape[1] == 0:
            self._state_ = new_state.state().copy()
        else:
            self._state_ = np.append(self._state_, new_state.state())
        
    def copy(self):
        state_copy = STATES(0)
        state_copy._state_ = self._state_.copy()
        return state_copy
    
    def select(self, idx_vector, INPLACE=False):
        fn_return_val = None
        if not INPLACE:
            state_copy = STATES(0)
            state_copy._state_ = self._state_[idx_vector]
            fn_return_val = state_copy
        else:
            self._state_ = self._state_[idx_vector]
        return fn_return_val
        
    
    def delete(self, idx_vector, INPLACE=True):
        fn_return_val = None
        if not INPLACE:
            state_copy = STATES(0)
            state_copy._state_ = np.delete(self._state_, idx_vector, 0)
            fn_return_val = state_copy
        else:
            self._state_ = np.delete(self._state_, idx_vector, 0)
        return fn_return_val
        
        
    def __getitem__(self, index):
        return self._state_[index].copy()
    
    def __setitem__(self, key, item):
        self._state_[key] = item
    

class PHD(object):
    def __init__(self, markov_predict_fn, obs_fn, likelihood_fn, 
                 state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn,
                 estimate_fn, 
                 phd_parameters={"nparticles":100,
                                 "elim_threshold":1e-3}):
        self.parameters = PARAMETERS()
        
        # Markov prediction
        self.parameters.markov_predict_fn = markov_predict_fn
        # Observation function - transform from state to obs space
        self.parameters.obs_fn = obs_fn
        # Likelihood function
        self.parameters.likelihood_fn = likelihood_fn
        
        # State update functions - unused here
        self.parameters.state_update_fn = state_update_fn
        # Clutter function
        self.parameters.clutter_fn = clutter_fn
        # Birth function
        self.parameters.birth_fn = birth_fn
        
        # Survival function
        self.parameters.ps_fn = ps_fn
        # Detection function
        self.parameters.pd_fn = pd_fn
        
        # Estimator function - unused here
        self.parameters.estimate_fn = estimate_fn
        
        # Other PHD parameters
        self.parameters.phd_parameters = phd_parameters
        
        self.states = STATES(0)
        self.weights = np.array([])
        self._states_ = STATES(0)
        self._weights_ = np.array([])
        
    
    def init(self, states, weights):
        self.states = states
        self.weights = weights
        self._states_ = states.copy()
        self._weights_ = weights.copy()
        
    
    def add_parameter(self, parameter_name, new_value=None):
        setattr(self.parameters, parameter_name, new_value)
    
    def get_parameter(self, parameter_name, *default_value):
        if len(default_value):
            getattr(self.parameters, parameter_name, default_value)
        else:
            getattr(self.parameters, parameter_name)
    
    def set_parameter(self, parameter_name, new_value):
        try:
            getattr(self.parameters, parameter_name)
        except AttributeError:
            print "Parameter ", parameter_name, " does not exist. Add it first"
            return
        setattr(self.parameters, parameter_name, new_value)
        
    
    def phdPredict(self):
        survival_probability = self.parameters.ps_fn.handle(self.states, self.parameters.ps_fn.parameters)
        self.weights.__imul__(survival_probability)
        self.states = self.parameters.markov_predict_fn.handle(self.states, 
                                self.parameters.markov_predict_fn.parameters)
    
    
    def phdGenerateBirth(self, observation_set):
        if not (self.parameters.birth_fn.handle is None):
            birth_states, birth_weights = \
                self.parameters.birth_fn.handle(observation_set, 
                                        self.parameters.birth_fn.parameters)
            return birth_states, birth_weights
        else:
            return [], np.empty(0)
        
        
    def phdAppendBirth(self, birth_states, birth_weights):
        self.states.append(birth_states)
        self.weights = np.append(self.weights, birth_weights)
    
    
    def phdUpdate(self, observation_set):
        """Generic particle PHD implementation. This function should be
        overloaded to suit the formulation of the filter."""

        num_observations = len(observation_set)
        
        detection_probability = self.parameters.pd_fn.handle(self.states, 
                                            self.parameters.pd_fn.parameters)
        clutter_pdf = [self.parameters.clutter_fn.handle(_observation_, 
                                        self.parameters.clutter_fn.parameters) 
                       for _observation_ in observation_set]
        
        # Account for missed detection and duplicate the state num_obs times
        self._states_ = STATES(0)
        pred_states = self.states()
        pred_states.shape = (1,)+pred_states.shape
        self._states_.set(pred_states)
        [self._states_.append(pred_states) for count in range(num_observations+1)]
        pred_states.shape = pred_states.shape[1:]
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # Scale the weights by detection probability -- same for all detection terms
        self.weights.__imul__(detection_probability)
        
        # If the PHD is approximated with a Gaussian mixture, we need to 
        # update the state using the observations.
        #if not (self.state_update_fn.handle is None):
        #    new_states = [self.state_update_fn.handle(copy.deepcopy(self.states), _observation_, self.state_update_fn.parameters) for _observation_ in observation_set]
        
        # Evaluate the likelihood (post-update)
        likelihood = [self.parameters.likelihood_fn.handle(_observation_, 
                                                           self.states, 
                                    self.parameters.likelihood_fn.parameters) 
                      for _observation_ in observation_set]
        
        
        for obs_count in range(num_observations):
            _this_observation_weights_ = self.weights*likelihood[obs_count]/(clutter_pdf[obs_count]+likelihood[obs_count].sum())
            self._weights_ += [_this_observation_weights_]
    
    
    def phdFlattenUpdate(self):
        self.weights = np.array(self._weights_).flatten()
        self.states = STATES(0)
        for counter in range(len(self._states_)):
            self.states.append(self._states_[counter])
        
        
    def phdPrune(self):
        if self.parameters.phd_parameters['elim_threshold'] <= 0:
            return
        # retain_indices holds the valid indices from each updated group --
        # retain_indices is a nested vector.
        retain_indices = np.flatnonzero(np.array([_weights_.sum() for _weights_ in self._weights_])>=self.parameters.phd_parameters['elim_threshold'])
        pruned_states = STATES(0)
        [pruned_states.append(self._states_[ri]) for ri in retain_indices]
        pruned_weights = [self._weights_[ri] for ri in retain_indices]
        self._states_ = pruned_states
        self._weights_ = pruned_weights
    
    
    def phdMerge(self):
        pass
    
    
    def phdEstimate(self):
        # Select from groups of particles with sum of weights >= 0.5
        # Use np.where instead?
        valid_indices = np.flatnonzero(np.array([_weights_.sum() 
                                        for _weights_ in self._weights_])>=0.5)
        filter_estimates = [self.parameters.estimate_fn.handle(self._states_[vi], 
                                                    self._weights_[vi],
                                                    self.parameters.estimate_fn.parameters)
                            for vi in valid_indices]
        return filter_estimates
        
    
    def phdResample(self, FORCE_COPY=True):
        sum_weights = [_weights_.sum() for _weights_ in self._weights_]
        nparticles = np.array(sum_weights)*self.parameters.phd_parameters["nparticles"]
        for count in range(len(self._weights_)):
            resample_indices = get_resample_index(self._weights_[count], nparticles[count])
            self._weights_[count] = (1.0/nparticles[count])*np.ones(nparticles[count])
            if FORCE_COPY:
                self._states_[count] = copy.deepcopy([self._states_[count][ridx] for ridx in resample_indices])
            else:
                self._states_[count] = [self._states_[count][ridx] for ridx in resample_indices]
    
    
    def intensity(self):
        return self.weights.sum()
