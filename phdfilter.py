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
from phdmisctools import get_resample_index


class PHD(object):
    def __init__(self, markov_predict_fn_handle, markov_predict_fn_parameters, 
                 state_update_fn_handle, state_update_fn_parameters,
                 obs_fn_handle, obs_fn_parameters, 
                 clutter_fn_handle, clutter_fn_parameters,
                 birth_fn_handle, birth_fn_parameters,
                 ps_fn_handle, pd_fn_handle, likelihood_fn_handle,
                 estimate_fn_handle, 
                 phd_parameters={"nparticles":100,
                                 "elim_threshold":1e-3}):
        self.markov_predict_fn_handle = markov_predict_fn_handle
        self.markov_predict_fn_parameters = markov_predict_fn_parameters
        
        self.state_update_fn_handle = state_update_fn_handle
        self.state_update_fn_parameters = state_update_fn_parameters
        
        self.obs_fn_handle = obs_fn_handle
        self.obs_fn_parameters = obs_fn_parameters
        
        self.clutter_fn_handle = clutter_fn_handle
        self.clutter_fn_parameters = clutter_fn_parameters
        
        self.birth_fn_handle = birth_fn_handle
        self.birth_fn_parameters = birth_fn_parameters
        
        self.ps_fn_handle = ps_fn_handle
        self.pd_fn_handle = pd_fn_handle
        self.likelihood_fn_handle = likelihood_fn_handle
        
        self.estimate_fn_handle = estimate_fn_handle
        
        self.phd_parameters = phd_parameters
    
    
    def init(self, states, weights):
        self.states = states
        self.weights = weights
    
    
    def phdPredict(self):
        survival_probability = self.ps_fn_handle(self.states)
        self.weights.__imul__(survival_probability)
        self.markov_predict_fn_handle(self.states, self.predict_fn_parameters)
    
    
    def phdAppendBirth(self, birth_states, birth_weights):
        self.states += birth_states
        self.weights = np.append(self.weights, birth_weights)
        pass
    
    
    def phdUpdate(self, observation_set):
        """Generic particle PHD implementation. This function should be
        overloaded to suit the formulation of the filter."""

        num_observations = len(observation_set)
        
        detection_probability = self.pd_fn_handle(self.states)
        clutter_pdf = [self.clutter_fn_handle(_observation_) for _observation_ in observation_set]
        
        # Account for missed detection
        self._states_ = [copy.deepcopy(self.states) for count in range(num_observations+1)]
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # Scale the weights by detection probability -- same for all detection terms
        self.weights.__imul__(detection_probability)
        
        # If the PHD is approximated with a Gaussian mixture, we need to 
        # update the state using the observations.
        #if not (self.state_update_fn_handle is None):
        #    new_states = [self.state_update_fn_handle(copy.deepcopy(self.states), _observation_, self.state_update_fn_parameters) for _observation_ in observation_set]
        
        # Evaluate the likelihood (post-update)
        likelihood = [self.likelihood_fn_handle(_observation_, self.states) for _observation_ in observation_set]
        
        
        for obs_count in range(num_observations):
            _this_observation_weights_ = self.weights*likelihood[obs_count]/(clutter_pdf[obs_count]+likelihood[obs_count].sum())
            self._weights_ += [_this_observation_weights_]
    
    
    def phdFlattenUpdate(self):
        self.weights = np.array(self._weights_).flatten()
        self.states = []
        for counter in range(len(self._states_)):
            self.states += self._states_[counter]
        
        
    def phdPrune(self):
        if self.phd_parameters.elim_threshold <= 0:
            return
        retain_indices = np.flatnonzero(np.array([_weights_.sum() for _weights_ in self._weights_])>=self.phd_parameters.elim_threshold)
        pruned_states = [self._states_[ri] for ri in retain_indices]
        pruned_weights = [self._weights_[ri] for ri in retain_indices]
        self._states_ = pruned_states
        self._weights_ = pruned_weights
    
    
    def phdMerge(self):
        pass
    
    
    def phdEstimate(self):
        # Select from groups of particles with sum of weights >= 0.5
        valid_indices = np.flatnonzero(np.array([_weights_.sum() for _weights_ in self._weights_])>=0.5)
        filter_estimates = [self.estimate_fn_handle(self._states_[vi], self._weights_[vi]) for vi in valid_indices]
        return filter_estimates
        
    
    def phdResample(self, forceCopy=False):
        sum_weights = [_weights_.sum() for _weights_ in self._weights_]
        nparticles = np.array(sum_weights)*self.phd_parameters["nparticles"]
        for count in range(len(self._weights_)):
            resample_indices = get_resample_index(self._weights_[count], nparticles[count])
            self._weights_[count] = (1.0/nparticles[count])*np.ones(nparticles[count])
            if forceCopy:
                self._states_[count] = copy.deepcopy([self._states_[count][ridx] for ridx in resample_indices])
            else:
                self._states_[count] = [self._states_[count][ridx] for ridx in resample_indices]

