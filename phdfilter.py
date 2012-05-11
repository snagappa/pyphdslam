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

def fn_params(handle=None, parameters=None):
    fn = (lambda:0)
    fn.handle = handle
    fn.parameters = parameters
    

class PHD(object):
    def __init__(self, markov_predict_fn, obs_fn, likelihood_fn, 
                 state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn,
                 estimate_fn, 
                 phd_parameters={"nparticles":100,
                                 "elim_threshold":1e-3}):
        # Markov prediction
        self.markov_predict_fn = markov_predict_fn
        # Observation function - transform from state to obs space
        self.obs_fn = obs_fn
        # Likelihood function
        self.likelihood_fn = likelihood_fn
        
        # State update functions - unused here
        self.state_update_fn = state_update_fn
        # Clutter function
        self.clutter_fn = clutter_fn
        # Birth function
        self.birth_fn = birth_fn
        
        # Survival function
        self.ps_fn = ps_fn
        # Detection function
        self.pd_fn = pd_fn
        
        # Estimator function - unused here
        self.estimate_fn = estimate_fn
        
        # Other PHD parameters
        self.phd_parameters = phd_parameters
        
        self.states = []
        self.weights = np.array([])
        self._states_ = []
        self._weights_ = np.array([])
        
    
    def init(self, states, weights):
        self.states = states
        self.weights = weights
        self._states_ = copy.deepcopy(states)
        self._weights_ = weights.copy()
        
    
    def phdPredict(self):
        survival_probability = self.ps_fn.handle(self.states, self.ps_fn.parameters)
        self.weights.__imul__(survival_probability)
        self.states = self.markov_predict_fn.handle(self.states, self.markov_predict_fn.parameters)
    
    
    def phdAppendBirth(self, birth_states, birth_weights):
        self.states += birth_states
        self.weights = np.append(self.weights, birth_weights)
    
    
    def phdUpdate(self, observation_set):
        """Generic particle PHD implementation. This function should be
        overloaded to suit the formulation of the filter."""

        num_observations = len(observation_set)
        
        detection_probability = self.pd_fn.handle(self.states, self.pd_fn.parameters)
        clutter_pdf = [self.clutter_fn.handle(_observation_, 
                                              self.clutter_fn.parameters) 
                       for _observation_ in observation_set]
        
        # Account for missed detection and duplicate the state num_obs times
        self._states_ = [copy.deepcopy(self.states) for count in range(num_observations+1)]
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # Scale the weights by detection probability -- same for all detection terms
        self.weights.__imul__(detection_probability)
        
        # If the PHD is approximated with a Gaussian mixture, we need to 
        # update the state using the observations.
        #if not (self.state_update_fn.handle is None):
        #    new_states = [self.state_update_fn.handle(copy.deepcopy(self.states), _observation_, self.state_update_fn.parameters) for _observation_ in observation_set]
        
        # Evaluate the likelihood (post-update)
        likelihood = [self.likelihood_fn.handle(_observation_, self.states, 
                                                self.likelihood_fn.parameters) 
                      for _observation_ in observation_set]
        
        
        for obs_count in range(num_observations):
            _this_observation_weights_ = self.weights*likelihood[obs_count]/(clutter_pdf[obs_count]+likelihood[obs_count].sum())
            self._weights_ += [_this_observation_weights_]
    
    
    def phdFlattenUpdate(self):
        self.weights = np.array(self._weights_).flatten()
        self.states = []
        for counter in range(len(self._states_)):
            self.states += self._states_[counter]
        
        
    def phdPrune(self):
        if self.phd_parameters['elim_threshold'] <= 0:
            return
        retain_indices = np.flatnonzero(np.array([_weights_.sum() for _weights_ in self._weights_])>=self.phd_parameters['elim_threshold'])
        pruned_states = [self._states_[ri] for ri in retain_indices]
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
        filter_estimates = [self.estimate_fn.handle(self._states_[vi], 
                                                    self._weights_[vi],
                                                    self.estimate_fn.parameters)
                            for vi in valid_indices]
        return filter_estimates
        
    
    def phdResample(self, forceCopy=True):
        sum_weights = [_weights_.sum() for _weights_ in self._weights_]
        nparticles = np.array(sum_weights)*self.phd_parameters["nparticles"]
        for count in range(len(self._weights_)):
            resample_indices = get_resample_index(self._weights_[count], nparticles[count])
            self._weights_[count] = (1.0/nparticles[count])*np.ones(nparticles[count])
            if forceCopy:
                self._states_[count] = copy.deepcopy([self._states_[count][ridx] for ridx in resample_indices])
            else:
                self._states_[count] = [self._states_[count][ridx] for ridx in resample_indices]

