#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       phdslam.py
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


import gmphdfilter
from phdfilter import PARAMETERS
import numpy as np
import misctools


#def placeholder():
#    return (lambda:0)


class GMPHD_SLAM_FEATURE(gmphdfilter.GMPHD):
    def __init__(self, markov_predict_fn, obs_fn, likelihood_fn,
                 state_update_fn, clutter_fn, birth_fn, ps_fn, pd_fn,
                 estimate_fn,
                 phd_parameters={"max_terms":100,
                                 "elim_threshold":1e-4,
                                 "merge_threshold":4}):
        super(GMPHD_SLAM_FEATURE, self).__init__(
                                    markov_predict_fn, obs_fn, likelihood_fn,
                                    state_update_fn, clutter_fn, birth_fn,
                                    ps_fn, pd_fn, estimate_fn, phd_parameters)
        
        
        def phdIterate(self, observations):
            # Update existing states
            self.phdUpdate(observations)
            # Generate estimates
            #estimates = self.phdEstimate()
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
            #return estimates


class PHDSLAM(object):
    def __init__(self, state_markov_predict_fn, state_obs_fn,
                 state_likelihood_fn, state__state_update_fn,
                 state_estimate_fn, state_parameters,
                 feature_markov_predict_fn, feature_obs_fn,
                 feature_likelihood_fn, feature__state_update_fn,
                 clutter_fn, birth_fn, ps_fn, pd_fn,
                 feature_estimate_fn, feature_parameters):
        
        self.parameters = PARAMETERS()
        self.set_slam_parameters(state_markov_predict_fn, state_obs_fn,
                               state_likelihood_fn, state__state_update_fn,
                               state_estimate_fn, state_parameters,
                               feature_markov_predict_fn, feature_obs_fn,
                               feature_likelihood_fn, feature__state_update_fn,
                               clutter_fn, birth_fn, ps_fn, pd_fn,
                               feature_estimate_fn, feature_parameters)
        
        # Vehicle states
        self.states = np.zeros(self.parameters.state_parameters["nparticles"], 
                              self.parameters.state_parameters.state_dims)
        # Map of landmarks approximated by the PHD conditioned on vehicle state
        self.maps = [self.create_default_feature() 
                    for i in range(self.parameters.state_parameters["nparticles"])]
        # Particle weights
        self.weights = 1/self.parameters.state_parameters["nparticles"]* \
                        np.ones(self.parameters.state_parameters["nparticles"])
        # Time from last update
        self.last_odo_predict_time = 0
        self.last_map_predict_time = 0
        self.last_update_time = 0
        # Last update was from either odometry or map landmarks
        self.last_update_type = None
        
        
    def set_slam_parameters(self, state_markov_predict_fn, state_obs_fn,
                            state_likelihood_fn, state__state_update_fn,
                            state_estimate_fn, state_parameters,
                            feature_markov_predict_fn, feature_obs_fn,
                            feature_likelihood_fn, feature__state_update_fn,
                            clutter_fn, birth_fn, ps_fn, pd_fn,
                            feature_estimate_fn, feature_parameters):
        
        self.set_state_parameters(state_markov_predict_fn, state_obs_fn,
                            state_likelihood_fn, state__state_update_fn,
                            state_estimate_fn, state_parameters)
        self.set_feature_parameters(
                            feature_markov_predict_fn, feature_obs_fn,
                            feature_likelihood_fn, feature__state_update_fn,
                            clutter_fn, birth_fn, ps_fn, pd_fn,
                            feature_estimate_fn, feature_parameters)
        
    
    def set_state_parameters(self, 
                                  state_markov_predict_fn, state_obs_fn,
                                  state_likelihood_fn, state__state_update_fn,
                                  state_estimate_fn, state_parameters):
        # Parameters for the parent process
        self.parameters.state_markov_predict_fn = state_markov_predict_fn
        self.parameters.state_obs_fn = state_obs_fn
        self.parameters.state_likelihood_fn = state_likelihood_fn
        self.parameters.state__state_update_fn = state__state_update_fn
        self.parameters.state_estimate_fn = state_estimate_fn
        self.parameters.state_parameters = state_parameters
        
    def set_feature_parameters(self,
                            feature_markov_predict_fn, feature_obs_fn,
                            feature_likelihood_fn, feature__state_update_fn,
                            clutter_fn, birth_fn, ps_fn, pd_fn,
                            feature_estimate_fn, feature_parameters):
        # Parameters for the (default) feature
        self.parameters.feature_markov_predict_fn = feature_markov_predict_fn
        self.parameters.feature_obs_fn = feature_obs_fn
        self.parameters.feature_likelihood_fn = feature_likelihood_fn
        self.parameters.feature__state_update_fn = feature__state_update_fn
        self.parameters.clutter_fn = clutter_fn
        self.parameters.birth_fn = birth_fn
        self.parameters.ps_fn = ps_fn
        self.parameters.pd_fn = pd_fn
        self.parameters.feature_estimate_fn = feature_estimate_fn
        self.parameters.feature_parameters = feature_parameters
        
        
    def set_states(self, states, weights):
        self.states = states
        self.weights = weights
        
    def set_maps(self, maps):
        self.maps = maps
    
    def add_parameter(self, parameter_name, new_value=None):
        setattr(self.parameters, parameter_name, new_value)
    
    def get_parameter(self, parameter_name, *default_value):
        if len(default_value):
            getattr(self.parameters, parameter_name, default_value)
        else:
            getattr(self.parameters, parameter_name)
    
    def set_parameter(self, parameter_name, new_value, force=0):
        if not force:
            try:
                getattr(self.parameters, parameter_name)
            except AttributeError:
                print "Parameter ", parameter_name, " does not exist. Add it first"
                return
        setattr(self.parameters, parameter_name, new_value)
        
    
    def create_default_feature(self):
        return GMPHD_SLAM_FEATURE(self.parameters.feature_markov_predict_fn,
                                  self.parameters.feature_obs_fn,
                                  self.parameters.feature_likelihood_fn,
                                  self.parameters.feature__state_update_fn,
                                  self.parameters.clutter_fn,
                                  self.parameters.birth_fn,
                                  self.parameters.ps_fn,
                                  self.parameters.pd_fn,
                                  self.parameters.feature_estimate_fn,
                                  self.parameters.feature_parameters)
    
    
    def predict(self, predict_to_time):
        odo_delta_t = predict_to_time - self.last_odo_predict_time
        self._predict_state_(odo_delta_t)
        self.last_odo_predict_time = predict_to_time
        map_delta_t = predict_to_time - self.last_map_predict_time
        self._predict_map_(map_delta_t)
        self.last_map_predict_time = predict_to_time
    
    
    def _predict_state_(self, delta_t):
        self.parameters.state_markov_predict_fn.parameters.delta_t = delta_t
        self.states = self.parameters.state_markov_predict_fn.handle(
          self.states, self.parameters.state_markov_predict_fn.parameters)
        # Store the current vehicla state so that this can be used in the map
        # functions
        for i in range(self.parameters.state_parameters["nparticles"]):
            setattr(self.maps[i].parameters.obs_fn.parameters, "parent_state", self.states[i])
            setattr(self.maps[i].parameters.pd_fn.parameters, "parent_state", self.states[i])
            setattr(self.maps[i].parameters.birth_fn.parameters, "parent_state", self.states[i])
        
    
    def _predict_map_(self, delta_t):
        self.parameters.state_parameters.delta_t = delta_t
        [self.maps[i].phdPredict() for i in range(self.parameters.state_parameters["nparticles"])]
    
    
    def update_with_feature(self, observation_set, update_to_time):
        if ((update_to_time > self.last_odo_predict_time) or 
            (update_to_time > self.last_map_predict_time)):
            self.predict(update_to_time)
        self._update_map_with_features_(observation_set)
        pass
    
    
    def update_with_odometry(self, odometry, update_to_time):
        odo_delta_t = update_to_time - self.last_odo_predict_time
        if odo_delta_t > 0:
            self._predict_state_(odo_delta_t)
            self.last_odo_predict_time = update_to_time
        
        # Perform update here
        self._update_state_with_odometry_(odometry)
        
    
    def _update_state_with_odometry_(self, odometry):
        # Evaluate the likelihood function to get the particle weights.
        pass
    
    
    def _rao_blackwellised__update_state_with_odometry(self, odometry):
        # Use a RB particle filter for the state update
        pass
    
    
    def _update_state_with_features_(self):
        # Some form of product of weights of the features
        pass
    
    
    def _update_map_with_features_(self, observation_set):
        [self.maps[i].phdIterate(observation_set) 
                for i in range(self.parameters.state_parameters["nparticles"])]
    
    
    def resample(self):
        resample_index = misctools.get_resample_index(self.weights)
        # self.states is a numpy array so the indexing operation forces a copy
        resampled_states = self.states[resample_index]
        resampled_maps = [self.maps[i].copy() for i in resample_index]
        resampled_weights = (
          np.ones(self.parameters.state_parameters["nparticles"], dtype=float)*
          1/self.parameters.state_parameters["nparticles"])
        
        self.weights = resampled_weights
        self.states = resampled_states
        self.maps = resampled_maps
    