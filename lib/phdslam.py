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


from phdfilter import gmphdfilter
from phdfilter.phdfilter import PARAMETERS
import numpy as np
from common import misctools
from common import blas
import code

#def placeholder():
#    return (lambda:0)


class GMPHD_SLAM_FEATURE(gmphdfilter.GMPHD):
    def __init__(self, *args, **kwargs):
        super(GMPHD_SLAM_FEATURE, self).__init__(*args, **kwargs)
    
    def phdIterate(self, observations):
        """
        Performs a single iteration of the PHD filter except for the 
        prediction.
        """
        # Update existing states
        slam_info = self.phdUpdate(observations)
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
        return slam_info


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
        self.states = np.zeros((self.parameters.state_parameters["nparticles"], 
                                self.parameters.state_parameters["ndims"]),
                                dtype=float)
        # Map of landmarks approximated by the PHD conditioned on vehicle state
        self.maps = [self.create_default_feature() 
                for i in range(self.parameters.state_parameters["nparticles"])]
        # Particle weights
        self.weights = 1/float(self.parameters.state_parameters["nparticles"])* \
                        np.ones(self.parameters.state_parameters["nparticles"])
        
        # Information for updating the state weight
        self._slam_wt_list_ = []
        # Estimates
        self._estimate_ = PARAMETERS()
        self._estimate_.state = np.empty(0)
        self._estimate_.map = self.maps[0]
        
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
                print \
                  "Parameter ", parameter_name, " does not exist. Add it first"
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
        
    
    def _predict_map_(self, delta_t):
        # Store the current vehicla state so that this can be used in the map
        # functions
        for i in range(self.parameters.state_parameters["nparticles"]):
            setattr(self.maps[i].parameters.obs_fn.parameters, 
                    "parent_state", self.states[i])
            setattr(self.maps[i].parameters.pd_fn.parameters, 
                    "parent_state", self.states[i])
            setattr(self.maps[i].parameters.birth_fn.parameters, 
                    "parent_state", self.states[i])
        self.parameters.state_parameters.delta_t = delta_t
        [self.maps[i].phdPredict() 
                for i in range(self.parameters.state_parameters["nparticles"])]
    
    
    def update_with_features(self, observation_set, update_to_time):
        if ((update_to_time > self.last_odo_predict_time) or 
            (update_to_time > self.last_map_predict_time)):
            self.predict(update_to_time)
        self._update_map_with_features_(observation_set)
        self._update_state_with_features_()
    
    
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
        # Check if the slam weight list is empty
        if not len(self._slam_wt_list_):
            return
        phd_weights = np.array([self._slam_wt_list_[i].likelihood 
            for i in range(self.parameters.state_parameters["nparticles"])])
        # Normalise the weights
        phd_weights *= 1/phd_weights.sum()
        
        self.weights *= phd_weights
        
        # Clear the slam weight list  - this might be unnecessary
        self._slam_wt_list_ = []
    
    
    def _update_map_with_features_(self, observation_set):
        self._slam_wt_list_ = [self.maps[i].phdIterate(observation_set) 
                for i in range(self.parameters.state_parameters["nparticles"])]
    
    def get_estimate(self):
        if self.parameters.state_estimate_fn.handle != None:
            return self.parameters.state_estimate_fn.handle(self.weights, 
                                                 self.states, self.maps)
        self._state_estimate_()
        self._map_estimate_()
        return self._estimate_.state.copy(), self._estimate_.map.copy()
        
    def _state_estimate_(self):
        self._estimate_.state, self._estimate_.cov = \
                            misctools.sample_mn_cv(self.states, self.weights)
        return self._estimate_.state
        
    def _state_covariance_(self):
        return self._estimate_.cov
        
        
    def _map_estimate_(self):
        max_weight_idx = np.argmax(self.weights)
        self._estimate_.map = self.maps[max_weight_idx]
        return self._estimate_.map
    
    def resample(self):
        # Effective number of particles
        eff_nparticles = 1/np.power(self.weights, 2).sum()
        resample_threshold = (
                eff_nparticles/self.parameters.state_parameters["nparticles"])
        # Check if we have particle depletion
        if (resample_threshold > 
                    self.parameters.state_parameters["resample_threshold"]):
            return
        # Otherwise we need to resample
        resample_index = misctools.get_resample_index(self.weights)
        # self.states is a numpy array so the indexing operation forces a copy
        resampled_states = self.states[resample_index]
        resampled_maps = [self.maps[i].copy() for i in resample_index]
        resampled_weights = (
          np.ones(self.parameters.state_parameters["nparticles"], dtype=float)*
          1/float(self.parameters.state_parameters["nparticles"]))
        
        self.weights = resampled_weights
        self.states = resampled_states
        self.maps = resampled_maps
    
