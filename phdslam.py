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
import numpy as np
import copy
import phdmisctools


def placeholder():
    return (lambda:0)


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

class PHDSLAM(object):
    def __init__(self, state_markov_predict_fn, state_obs_fn,
                 state_likelihood_fn, state__state_update_fn,
                 state_estimate_fn, state_parameters,
                 feature_markov_predict_fn, feature_obs_fn,
                 feature_likelihood_fn, feature__state_update_fn,
                 clutter_fn, birth_fn, ps_fn, pd_fn,
                 feature_estimate_fn, feature_parameters):
        
        self.parameters = placeholder()
        self.set_slam_parameters(state_markov_predict_fn, state_obs_fn,
                               state_likelihood_fn, state__state_update_fn,
                               state_estimate_fn, state_parameters,
                               feature_markov_predict_fn, feature_obs_fn,
                               feature_likelihood_fn, feature__state_update_fn,
                               clutter_fn, birth_fn, ps_fn, pd_fn,
                               feature_estimate_fn, feature_parameters)
        
        
        self.states = np.zeros(self.state_parameters.num_particles, 
                              self.state_parameters.state_dims).tolist()
        self.maps = [self.create_default_feature() 
                    for i in range(state_parameters.num_particles)]
        self.weights = 1/self.state_parameters.num_particles* \
                        np.ones(self.state_parameters.num_particles)
        
        
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
        
        
    def init(self, particles, weights):
        self.particles = particles
        self.weights = weights
        
    
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
            except:
                print "Parameter ", parameter_name, " does not exist. Add it first"
                return
        setattr(self.parameters, parameter_name, new_value)
        
    
    def create_default_feature(self):
        return GMPHD_SLAM_FEATURE(self.parameters.feature_markov_transition,
                                  self.parameters.feature_obs_fn,
                                  self.parameters.feature_likelihood_fn,
                                  self.parameters.feature__state_update_fn,
                                  self.parameters.clutter_fn,
                                  self.parameters.birth_fn,
                                  self.parameters.ps_fn,
                                  self.parameters.pd_fn,
                                  self.parameters.feature_estimate_fn,
                                  self.parameters.feature_parameters)
    
    
    def predict(self):
        self._predict_state_()
        self._predict_map_()
    
    
    def _predict_state_(self):
        self.states = self.parameters.state_markov_predict_fn.handle(
          self.states, self.parameters.state_markov_predict_fn.parameters)
        for i in range(self.parameters.state_parameters.num_particles):
            setattr(self.maps[i].parameters.obs_fn.parameters, "parent_state", self.states[i], force=1)
            setattr(self.maps[i].parameters.pd_fn.parameters, "parent_state", self.states[i], force=1)
            setattr(self.maps[i].parameters.birth_fn.parameters, "parent_state", self.states[i], force=1)
        
    
    def _predict_map_(self):
        [self.maps[i].phdPredict() for i in range(self.parameters.state_parameters.num_particles)]
    
    
    def update_with_feature(self):
        pass
    
    
    def update_with_odometry(self):
        pass
    
    
    def _update_state_with_odometry_(self, z):
        pass
    
    
    def _update_state_with_features_(self):
        
        pass
    
    
    def _update_map_with_features_(self, observation_set):
        [self.maps[i].phdUpdate(observation_set) for i in range(self.parameters.state_parameters.num_particles)]
        
    
    def resample(self):
        pass
    