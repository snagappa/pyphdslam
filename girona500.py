# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:56:18 2012

@author: snagappa
"""

"""
Module containing the dynamics of Girona 500. These functions should be 
plugged into the phdslam module. The dynamics are copied directly from the 
g500_navigation module.
"""

import collections
from phdfilter import fn_params, PARAMETERS
import gmphdfilter
import phdslam
import numpy as np
import misctools

SLAM_FN_DEFS = collections.namedtuple("SLAM_FN_DEFS", 
                "state_markov_predict_fn state_obs_fn state_likelihood_fn \
                state__state_update_fn state_estimate_fn state_parameters \
                feature_markov_predict_fn feature_obs_fn \
                feature_likelihood_fn feature__state_update_fn clutter_fn \
                birth_fn ps_fn pd_fn feature_estimate_fn feature_parameters")


class G500_PHDSLAM(phdslam.PHDSLAM):
    def __init__(self, state_markov_predict_fn, state_obs_fn,
                 state_likelihood_fn, state__state_update_fn,
                 state_estimate_fn, state_parameters,
                 feature_markov_predict_fn, feature_obs_fn,
                 feature_likelihood_fn, feature__state_update_fn,
                 clutter_fn, birth_fn, ps_fn, pd_fn,
                 feature_estimate_fn, feature_parameters):
        # Force the state space to 6: x, y, z, vx, vy, vz
        # roll, pitch yaw + velocities must be fed from parent
        state_parameters["ndims"] = 6
        super(G500_PHDSLAM, self).__init__(
                            state_markov_predict_fn, state_obs_fn,
                            state_likelihood_fn, state__state_update_fn,
                            state_estimate_fn, state_parameters,
                            feature_markov_predict_fn, feature_obs_fn,
                            feature_likelihood_fn, feature__state_update_fn,
                            clutter_fn, birth_fn, ps_fn, pd_fn,
                            feature_estimate_fn, feature_parameters)
        
        
    def get_states(self, rows=None, cols=None):
        if rows==None:
            rows = range(self.state_parameters["nparticles"])
        if cols==None:
            cols = range(self.state_parameters["ndims"])
        return self.states[rows,cols]
    
    def x(self, idx=None):
        return self.get_states(cols=0)
    def y(self, idx=None):
        return self.get_states(cols=1)
    def z(self, idx=None):
        return self.get_states(cols=2)
    def vx(self, idx=None):
        return self.get_states(cols=3)
    def vy(self, idx=None):
        return self.get_states(cols=4)
    def vz(self, idx=None):
        return self.get_states(cols=5)
    
    def reset_states(self):
        self.states[:] = 0
        self.weights = 1/self.parameters.state_parameters["nparticles"]* \
                        np.ones(self.parameters.state_parameters["nparticles"])
                        
    def reset_maps(self):
        self.maps = [self.create_default_feature() 
                    for i in range(self.parameters.state_parameters["nparticles"])]
    
    
    def predict_state(self, u, predict_to_time):
        delta_t = self.last_odo_predict_time - predict_to_time
        self.last_odo_predict_time = predict_to_time
        g500_state_predict(self.states, u, delta_t, INPLACE=True)
    
    
    
#state_markov_predict_fn
def g500_state_predict(states, u, delta_t, INPLACE=True):
    if not INPLACE:
        states = states.copy()
    roll = u[0]
    pitch = u[1]
    yaw = u[2]
    x1 = x_1[0]
    y1 = x_1[1]
    vx1 = x_1[2]
    vy1 = x_1[3]
    vz1 = x_1[4]
    x = zeros(5)
    
    # Compute Prediction Model with constant velocity
    x[0] = x1 + cos(pitch)*cos(yaw)*vx1*t - cos(roll)*sin(yaw)*vy1*t + sin(roll)*sin(pitch)*cos(yaw)*vy1*t + sin(roll)*sin(yaw)*vz1*t + cos(roll)*sin(pitch)*cos(yaw)*vz1*t
    x[1] = y1 + cos(pitch)*sin(yaw)*vx1*t + cos(roll)*cos(yaw)*vy1*t + sin(roll)*sin(pitch)*sin(yaw)*vy1*t - sin(roll)*cos(yaw)*vz1*t + cos(roll)*sin(pitch)*sin(yaw)*vz1*t
    x[2] = vx1
    x[3] = vy1        
    x[4] = vz1
    
    
    #A = self.computeA(u, t)
    roll = u[0]
    pitch = u[1]
    yaw = u[2]
    
    A = eye(5)
    A[0,2] = cos(pitch)*cos(yaw)*t
    A[0,3] = -cos(roll)*sin(yaw)*t + sin(roll)*sin(pitch)*cos(yaw)*t
    A[0,4] = sin(roll)*sin(yaw)*t + cos(roll)*sin(pitch)*cos(yaw)*t
    A[1,2] = cos(pitch)*sin(yaw)*t
    A[1,3] = cos(roll)*cos(yaw)*t + sin(roll)*sin(pitch)*sin(yaw)*t
    A[1,4] = -sin(roll)*cos(yaw)*t + cos(roll)*sin(pitch)*sin(yaw)*t
        
    W = self.computeW(u, t)
    self._x_ = self.f(self.x, u, t)
    self._P_ = dot(dot(A, self.P), A.T) + dot(dot(W, self.Q), W.T)
    
    def computeA(self, u, t):
        
        return A
        
        
    def computeQ(self, q_var):
        Q = eye(3)
        return Q*q_var
    
    
    
    
    return True
    

def state_obs_fn():
    pass

def state_likelihood_fn():
    pass

def state__state_update_fn():
    pass

def g500_state_to_dvl_obs(*args, **kwargs):
    pass

def dvl_likelihood(*args, **kwargs):
    pass

#def gps_likelihood(*args, **kwargs):
#    pass


def g500_slam_fn_defs():
    # Vehicle state prediction
    state_markov_predict_fn_handle = g500_state_predict
    state_markov_predict_fn_parameters = PARAMETERS()
    state_markov_predict_fn = fn_params(state_markov_predict_fn_handle,
                                            state_markov_predict_fn_parameters)
    # Vehicle state to observation space
    state_obs_fn_handle = g500_state_to_dvl_obs
    state_obs_fn_parameters = PARAMETERS()
    state_obs_fn = fn_params(state_obs_fn_handle, state_obs_fn_parameters)
    
    # Likelihood function for importance sampling
    state_likelihood_fn_handle = dvl_likelihood
    state_likelihood_fn_parameters = PARAMETERS()
    state_likelihood_fn = fn_params(state_likelihood_fn_handle, 
                                    state_likelihood_fn_parameters)
    
    # Update function for state
    state__state_update_fn_handle = None
    state__state_update_fn_parameters = PARAMETERS()
    state__state_update_fn = fn_params(state__state_update_fn_handle,
                                           state__state_update_fn_parameters)
    
    # State estimation from particles
    state_estimate_fn_handle = misctools.sample_mn_cv
    state_estimate_fn_parameters = PARAMETERS()
    state_estimate_fn = fn_params(state_estimate_fn_handle,
                                  state_estimate_fn_parameters)
    
    # Parameters for the filter
    # Roll, pitch, yaw + velocities is common to all particles - we assume
    # that the information from the imu is perfect
    # We only need to estimate x,y,z. The roll, pitch and yaw must be fed
    # externally
    state_parameters = {"nparticles":48,
                        "ndims":6}
    
    
    # Parameters for the PHD filter
    feature_parameters = {"max_terms":100, 
                          "elim_threshold":1e-4, 
                          "merge_threshold":4,
                          "ndims":3}
    ndims = feature_parameters["ndims"]
    
    # Landmark state-prediction
    feature_markov_predict_fn_handle = gmphdfilter.markov_predict
    feature_markov_predict_fn_parameters = PARAMETERS()
    feature_markov_predict_fn_parameters.F = np.eye(3)
    feature_markov_predict_fn_parameters.Q = np.zeros(ndims)
    feature_markov_predict_fn = fn_params(feature_markov_predict_fn_handle, 
                                  feature_markov_predict_fn_parameters)
    
    # Landmark state-to-observation function
    feature_obs_fn_handle = None
    feature_obs_fn_parameters = PARAMETERS()
    feature_obs_fn_parameters.H = np.eye(ndims)
    feature_obs_fn_parameters.R = np.eye(ndims)
    feature_obs_fn = fn_params(feature_obs_fn_handle, 
                               feature_obs_fn_parameters)
    
    # Likelihood function - not used for the GM PHD filter
    feature_likelihood_fn = fn_params()
    
    # Landmark state update function - not used
    feature__state_update_fn = fn_params()
    
    # Clutter function
    clutter_fn_handle = gmphdfilter.uniform_clutter
    clutter_fn_parameters = PARAMETERS()
    clutter_fn_parameters.intensity = 2
    # Range should be the field of view of the sensor
    clutter_fn_parameters.range = [[-1, 1], [-1, 1], [-1, 1]]
    clutter_fn = fn_params(clutter_fn_handle, clutter_fn_parameters)
    
    # Birth function
    birth_fn_handle = gmphdfilter.measurement_birth
    birth_fn_parameters = PARAMETERS()
    birth_fn_parameters.intensity = 0.01
    birth_fn_parameters.obs2state = lambda x: np.array(x)
    birth_fn = fn_params(birth_fn_handle, birth_fn_parameters)
    
    # Survival/detection probability
    ps_fn_handle = gmphdfilter.constant_survival
    ps_fn_parameters = PARAMETERS()
    ps_fn_parameters.ps = 1
    ps_fn = fn_params(ps_fn_handle, ps_fn_parameters)
    pd_fn_handle = gmphdfilter.constant_detection
    pd_fn_parameters = PARAMETERS()
    pd_fn_parameters.pd = 0.98
    pd_fn = fn_params(pd_fn_handle, pd_fn_parameters)
    
    # Use default estimator
    feature_estimate_fn = fn_params()
    
    
    
    return SLAM_FN_DEFS(state_markov_predict_fn, state_obs_fn,
                        state_likelihood_fn, state__state_update_fn,
                        state_estimate_fn, state_parameters,
                        feature_markov_predict_fn, feature_obs_fn,
                        feature_likelihood_fn, feature__state_update_fn,
                        clutter_fn, birth_fn, ps_fn, pd_fn,
                        feature_estimate_fn, feature_parameters)
    