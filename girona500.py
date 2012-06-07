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
import blas_tools as blas


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
        predict_fn = self.parameters.state_markov_predict_fn
        self.states = predict_fn.handle(self.states, u, delta_t, 
                                        predict_fn.parameters)
    
    
    def update_gps(self, gps_obs):
        pred_gps_obs = self.state[:,0:2]
        likelihood = misctools.mvnpdf(np.array([gps_obs]), pred_gps_obs, 
                np.array([self.parameters.state_likelihood_fn.parameters.gps_obs_noise]))
        self.weights *= likelihood
    
    def update_dvl(self, dvl_obs):
        pred_dvl_obs = self.state[:,3:]
        likelihood = misctools.mvnpdf(np.array([dvl_obs]), pred_dvl_obs,
                np.array([self.parameters.state_likelihood_fn.parameters.dvl_obs_noise]))
        self.weights *= likelihood
    
    def update_svs(self, svs_obs):
        pred_svs_obs = self.state[:,2]
        likelihood = misctools.mvnpdf(np.array([svs_obs]), pred_svs_obs,
                np.array([0.2]))
        self.weights *= likelihood
    
#state_markov_predict_fn
def g500_state_predict(states, u, delta_t, parameters):
    # u is assumed to be ordered as [roll, pitch, yaw]
    r, p, y = 0, 1, 2
    # Evaluate cosine and sine of roll, pitch, yaw
    c = np.cos(u)
    s = np.sin(u)
    
    # Specify the rotation matrix
    # See http://en.wikipedia.org/wiki/Rotation_matrix
    R = delta_t * np.array(
            [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
             [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
             [-s[p], s[r]*c[p], c[r]*c[p] ]])
    # Transition matrix
    F = np.array([ np.vstack(( np.hstack((np.eye(3), R)),
                               np.hstack((np.zeros(3), np.eye(3))) )) ])
    # Multiply the transition matrix with each state
    pred_states = blas.dgemv(F, states, beta=0.0)
    
    ## Add white Gaussian noise to the predicted states
    # Compute scaling for the noise
    scale_matrix = np.array([np.vstack((R*delta_t/2,
                              np.eye(3)))])
    process_noise = np.array([parameters.process_noise])
    # Compute the process noise as scale_matrix*process_noise*scale_matrix'
    Q = blas.dgemm(scale_matrix, 
                   blas.dgemm(process_noise, scale_matrix, 
                              TRANSPOSE_B=True, beta=0.0), beta=0.0)[0]
    
    pred_states += np.random.multivariate_normal(mean=np.zeros(6, dtype=float),
                                                 cov=Q, size=(states.shape))
    return pred_states
    


def g500_slam_fn_defs():
    # Vehicle state prediction
    state_markov_predict_fn_handle = g500_state_predict
    state_markov_predict_fn_parameters = PARAMETERS()
    state_markov_predict_fn = fn_params(state_markov_predict_fn_handle,
                                            state_markov_predict_fn_parameters)
    # Vehicle state to observation space
    state_obs_fn_handle = None
    state_obs_fn_parameters = PARAMETERS()
    state_obs_fn = fn_params(state_obs_fn_handle, state_obs_fn_parameters)
    
    # Likelihood function for importance sampling
    state_likelihood_fn_handle = None
    state_likelihood_fn_parameters = PARAMETERS()
    state_likelihood_fn = fn_params(state_likelihood_fn_handle, 
                                    state_likelihood_fn_parameters)
    
    # Update function for state
    state__state_update_fn_handle = None
    state__state_update_fn_parameters = PARAMETERS()
    state__state_update_fn = fn_params(state__state_update_fn_handle,
                                           state__state_update_fn_parameters)
    
    # State estimation from particles
    state_estimate_fn_handle = None #misctools.sample_mn_cv
    state_estimate_fn_parameters = PARAMETERS()
    state_estimate_fn = fn_params(state_estimate_fn_handle,
                                  state_estimate_fn_parameters)
    
    # Parameters for the filter
    # Roll, pitch, yaw + velocities is common to all particles - we assume
    # that the information from the imu is perfect
    # We only need to estimate x,y,z. The roll, pitch and yaw must be fed
    # externally
    state_parameters = {"nparticles":48,
                        "ndims":6,
                        "resample_threshold":0.6}
    
    
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
    