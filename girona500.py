#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       girona500.py
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
import rospy
import code

from gmphdfilter import blas_kf_update

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
        # Covariances for Gaussian mixture representation of the state
        #self.covariances = np.zeros((self.parameters.state_parameters["nparticles"],
        #                            self.parameters.state_parameters["ndims"],
        #                            self.parameters.state_parameters["ndims"]),
        #                            dtype=float)
        self.covariances = np.repeat([1*np.eye(state_parameters["ndims"])], 
                                      state_parameters["nparticles"], 0)
        
        
    def get_states(self, rows=None, cols=None):
        if rows == None:
            rows = range(self.parameters.state_parameters["nparticles"])
        if cols == None:
            cols = range(self.parameters.state_parameters["ndims"])
        return self.states[rows, cols]
    
    
    def reset_states(self):
        self.states[:] = 0
        self.weights = 1/float(self.parameters.state_parameters["nparticles"])* \
                        np.ones(self.parameters.state_parameters["nparticles"])
                        
    def reset_maps(self):
        self.maps = [self.create_default_feature() 
                    for i in range(self.parameters.state_parameters["nparticles"])]
    
    def trans_matrices(self, ctrl_input, delta_t):
        # Get process noise
        process_noise = np.array([
            self.parameters.state_markov_predict_fn.parameters.process_noise])
        
        # u is assumed to be ordered as [roll, pitch, yaw]
        r, p, y = 0, 1, 2
        # Evaluate cosine and sine of roll, pitch, yaw
        c = np.cos(ctrl_input)
        s = np.sin(ctrl_input)
        
        # Specify the rotation matrix
        # See http://en.wikipedia.org/wiki/Rotation_matrix
        rot_mat = delta_t * np.array(
                [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
                 [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
                 [-s[p], s[r]*c[p], c[r]*c[p] ]])
        # Transition matrix
        trans_mat = np.array([ np.vstack(( np.hstack((np.eye(3), rot_mat)),
                                   np.hstack((np.zeros((3,3)), np.eye(3))) )) ])
        ## Add white Gaussian noise to the predicted states
        # Compute scaling for the noise
        scale_matrix = np.array([np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))])
        
        # Compute the process noise as scale_matrix*process_noise*scale_matrix'
        sc_process_noise = blas.dgemm(scale_matrix, 
                       blas.dgemm(process_noise, scale_matrix, 
                                  TRANSPOSE_B=True, beta=0.0), beta=0.0)[0]
        return trans_mat, sc_process_noise
    
    def predict_state(self, ctrl_input, predict_to_time):
        if self.last_odo_predict_time==0:
            self.last_odo_predict_time = predict_to_time
            return
        delta_t = predict_to_time - self.last_odo_predict_time
        self.last_odo_predict_time = predict_to_time
        if delta_t < 0:
            #print "negative delta_t, ignoring"
            return
        
        trans_mat, sc_process_noise = self.trans_matrices(ctrl_input, delta_t)
        pred_states = blas.dgemv(trans_mat, self.states)
        nparticles = self.parameters.state_parameters["nparticles"]
        ndims = self.parameters.state_parameters["ndims"]
        pred_states += np.random.multivariate_normal(mean=np.zeros(ndims, dtype=float),
                                                  cov=sc_process_noise, 
                                                  size=(nparticles))
        self.states = pred_states

        
    
    def update_gps(self, gps_obs):
        pred_gps_obs = np.array(self.states[:, 0:2])
        likelihood = misctools.mvnpdf(np.array([gps_obs]), pred_gps_obs, 
                np.array([self.parameters.state_likelihood_fn.parameters.gps_obs_noise]))
        self.weights *= likelihood
        self.weights /= self.weights.sum()
    
    def update_dvl(self, dvl_obs):
        pred_dvl_obs = np.array(self.states[:, 3:])
        npa_dvl_obs = np.array([dvl_obs])
        cov = np.array([self.parameters.state_likelihood_fn.parameters.dvl_obs_noise])
        loglikelihood = np.log(misctools.mvnpdf(npa_dvl_obs, pred_dvl_obs, cov) +
                            np.finfo(np.double).tiny)
        loglikelihood -= max(loglikelihood)
        likelihood = np.exp(loglikelihood)
        self.weights *= likelihood
        self.weights /= self.weights.sum()
        
    def update_svs(self, svs_obs):
        pred_svs_obs = self.states[:, 2].copy()
        # Fix the shape so it is two-dimensional
        pred_svs_obs.shape += (1,)
        loglikelihood = np.log(misctools.mvnpdf(np.array([[svs_obs]]), 
                                                pred_svs_obs, 
                                                np.array([[[0.2]]])) +
                               np.finfo(np.double).tiny)
        loglikelihood -= max(loglikelihood)
        likelihood = np.exp(loglikelihood)
        self.weights *= likelihood
        self.weights /= self.weights.sum()
    
    

def g500_kf_update(weights, states, covs, obs_matrix, obs_noise, z):
    upd_weights = weights.copy()
    upd_states = np.empty(states.shape)
    upd_covs = np.empty(covs.shape)
    for count in range(states.shape[0]):
        this_state = states[count]
        this_state.shape = (1,) + this_state.shape
        this_cov = covs[count]
        this_cov.shape = (1,) + this_cov.shape
        (upd_state, upd_covariance, kalman_info) = \
                blas_kf_update(this_state, this_cov, obs_matrix, obs_noise, z, False)
        #x_pdf = misctools.mvnpdf(x, mu, sigma)
        x_pdf = np.exp(-0.5*np.power(
                blas.dgemv(kalman_info.inv_sqrt_S, kalman_info.residuals), 2).sum(axis=1))/ \
                np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
        upd_weights[count] *= x_pdf
        upd_states[count] = upd_state
        upd_covs[count] = upd_covariance
    upd_weights /= upd_weights.sum()
    return upd_weights, upd_states, upd_covs
    

def g500_slam_fn_defs():
    # Vehicle state prediction
    state_markov_predict_fn_handle = None
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
    state_parameters = {"nparticles":50,
                        "ndims":6,
                        "resample_threshold":0.9}
    
    
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
    
    
##############################################################################
# Start ROS related functions
def add_ros_param(container, param):
    subst_names = {"teledyne_explorer_dvl" : "dvl",
                   "tritech_igc_gyro" : "imu",
                   "valeport_sound_velocity" : "svs"}
    if rospy.has_param(param):
        param_value = np.array(rospy.get_param(param))
    else:
        rospy.logfatal(param + " param not found")
    
    # Check if param is a tf frame - we need to prefix the tf with a unique
    # identifier
    if param[-3:] == "/tf":
        if param[0] == '/':
            param = param[1:]
        # select a substitution if specified
        sensor = param[0:param.find("/")]
        if sensor in subst_names:
            sensor = subst_names[sensor]
        param_name = sensor + "_tf_data"
        
    else:
        param_name = param[param.rfind('/')+1:]
    setattr(container, param_name, param_value)
    
    
def get_config():
    config = PARAMETERS()
    ros_param_list = ["teledyne_explorer_dvl/tf",
                      "tritech_igc_gyro/tf",
                      "valeport_sound_velocity/tf",
                      "navigator/dvl_bottom_covariance",
                      "navigator/dvl_water_covariance",
                      "navigator/gps_covariance",
                      "navigator/model_covariance",
                      "navigator/dvl_max_v",
                      "navigator/gps_update",
                      "navigator/gps_init_samples",
                      "navigator/check_sensors_period",
                      "navigator/dvl_max_period_error",
                      "navigator/svs_max_period_error",
                      "navigator/imu_max_period_error",
                      "navigator/max_init_time" ]
    for param in ros_param_list:
        add_ros_param(config, param)
    return config
# End ROS related functions    
##############################################################################