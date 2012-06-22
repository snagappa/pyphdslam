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

import ukf
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
    
    def x(self, idx=None):
        return self.get_states(idx, cols=0)
    def y(self, idx=None):
        return self.get_states(idx, cols=1)
    def z(self, idx=None):
        return self.get_states(idx, cols=2)
    def vx(self, idx=None):
        return self.get_states(idx, cols=3)
    def vy(self, idx=None):
        return self.get_states(idx, cols=4)
    def vz(self, idx=None):
        return self.get_states(idx, cols=5)
    
    def reset_states(self):
        self.states[:] = 0
        self.weights = 1/float(self.parameters.state_parameters["nparticles"])* \
                        np.ones(self.parameters.state_parameters["nparticles"])
                        
    def reset_maps(self):
        self.maps = [self.create_default_feature() 
                    for i in range(self.parameters.state_parameters["nparticles"])]
    
    
    def predict_state(self, ctrl_input, predict_to_time):
        if self.last_odo_predict_time==0:
            self.last_odo_predict_time = predict_to_time
            return
        delta_t = predict_to_time - self.last_odo_predict_time
        self.last_odo_predict_time = predict_to_time
        if delta_t < 0:
            #print "negative delta_t, ignoring"
            return
        #print "delta_t = ", str(delta_t), ". predicting new state..."
        predict_fn = self.parameters.state_markov_predict_fn
        pred_states, pred_covs = predict_fn.handle(self.states, 
                                                   self.covariances, 
                                                   ctrl_input, delta_t, 
                                                   predict_fn.parameters)
        self.states = pred_states
        self.covariances = pred_covs
        
    
    def update_gps(self, gps_obs):
        pred_gps_obs = np.array(self.states[:, 0:2])
        likelihood = misctools.mvnpdf(np.array([gps_obs]), pred_gps_obs, 
                np.array([self.parameters.state_likelihood_fn.parameters.gps_obs_noise]))
        self.weights *= likelihood
        self.weights /= self.weights.sum()
    
    def update_dvl(self, dvl_obs):
        USE_KF = False
        if USE_KF:
            obs_matrix = np.array([np.hstack(( np.zeros((3,3)), np.eye(3) ))])
            obs_noise = np.array([self.parameters.state_likelihood_fn.parameters.dvl_obs_noise])
            upd_weights, upd_states, upd_covs = \
                    g500_kf_update(self.weights, self.states, self.covariances, 
                                   obs_matrix, obs_noise, dvl_obs)
            self.weights = upd_weights
            self.states = upd_states
            self.covariances = upd_covs
        else:
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
        if False:
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
        else:
            self.states[:,2] = svs_obs
    
    
#state_markov_predict_fn
def g500_state_predict(states, covs, ctrl_input, delta_t, parameters):
    PREDICTION = "particles"
    trans_mat, sc_process_noise = trans_matrices(ctrl_input, delta_t, parameters)
    if PREDICTION=="ekf":
        # EKF prediction
        pred_states = _state_predict_(states, ctrl_input, delta_t, parameters, trans_mat)
        pred_covs = blas.dgemm(trans_mat, 
                               blas.dgemm(covs, trans_mat, TRANSPOSE_B=True, 
                                          beta=0.0), beta=0.0)
    elif PREDICTION=="ukf":
        # UKF prediction
        pred_states = np.empty(states.shape)
        pred_covs = np.empty(covs.shape)
        for count in range(states.shape[0]):
            #this_pred_state, this_pred_cov = ukf.ukf_predict(states[count], 
            #                                                 covs[count], 
            #                ctrl_input, sc_process_noise, _state_predict_, delta_t, parameters)
            this_pred_state, this_pred_cov = g500_ukf_prediction(states[count], 
                                                                 covs[count], 
                                                ctrl_input, delta_t, parameters)
            pred_states[count] = this_pred_state
            pred_covs[count] = this_pred_cov
    elif PREDICTION=="particles":
        pred_covs = np.zeros(covs.shape)
        pred_states = _state_predict_(states, ctrl_input, delta_t, parameters, trans_mat)
        awg_noise = np.random.multivariate_normal(mean=np.zeros(6, dtype=float),
                                                  cov=sc_process_noise, 
                                                  size=(states.shape[0]))
        pred_states += awg_noise
    return pred_states, pred_covs
    

def trans_matrices(ctrl_input, delta_t, parameters):
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
    process_noise = np.array([parameters.process_noise])
    # Compute the process noise as scale_matrix*process_noise*scale_matrix'
    sc_process_noise = blas.dgemm(scale_matrix, 
                   blas.dgemm(process_noise, scale_matrix, 
                              TRANSPOSE_B=True, beta=0.0), beta=0.0)[0]
    return trans_mat, sc_process_noise
    
def _state_predict_(states, ctrl_input, delta_t, parameters, trans_mat=None):
    if trans_mat==None:
        trans_mat, sc_process_noise = trans_matrices(ctrl_input, delta_t, parameters)
    # Multiply the transition matrix with each state
    pred_states = blas.dgemv(trans_mat, states)
    return pred_states


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
    
    

def g500_ukf_prediction(x, P, ctrl_input, delta_t, parameters, _alpha=1e-3, _beta=2, _kappa=0):
    #_L = x.shape[0]
    # UKF parameters
    #_lambda = _alpha**2 * (_L+_kappa) - _L
    #_gamma = (_L + _lambda)**0.5
    
    trans_mat, sc_process_noise = trans_matrices(ctrl_input, delta_t, parameters)
    P += sc_process_noise #+ 1e-1*np.eye(P.shape[0])
    
    # UKF prediction
    # Create the Sigma points
    (x_sigma, x_weight, P_weight) = createSigmaPoints(x, P, _alpha, _beta, _kappa)
    
    # Predict Sigma points and multiply by weight
    x_sigma_predicted = _state_predict_(x_sigma, ctrl_input, delta_t, 
                                        parameters, trans_mat)
    blas.dscal(x_weight, x_sigma_predicted)
    
    # Take the weighted mean of the Sigma points to get the predicted mean
    pred_state = np.add.reduce(x_sigma_predicted)
    
    # Generate the weighted Sigma covariance and add Q to get predicted cov
    pred_cov = evalSigmaCovariance(P_weight, x_sigma_predicted, pred_state) #+ sc_process_noise
    return pred_state, pred_cov

    
def createSigmaPoints(x, P, _alpha, _beta, _kappa):
    _L = x.shape[0]
    # UKF parameters
    _lambda = _alpha**2 * (_L+_kappa) - _L
    _gamma = (_L + _lambda)**0.5
    # Square root of scaled covariance matrix
    #sqrt_cov = _gamma*np.tril(blas.dpotrf(np.array([P]))[0])
    sqrt_cov = _gamma*np.linalg.cholesky(P)
    
    # Array of the sigma points
    x_plus = np.array([x+sqrt_cov[:,count] for count in range(_L)])
    x_minus = np.array([x-sqrt_cov[:,count] for count in range(_L)])
    sigma_x = np.vstack((x, x_minus, x_plus))
    
    # Array of the weights for each sigma point
    wt_mn = np.array([_lambda] + [0.5]*2*_L)/(_L + _lambda)
    wt_cv = wt_mn.copy()
    wt_cv[0] = wt_cv[0] + (1 - _alpha**2 + _beta);
    
    return (sigma_x, wt_mn, wt_cv)
    
    
def evalSigmaCovariance(wt_vector, sigma_x1, x1, sigma_x2=None, x2=None):
    #difference1 = [_sigma_x1 - x1 for _sigma_x1 in sigma_x1]
    difference1 = sigma_x1.copy()
    blas.daxpy(-1.0, np.array([x1]), difference1)
    if not (sigma_x2 is None):
        #difference2 = [_sigma_x2 - x2 for _sigma_x2 in sigma_x2]
        difference2 = sigma_x2.copy()
        blas.daxpy(-1.0, np.array([x2]), difference2)
    else:
        difference2 = difference1
    
    #sigma_cov = [this_wt_vector*_diff1.T*_diff2 for (this_wt_vector,_diff1, _diff2) in zip(wt_vector, difference1, difference2)]
    sigma_cov = blas.dger(difference1, difference2, wt_vector)
    return np.add.reduce(sigma_cov)