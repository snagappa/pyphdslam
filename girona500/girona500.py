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
from lib.common import misctools, blas
blas.SET_DEBUG(True)
from lib.phdfilter.phdfilter import fn_params, PARAMETERS
from lib.phdfilter import gmphdfilter
from lib import phdslam
import numpy as np

import rospy
import code
from lib.common.kalmanfilter import kf_predict_cov, kf_update, kf_update_cov, kf_update_x
import featuredetector
import copy

SLAM_FN_DEFS = collections.namedtuple("SLAM_FN_DEFS", 
                "state_markov_predict_fn state_obs_fn state_likelihood_fn \
                state__state_update_fn state_estimate_fn state_parameters \
                feature_markov_predict_fn feature_obs_fn \
                feature_likelihood_fn feature__state_update_fn clutter_fn \
                birth_fn ps_fn pd_fn feature_estimate_fn feature_parameters")

class G500_SLAM_FEATURE(gmphdfilter.GMPHD):
    def __init__(self, *args, **kwargs):
        super(G500_SLAM_FEATURE, self).__init__(*args, **kwargs)
    
    def phdPredict(self):
        #print type(self.states._state_)
        if self.states._covariance_.shape[0]:
            extra_cov = 0.1*np.eye(3)
            self.states._covariance_ += extra_cov[np.newaxis, :, :]
        #self.states._covariance_ += self.parameters.markov_predict_fn.handle(self.states, 
        #                        self.parameters.markov_predict_fn.parameters)

    def phdUpdate(self, observation_set):
        # Container for slam parent update
        slam_info = PARAMETERS()
        num_observations = observation_set.shape[0]
        if num_observations:
            z_dim = observation_set.shape[1]
        else:
            z_dim = 0
        
        if not self.weights.shape[0]:
            self._states_ = self.states.copy()
            self._weights_ = self.weights.copy()
            return
        
        detection_probability = self.parameters.pd_fn.handle(self.states, 
                                            self.parameters.pd_fn.parameters)
        #clutter_pdf = [self.clutter_fn.handle(_observation_, 
        #                                      self.clutter_fn.parameters) \
        #               for _observation_ in observation_set]
        clutter_pdf = self.parameters.clutter_fn.handle(observation_set, 
                                        self.parameters.clutter_fn.parameters)
        # Account for missed detection
        self._states_ = self.states.copy()
        self._weights_ = [self.weights*(1-detection_probability)]
        
        # SLAM,  step 1:
        slam_info.exp_sum__pd_predwt = np.exp(-self.weights.sum())
        
        # Split x and P out from the combined state vector
        detected_indices = detection_probability > 0.1
        detected_states = self.states[detected_indices]
        x = detected_states.state
        P = detected_states.covariance
        # Scale the weights by detection probability 
        weights = self.weights[detected_indices]*detection_probability[detected_indices]
        
        # SLAM, prep for step 2:
        slam_info.sum__clutter_with_pd_updwt = np.zeros(num_observations)
        
        if x.shape[0]:
            # Part of the Kalman update is common to all observation-updates
            x, P, kalman_info = kf_update(x, P, 
                                np.array([self.parameters.obs_fn.parameters.H]), 
                                np.array([self.parameters.obs_fn.parameters.R]), 
                                None, INPLACE=True)#USE_NP=0)
                
            # Container for the updated states
            new_gmstate = self.states.__class__(0)
            # Predicted observation from the current states
            pred_z = featuredetector.tf.relative(self.parameters.obs_fn.parameters.parent_state_xyz, 
                                                 self.parameters.obs_fn.parameters.parent_state_rpy, 
                                                 x)
            #print "PREDICTED Z:"
            #print pred_z
            # We need to update the states and find the updated weights
            for (_observation_, obs_count) in zip(observation_set, 
                                                  range(num_observations)):
                #new_x = copy.deepcopy(x)
                # Apply the Kalman update to get the new state - update in-place
                # and return the residuals
                new_x, residuals = kf_update_x(x, pred_z, 
                                            _observation_, kalman_info.kalman_gain,
                                            INPLACE=False)
                code.interact(local=locals())
                # Calculate the weight of the Gaussians for this observation
                # Calculate term in the exponent
                x_pdf = np.exp(-0.5*np.power(
                    blas.dgemv(kalman_info.inv_sqrt_S, residuals), 2).sum(axis=1))/ \
                    np.sqrt(kalman_info.det_S*(2*np.pi)**z_dim) 
                
                new_weight = weights*x_pdf
                # Normalise the weights
                normalisation_factor = clutter_pdf[obs_count] + new_weight.sum()
                new_weight /= normalisation_factor
                # SLAM, step 2:
                slam_info.sum__clutter_with_pd_updwt[obs_count] = \
                                                            normalisation_factor
                
                # Create new state with new_x and P to add to _states_
                new_gmstate.set(new_x, P)
                self._states_.append(new_gmstate)
                self._weights_ += [new_weight]
            
        else:
            slam_info.sum__clutter_with_pd_updwt = np.array(clutter_pdf)
            
        self._weights_ = np.concatenate(self._weights_)
        # SLAM, finalise:
        slam_info.likelihood = (slam_info.exp_sum__pd_predwt * 
                                slam_info.sum__clutter_with_pd_updwt.prod())
        return slam_info
        
    
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
        
        # Merge components
        self.phdMerge()
        
        # End of iteration call
        self.phdFlattenUpdate()
        
        # Create birth terms from measurements
        birth_states, birth_weights = self.phdGenerateBirth(observations)
        self.phdPredict()
        # Append birth terms to Gaussian mixture
        self.phdAppendBirth(birth_states, birth_weights)
        
        #return estimates
        return slam_info
        
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
        state_parameters["nparticles"] = 2*state_parameters["ndims"] + 1
        
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
        self.transition_matrix = np.array([np.eye(6)])
        
        # Override survival probability
        #self.parameters.ps_fn.handle = self.landmark_ps
        #self.parameters.feature_markov_predict_fn.handle = self.landmark_predict
        
        # Override probability of detection using own method
        self.sensor_fov = featuredetector.sensors.camera_fov()
        self.parameters.pd_fn.handle = self.camera_pd
        self.parameters.clutter_fn.handle = self.camera_clutter
        
    def create_default_feature(self):
        return G500_SLAM_FEATURE(self.parameters.feature_markov_predict_fn,
                                  self.parameters.feature_obs_fn,
                                  self.parameters.feature_likelihood_fn,
                                  self.parameters.feature__state_update_fn,
                                  self.parameters.clutter_fn,
                                  self.parameters.birth_fn,
                                  self.parameters.ps_fn,
                                  self.parameters.pd_fn,
                                  self.parameters.feature_estimate_fn,
                                  self.parameters.feature_parameters)
    
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
        trans_mat = self.transition_matrix
        process_noise = np.array([
            self.parameters.state_markov_predict_fn.parameters.process_noise])
        
        rot_mat = delta_t * featuredetector.tf.rotation_matrix(ctrl_input)
        trans_mat[0,0:3,3:] = rot_mat
        
        scale_matrix = np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))
        sc_process_noise = np.dot(scale_matrix, np.dot(process_noise, scale_matrix.T)).squeeze() #+ delta_t/10*np.eye(6)
        return trans_mat, sc_process_noise
    
    def predict(self, ctrl_input, predict_to_time):
        USE_KF = True
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
        #pred_states2 = np.dot(trans_mat[0], self.states.T).T
        nparticles = self.parameters.state_parameters["nparticles"]
        ndims = self.parameters.state_parameters["ndims"]
        if not USE_KF:
            pred_states += np.random.multivariate_normal(mean=np.zeros(ndims, dtype=float),
                                                         cov=sc_process_noise, 
                                                         size=(nparticles))
        self.states = pred_states
        self.covariances = kf_predict_cov(self.covariances, trans_mat, 
                                          sc_process_noise)
        
        states_xyz = np.array(pred_states[:,0:3])
        #Calculate the rotation matrix to store for the map update
        rot_mat = featuredetector.tf.rotation_matrix(ctrl_input)
        # Copy the predicted states to the "parent state" attribute and 
        # perform a prediction for the map
        for i in range(self.parameters.state_parameters["nparticles"]):
            #self.maps[i].parameters.obs_fn.H = self.trans_matrices(-ctrl_input, 1.0)[0]
            setattr(self.maps[i].parameters.obs_fn.parameters, 
                    "parent_state_xyz", states_xyz[i])
            setattr(self.maps[i].parameters.obs_fn.parameters, 
                    "parent_state_rpy", ctrl_input)
            self.maps[i].parameters.obs_fn.H = rot_mat
            setattr(self.maps[i].parameters.pd_fn.parameters, 
                    "parent_state_xyz", states_xyz[i])
            setattr(self.maps[i].parameters.pd_fn.parameters, 
                    "parent_state_rpy", ctrl_input)
            setattr(self.maps[i].parameters.birth_fn.parameters, 
                    "parent_state_xyz", states_xyz[i])
            setattr(self.maps[i].parameters.birth_fn.parameters, 
                    "parent_state_rpy", ctrl_input)
        
        #self.parameters.state_parameters.delta_t = delta_t
        # PHD Prediction is not necessary - ps = 1, Q = 0
        #[self.maps[i].phdPredict() 
        #        for i in range(self.parameters.state_parameters["nparticles"])]
        
    
    def update_gps(self, gps_obs):
        USE_KF = True
        if USE_KF == True:
            obs_matrix = np.array([self.parameters.state_obs_fn.parameters.gpsH])
            obs_noise = np.array([self.parameters.state_likelihood_fn.parameters.gps_obs_noise])
            upd_weights, upd_states, upd_covs = \
                    g500_kf_update(self.weights, self.states, self.covariances,
                                   obs_matrix, obs_noise, gps_obs)
            self.weights = upd_weights
            self.states = upd_states
            self.covariances = upd_covs
        else:
            pred_gps_obs = np.array(self.states[:, 0:2])
            likelihood = misctools.mvnpdf(np.array([gps_obs]), pred_gps_obs, 
                    np.array([self.parameters.state_likelihood_fn.parameters.gps_obs_noise]))
            self.weights *= likelihood
            self.weights /= self.weights.sum()
    
    def update_dvl(self, dvl_obs):
        USE_KF = True
        if USE_KF:
            obs_matrix = np.array([self.parameters.state_obs_fn.parameters.dvlH])
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
                                np.finfo(np.double).eps)
            loglikelihood -= max(loglikelihood)
            likelihood = np.exp(loglikelihood)
            self.weights *= likelihood
            self.weights /= self.weights.sum()
        
    def update_svs(self, svs_obs):
        OVERWRITE_DEPTH = True
        if OVERWRITE_DEPTH:
            self.states[:, 2] = svs_obs
        else:
            pred_svs_obs = self.states[:, 2].copy()
            # Fix the shape so it is two-dimensional
            pred_svs_obs.shape += (1,)
            loglikelihood = np.log(misctools.mvnpdf(np.array([[svs_obs]]), 
                                                    pred_svs_obs, 
                                                    np.array([[[0.2]]])) +
                                   np.finfo(np.double).eps)
            loglikelihood -= max(loglikelihood)
            likelihood = np.exp(loglikelihood)
            self.weights *= likelihood
            self.weights /= self.weights.sum()
    
    def update_with_features(self, observation_set, update_to_time):
        #print "Observed:"
        #print observation_set
        #state_xyz = self.maps[0].parameters.obs_fn.parameters.parent_state_xyz
        #state_rpy = self.maps[0].parameters.obs_fn.parameters.parent_state_rpy
        state_rpy = np.array([0, 0, self.maps[0].parameters.obs_fn.parameters.parent_state_rpy[2]])
        # Extract covariance
        #obs_cov = observation_set[:,3:].copy()
        # Extract states
        if observation_set.shape[0]:
            observation_set = observation_set[:,0:3].copy()
        else:
            observation_set = np.empty(0)
        #observation_set = np.array(observation_set[:, [1, 0, 2]], order='C')
        feature_abs_posn = np.array([self.weights[i]*featuredetector.tf.absolute(self.maps[i].parameters.obs_fn.parameters.parent_state_xyz, state_rpy, observation_set) for i in range(len(self.maps))])
        feature_abs_posn = feature_abs_posn.sum(axis=0)
        
        #print "Absolute (inverse):"
        #print feature_abs_posn
        #print "state xyz:"
        #print state_xyz
        
        [self.maps[i].phdIterate(observation_set) for i in range(self.weights.shape[0])]
        sum_weights = [self.maps[i].intensity() for i in range(self.weights.shape[0])]
        print "Tracking ", np.mean(sum_weights), " landmarks."
        #print "individual map intensities:"
        #print sum_weights
        
        
    def resample(self):
        if self.parameters.state_parameters["resample_threshold"] < 0:
            return
        # Effective number of particles
        eff_nparticles = 1/np.power(self.weights, 2).sum()
        resample_threshold = (
                eff_nparticles/self.parameters.state_parameters["nparticles"])
        # Check if we have particle depletion
        if (resample_threshold > 
                    self.parameters.state_parameters["resample_threshold"]):
            return
        # Otherwise we need to resample
        max_wt_index = self.weights.argmax()
        max_wt_state = self.states[max_wt_index].copy()
        max_wt_state[4:] = 0
        max_wt_map = self.maps[max_wt_index].copy()
        
        resample_index = misctools.get_resample_index(self.weights, 
                            self.parameters.state_parameters["nparticles"]-1)
        # self.states is a numpy array so the indexing operation forces a copy
        resampled_states = self.states[resample_index]
        resampled_states.resize((resampled_states.shape[0]+1, resampled_states.shape[1]))
        resampled_states[-1] = max_wt_state
        resampled_maps = [self.maps[i].copy() for i in resample_index]
        resampled_maps.append(max_wt_map)
        resampled_weights = (
          np.ones(self.parameters.state_parameters["nparticles"], dtype=float)*
          1/float(self.parameters.state_parameters["nparticles"]))
        
        self.weights = resampled_weights
        self.states = resampled_states
        self.maps = resampled_maps
        
    def landmark_ps(self, states, parameters):
        ps = np.ones(states._state_.shape[0])
        return ps
    
    def landmark_predict(self, states, parameters):
        if states._covariance_.shape[0]:
            extra_cov = 1e-3*np.eye(3)
            states._covariance_ += extra_cov[np.newaxis,:,:]
            return states
        
    def camera_pd(self, states, parameters):
        # Transform points to local frame
        rel_landmarks = featuredetector.tf.relative(parameters.parent_state_xyz, 
                                                    parameters.parent_state_rpy, 
                                                    states.state())
        return self.sensor_fov.is_visible(rel_landmarks).astype(np.float)*parameters.pd
        
    def camera_clutter(self, observations, parameters):
        if observations.shape[0]:
            observations = observations[:,0]
        return parameters.intensity*self.sensor_fov.z_prob(observations)
        #return parameters.intensity*1/(1/3.0*
        #    self.sensor_fov.fov_far_m*
        #    self.sensor_fov.get_rect__half_width_height(self.sensor_fov.fov_far_m).prod())
"""
def feature_relative_position(vehicle_xyz, vehicle_rpy, features_xyz):
    if not features_xyz.shape[0]: return np.empty(0)
    relative_position = features_xyz - vehicle_xyz
    r, p, y = 0, 1, 2
    c = np.cos(vehicle_rpy)
    s = np.sin(vehicle_rpy)
    rotation_matrix = np.array([
                [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
                 [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
                 [-s[p], s[r]*c[p], c[r]*c[p] ]]])
    
    relative_position = blas.dgemv(rotation_matrix, relative_position)
    #np.dot(rotation_matrix, relative_position.T).T
    return relative_position

def feature_absolute_position(vehicle_xyz, vehicle_rpy, features_xyz):
    if not features_xyz.shape[0]: return np.empty(0)
    r, p, y = 0, 1, 2
    c = np.cos(vehicle_rpy)
    s = np.sin(vehicle_rpy)
    rotation_matrix = np.array([
                [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
                 [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
                 [-s[p], s[r]*c[p], c[r]*c[p] ]]])
    
    absolute_position = blas.dgemv(rotation_matrix, features_xyz) + vehicle_xyz
    return absolute_position
"""

def g500_kf_update(weights, states, covs, obs_matrix, obs_noise, z):
    upd_weights = weights.copy()
    #upd_states = np.empty(states.shape)
    #upd_covs = np.empty(covs.shape)
    # Covariance is the same for all the particles
    upd_cov0, kalman_info = kf_update_cov(np.array([covs[0]]), obs_matrix, obs_noise, False)
    upd_covs = np.repeat(upd_cov0, covs.shape[0], axis=0)
    # Update the states
    pred_z = blas.dgemv(obs_matrix, states)
    upd_states, residuals = kf_update_x(states, pred_z, z, kalman_info.kalman_gain)
    # Evaluate the new weight
    x_pdf = np.exp(-0.5*np.power(
                blas.dgemv(kalman_info.inv_sqrt_S, residuals), 2).sum(axis=1))/ \
                np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
    upd_weights = weights * x_pdf
    upd_weights /= upd_weights.sum()
    
    """
    for count in range(states.shape[0]):
        this_state = states[count]
        this_state.shape = (1,) + this_state.shape
        this_cov = covs[count]
        this_cov.shape = (1,) + this_cov.shape
        (upd_state, upd_covariance, kalman_info) = \
                kf_update(this_state, this_cov, obs_matrix, obs_noise, z, False)
        #x_pdf = misctools.mvnpdf(x, mu, sigma)
        x_pdf = np.exp(-0.5*np.power(
                blas.dgemv(kalman_info.inv_sqrt_S, kalman_info.residuals), 2).sum(axis=1))/ \
                np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
        upd_weights[count] *= x_pdf
        upd_states[count] = upd_state
        upd_covs[count] = upd_covariance
    upd_weights /= upd_weights.sum()
    """
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
    state_obs_fn_parameters.gpsH = np.hstack(( np.eye(2), np.zeros((2,4)) ))
    state_obs_fn_parameters.dvlH = np.hstack(( np.zeros((3,3)), np.eye(3) ))
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
    state_parameters = {"nparticles":13,
                        "ndims":6,
                        "resample_threshold":-1}
    
    
    # Parameters for the PHD filter
    feature_parameters = {"max_terms":100, 
                          "elim_threshold":1e-4, 
                          "merge_threshold":3,
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
    feature_obs_fn_parameters.R = 0.1*np.eye(ndims)
    feature_obs_fn_parameters.parent_state_xyz = np.zeros(3)
    feature_obs_fn_parameters.parent_state_rpy = np.zeros(3)
    feature_obs_fn = fn_params(feature_obs_fn_handle, 
                               feature_obs_fn_parameters)
    
    # Likelihood function - not used for the GM PHD filter
    feature_likelihood_fn = fn_params()
    
    # Landmark state update function - not used
    feature__state_update_fn = fn_params()
    
    # Clutter function
    clutter_fn_handle = None#gmphdfilter.uniform_clutter
    clutter_fn_parameters = PARAMETERS()
    clutter_fn_parameters.intensity = 0.01
    # Range should be the field of view of the sensor
    clutter_fn_parameters.range = [[-1, 1], [-1, 1], [-1, 1]]
    clutter_fn = fn_params(clutter_fn_handle, clutter_fn_parameters)
    
    # Birth function
    birth_fn_handle = camera_birth
    birth_fn_parameters = PARAMETERS()
    birth_fn_parameters.intensity = 0.001
    birth_fn_parameters.obs2state = lambda x: np.array(x)
    birth_fn_parameters.parent_state_xyz = np.zeros(3)
    birth_fn_parameters.parent_state_rpy = np.zeros(3)
    birth_fn_parameters.R = 0.1*np.eye(3)
    birth_fn = fn_params(birth_fn_handle, birth_fn_parameters)
    
    # Survival/detection probability
    ps_fn_handle = gmphdfilter.constant_survival
    ps_fn_parameters = PARAMETERS()
    ps_fn_parameters.ps = 1
    ps_fn = fn_params(ps_fn_handle, ps_fn_parameters)
    pd_fn_handle = None
    pd_fn_parameters = PARAMETERS()
    pd_fn_parameters.width = 2.0
    pd_fn_parameters.depth = 3.0
    pd_fn_parameters.height = 1.0
    pd_fn_parameters.pd = 0.98
    pd_fn_parameters.parent_state_xyz = np.zeros(3)
    pd_fn_parameters.parent_state_rpy = np.zeros(3)
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
    

#def camera_pd(states, parameters):
#    # Rotate the states by parent state orientation
#    featuredetector.tf.relative(parameters.parent_state_xyz, 
#                                parameters.parent_state_rpy, 
#                                states)
    

def camera_birth(z, parameters):
    # Convert the relative z to absolute z
    abs_z = featuredetector.tf.absolute(parameters.parent_state_xyz, 
                                parameters.parent_state_rpy, z)
    #
    birth_states, birth_weights = gmphdfilter.measurement_birth(abs_z, parameters)
    return birth_states, birth_weights

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
