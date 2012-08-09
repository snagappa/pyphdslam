# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:15:54 2012

@author: snagappa
"""


USE_CYTHON = False


import numpy as np
from lib.common import misctools, blas
#from featuredetector import sensors, tf
import featuredetector
from lib.common.kalmanfilter import kf_predict_cov
from lib.common.kalmanfilter import np_kf_update_cov, kf_update_cov, kf_update_x
from collections import namedtuple
import copy
import code

DEBUG = True
blas.SET_DEBUG(False)

### CYTHON STUFF ###
if USE_CYTHON:
    exec """
    import cython
    cimport numpy as np
    
    @cython.boundscheck(False)
    @cython.wraparound(False)"""

class STRUCT(object):
    pass

SAMPLE = namedtuple("SAMPLE", "weight state covariance")
    
class GMPHD(object):
    def __init__(self):
        self.weights = np.zeros(0)
        self.states = np.zeros((0, 3))
        self.covs = np.zeros((0, 3, 3))
        self.parent_ned = np.zeros(3)
        self.parent_rpy = np.zeros(3)
        
        self._estimate_ = SAMPLE(np.zeros(0), 
                                 np.zeros((0, 3)), np.zeros((0, 3, 3)))
        
        self.vars = STRUCT()
        self.vars.prune_threshold = 1e-4
        self.vars.merge_threshold = 1
        self.vars.birth_intensity = 0.1
        self.vars.clutter_intensity = 1
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = False
        
        self.sensors = STRUCT()
        self.sensors.camera = featuredetector.sensors.camera_fov()
    
    def copy(self):
        new_object = GMPHD()
        new_object.weights = self.weights.copy()
        new_object.states = self.states.copy()
        new_object.covs = self.states.copy()
        new_object.parent_ned = self.parent_ned.copy()
        new_object.parent_rpy = self.parent_rpy.copy()
        
        new_object._estimate_ = self.estimate()
        new_object.vars = copy.copy(self.vars)
        new_object.flags = copy.copy(self.flags)
        new_object.sensors.camera = copy.deepcopy(self.sensors.camera)
    
    def set_states(self, ptr_weights, ptr_states, ptr_covs):
        self.flags.ESTIMATE_IS_VALID = False
        self.weights = ptr_weights
        self.states = ptr_states
        self.covs = ptr_covs
    
    def set_parent(self, parent_ned, parent_rpy):
        self.parent_ned = parent_ned.copy()
        self.parent_rpy = parent_rpy.copy()
    
    def birth(self, features_rel, features_cv=None, APPEND=False):
        b_wt, b_st, b_cv = self.camera_birth(self.parent_ned, self.parent_rpy, 
                                             features_rel, features_cv)
        if APPEND and b_wt.shape[0]:
            self.flags.ESTIMATE_IS_VALID = False
            self.append(b_wt, b_st, b_cv)
        return (b_wt, b_st, b_cv)
    
    def predict(self, *args, **kwargs):
        #self.flags.ESTIMATE_IS_VALID = False
        #if self.covariance.shape[0]:
        #    extra_cov = 0.1*np.eye(3)
        #    self.covariance += extra_cov[np.newaxis, :, :]
        pass
    
    def update(self, observations, observation_noise):
        self.flags.ESTIMATE_IS_VALID = False
        # Container for slam parent update
        slam_info = STRUCT()
        slam_info.likelihood = 1
        num_observations, z_dim = (observations.shape + (3,))[0:2]
        
        if not self.weights.shape[0]:
            return slam_info
        
        detection_probability = self.camera_pd(self.parent_ned, 
                                               self.parent_rpy, self.states)
        clutter_pdf = self.camera_clutter(observations)
        # Account for missed detection
        prev_weights = self.weights.copy()
        prev_states = self.states.copy()
        prev_covs = self.covs.copy()
        
        updated = STRUCT()
        updated.weights = [self.weights*(1-detection_probability)]
        updated.states = [self.states]
        updated.covs = [self.covs]
        #ZZ SLAM,  step 1:
        slam_info.exp_sum__pd_predwt = np.exp(-prev_weights.sum())
        # Do the update only for detected landmarks
        detected_indices = detection_probability > 0
        detected = STRUCT()
        detected.weights = ( prev_weights[detected_indices]*
                             detection_probability[detected_indices] )
        detected.states = prev_states[detected_indices]
        detected.covs = prev_covs[detected_indices]
        
        # SLAM, prep for step 2:
        slam_info.sum__clutter_with_pd_updwt = np.zeros(num_observations)
        
        if detected.weights.shape[0]:
            # Covariance update part of the Kalman update is common to all 
            # observation-updates
            if observations.shape[0]:
                h_mat = featuredetector.tf.relative_rot_mat(self.parent_rpy)
                # Predicted observation from the current states
                pred_z = featuredetector.tf.relative(self.parent_ned, 
                    self.parent_rpy, detected.states)
                observation_noise = observation_noise[0]
                detected.covs, kalman_info = kf_update_cov(detected.covs, 
                                                           h_mat, 
                                                           observation_noise, 
                                                           INPLACE=True)
            # We need to update the states and find the updated weights
            for (_observation_, obs_count) in zip(observations, 
                                                  range(num_observations)):
                #new_x = copy.deepcopy(x)
                # Apply the Kalman update to get the new state - 
                # update in-place and return the residuals
                upd_states, residuals = kf_update_x(detected.states, pred_z, 
                                                    _observation_, 
                                                    kalman_info.kalman_gain,
                                                    INPLACE=False)
                # Calculate the weight of the Gaussians for this observation
                # Calculate term in the exponent
                x_pdf = np.exp(-0.5*np.power(
                    blas.dgemv(kalman_info.inv_sqrt_S, 
                               residuals), 2).sum(axis=1))/ \
                    np.sqrt(kalman_info.det_S*(2*np.pi)**z_dim) 
                
                upd_weights = detected.weights*x_pdf
                # Normalise the weights
                normalisation_factor = ( clutter_pdf[obs_count] + 
                                         upd_weights.sum() )
                upd_weights /= normalisation_factor
                # SLAM, step 2:
                slam_info.sum__clutter_with_pd_updwt[obs_count] = \
                    normalisation_factor
                
                # Create new state with new_x and P to add to _states_
                updated.weights += [upd_weights]
                updated.states += [upd_states]
                updated.covs += [detected.covs.copy()]
            
        else:
            slam_info.sum__clutter_with_pd_updwt = np.array(clutter_pdf)
            
        self.weights = np.concatenate(updated.weights)
        self.states = np.concatenate(updated.states)
        self.covs = np.concatenate(updated.covs)
        # SLAM, finalise:
        slam_info.likelihood = (slam_info.exp_sum__pd_predwt * 
                                slam_info.sum__clutter_with_pd_updwt.prod())
        assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        return slam_info
    
    def estimate(self):
        if not self.flags.ESTIMATE_IS_VALID:
            self.flags.ESTIMATE_IS_VALID = True
            intensity = self.weights.sum()
            num_targets = min((intensity, self.weights.shape[0]))
            num_targets = int(round(num_targets))
            if num_targets:
                inds = np.flipud(self.weights.argsort())
                inds = inds[0:num_targets]
                est_weights = self.weights[inds]
                est_states = self.states[inds]
                est_covs = self.covs[inds]
                # Discard states with low weight
                valid_idx = np.where(est_weights>0.5)[0]
                self._estimate_ = SAMPLE(est_weights[valid_idx],
                                         est_states[valid_idx], 
                                         est_covs[valid_idx])
            else:
                self._estimate_ = SAMPLE(np.zeros(0), 
                                         np.zeros((0, 3)), np.zeros((0, 3, 3)))
        return self._estimate_
        #return SAMPLE(self._estimate_.weight.copy(), 
        #              self._estimate_.state.copy(), 
        #              self._estimate_.covariance.copy())
    
    def prune(self):
        if self.vars.prune_threshold <= 0 or (not self.weights.shape[0]):
            return
        self.flags.ESTIMATE_IS_VALID = False
        valid_idx = self.weights >= self.vars.prune_threshold
        self.weights = self.weights[valid_idx]
        self.states = self.states[valid_idx]
        self.covs = self.covs[valid_idx]
        assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
    
    def merge(self):
        if (self.vars.merge_threshold < 0) or (self.weights.shape[0] < 2):
            return
        self.flags.ESTIMATE_IS_VALID = False
        merged_wts = []
        merged_sts = []
        merged_cvs = []
        num_remaining_components = self.weights.shape[0]
        while num_remaining_components:
            max_wt_index = self.weights.argmax()
            max_wt_state = np.array([self.states[max_wt_index]])
            max_wt_cov = np.array([self.covs[max_wt_index]])
            mahalanobis_dist = misctools.mahalanobis(max_wt_state, 
                                                     max_wt_cov, 
                                                     self.states)
            merge_list_indices = ( np.where(mahalanobis_dist <= 
                                                self.vars.merge_threshold)[0] )
            new_wt, new_st, new_cv = misctools.merge_states(
                                            self.weights[merge_list_indices], 
                                            self.states[merge_list_indices],
                                            self.covs[merge_list_indices])
            merged_wts += [new_wt]
            merged_sts += [new_st]
            merged_cvs += [new_cv]
            # Remove merged states from the list
            #self.weights = np.delete(self.weights, merge_list_indices)
            #self.states = np.delete(self.states, merge_list_indices, 0)
            #self.covs = np.delete(self.covs, merge_list_indices, 0)
            retain_idx = misctools.gen_retain_idx(self.weights.shape[0], 
                                                  merge_list_indices)
            self.weights = self.weights[retain_idx]
            self.states = self.states[retain_idx]
            self.covs = self.covs[retain_idx]
            num_remaining_components = self.weights.shape[0]
        
        self.set_states(np.array(merged_wts), 
                        np.array(merged_sts), np.array(merged_cvs))
        assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
        
    def append(self, weights, states, covs):
        self.flags.ESTIMATE_IS_VALID = False
        self.weights = np.hstack((self.weights, weights))
        if DEBUG:
            assert len(states.shape) == 2, "states must be a nxm ndarray"
            assert len(covs.shape) == 3, "covs must be a nxmxm ndarray"
        self.states = np.vstack((self.states, states))
        self.covs = np.vstack((self.covs, covs))
        assert self.weights.shape[0] == self.states.shape[0] == self.covs.shape[0], "Lost states!!"
    
    #####################################
    ## Default iteration of PHD filter ##
    #####################################
    def iterate(self, observations, obs_noise):
        self.predict()
        slam_info = self.update(observations, obs_noise)
        self.estimate()
        self.prune()
        self.birth(observations, obs_noise, APPEND=True)
        self.merge()
        return slam_info
    
    def intensity(self):
        return self.weights.sum()
    
    def camera_birth(self, parent_ned, parent_rpy, features_rel, 
                     features_cv=None):
        if not features_rel.shape[0]:
            birth_wt = np.empty(0)
            birth_st = np.empty((0, 3))
            birth_cv = np.empty((0, 3, 3))
        else:
            birth_wt = self.vars.birth_intensity*np.ones(features_rel.shape[0])
            birth_st = self.sensors.camera.rel_to_abs(parent_ned, parent_rpy, 
                                                      features_rel)
            if features_cv is None:
                features_cv = np.repeat([np.eye(3)], features_rel.shape[0], 0)
            else:
                features_cv = features_cv.copy()
            birth_cv = features_cv
        return (birth_wt, birth_st, birth_cv)
        
    def camera_pd(self, parent_ned, parent_rpy, features_abs):
        return self.sensors.camera.pdf_detection(parent_ned, 
                                                 parent_rpy, features_abs)
    
    def camera_clutter(self, observations):
        return self.sensors.camera.z_prob(observations[:, 0])
        
    
###############################################################################
###############################################################################


class PHDSLAM(object):
    def __init__(self):
        self.map_instance = GMPHD
        
        self.flags = STRUCT()
        self.flags.ESTIMATE_IS_VALID = True
        
        self.vars = STRUCT()
        self.vars.ndims = 6
        self.vars.nparticles = 7#2*self.vars.ndims + 1
        self.vars.resample_threshold = -1
        
        self.vars.F = np.array([np.eye(self.vars.ndims)])
        self.vars.Q = None
        self.vars.gpsH = None
        self.vars.dvlH = None
        self.vars.gpsR = None
        self.vars.dvl_w_R = None
        self.vars.dvl_b_R = None
        
        self.vehicle = STRUCT()
        self.vehicle.weights = 1.0/self.vars.nparticles * \
            np.ones(self.vars.nparticles)
        self.vehicle.states = np.zeros((self.vars.nparticles, self.vars.ndims))
        self.vehicle.covs = np.zeros((self.vars.nparticles, 
                                      self.vars.ndims, self.vars.ndims))
        self.vehicle.maps = [self.map_instance() 
            for i in range(self.vars.nparticles)]
        
        self.last_time = STRUCT()
        self.last_time.predict = 0
        #self.last_time.gps = 0
        #self.last_time.dvl = 0
        #self.last_time.svs = 0
        #self.last_time.map = 0
        
        self._estimate_ = STRUCT()
        self._estimate_.vehicle = STRUCT()
        self._estimate_.vehicle.ned = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.vehicle.vel_xyz = SAMPLE(1, np.zeros(3), 
                                                 np.zeros((3, 3)))
        self._estimate_.vehicle.rpy = SAMPLE(1, np.zeros(3), np.zeros((3, 3)))
        self._estimate_.map = SAMPLE(np.zeros(0), 
                                     np.zeros((0, 3)), np.zeros((0, 3, 3)))
    
    def set_parameters(self, Q, gpsH, gpsR, dvlH, dvl_b_R, dvl_w_R):
        self.vars.Q = Q
        self.vars.gpsH = gpsH
        self.vars.dvlH = dvlH
        self.vars.gpsR = gpsR
        self.vars.dvl_b_R = dvl_b_R
        self.vars.dvl_w_R = dvl_w_R
        
    def set_states(self, weights=None, states=None):
        if (weights is None) and (states is None):
            return
        new_weights = self.vehicle.weights if weights is None else weights
        new_states = self.vehicle.states if states is None else states
        assert new_weights.shape[0] == new_states.shape[0], (
            "Number of elements must match" )
        assert len(new_weights.shape) == 1 and len(new_states.shape) == 2, (
            "New weights must be 1D and new states must be 2D")
        self.vehicle.states = new_states
        self.vehicle.weights = new_weights
    
    def reset_states(self):
        self.vehicle.states[:] = 0
        self.vehicle.weights = 1.0/float(self.vars.nparticles)* \
            np.ones(self.vars.nparticles)
    
    def trans_matrices(self, ctrl_input, delta_t):
        # Get process noise
        trans_mat = self.vars.F
        process_noise = self.vars.Q
        
        rot_mat = delta_t * featuredetector.tf.rotation_matrix(ctrl_input)
        trans_mat[0, 0:3, 3:] = rot_mat
        
        scale_matrix = np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))
        sc_process_noise = np.dot(scale_matrix, 
            np.dot(process_noise, scale_matrix.T)).squeeze() \
            + delta_t/10*np.eye(6)
        return trans_mat, sc_process_noise
    
    def predict(self, ctrl_input, predict_to_time):
        if self.last_time.predict == 0:
            self.last_time.predict = predict_to_time
            return
        delta_t = predict_to_time - self.last_time.predict
        self.last_time.predict = predict_to_time
        if delta_t < 0:
            return
        
        # Predict states
        trans_mat, sc_process_noise = self.trans_matrices(ctrl_input, delta_t)
        #pred_states = blas.dgemv(trans_mat, self.vehicle.states)
        pred_states = np.array( 
            np.dot(trans_mat[0], self.vehicle.states.T).T, order='C')
        self.vehicle.states = pred_states
        # Predict covariance
        #code.interact(local=locals())
        self.vehicle.covs = kf_predict_cov(self.vehicle.covs, trans_mat, 
                                          sc_process_noise)
        
        # Copy the particle state to the PHD parent state
        parent_ned = np.array(pred_states[:, 0:3])
        #Calculate the rotation matrix to store for the map update
        rot_mat = featuredetector.tf.rotation_matrix(ctrl_input)
        # Copy the predicted states to the "parent state" attribute and 
        # perform a prediction for the map
        for i in range(self.vars.nparticles):
            self.vehicle.maps[i].parent_ned = parent_ned[i]
            self.vehicle.maps[i].parent_rpy = ctrl_input
            self.vehicle.maps[i].vars.H = rot_mat
            # self.vehicle.maps.predict()  # Not needed
    
    def _kf_update_(self, weights, states, covs, h_mat, r_mat, z):
        # predicted observations
        #pred_z = blas.dgemv(h_mat, states)
        pred_z = np.array(np.dot(h_mat, states.T).T, order='C')
        # covariance is the same for all states, do the update for one matrix
        
        upd_weights = weights.copy()
        upd_cov0, kalman_info = np_kf_update_cov(covs[0], h_mat, r_mat, False)
        
        upd_covs = np.repeat(np.array([upd_cov0]), covs.shape[0], axis=0)
        # Update the states
        upd_states, residuals = kf_update_x(states, pred_z, z, 
            np.array([kalman_info.kalman_gain]))
        if not upd_states.flags.c_contiguous:
            upd_states = np.array(upd_states, order='C')
        # Evaluate the new weight
        #x_pdf = np.exp(-0.5*np.power(
        #    blas.dgemv(np.array([kalman_info.inv_sqrt_S]), residuals), 2).sum(axis=1))/ \
        #    np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
        x_pdf = np.exp(-0.5*np.power(
            np.dot(kalman_info.inv_sqrt_S, residuals.T).T, 2).sum(axis=1))/ \
            np.sqrt(kalman_info.det_S*(2*np.pi)**z.shape[0])
        upd_weights = weights * x_pdf
        upd_weights /= upd_weights.sum()
        return upd_weights, upd_states, upd_covs
    
    def update_gps(self, gps):
        self.flags.ESTIMATE_IS_VALID = False
        h_mat = self.vars.gpsH #np.array([self.vars.gpsH])
        r_mat = self.vars.gpsR #np.array([self.vars.gpsR])
        upd_weights, upd_states, upd_covs = \
                self._kf_update_(self.vehicle.weights, self.vehicle.states, 
                                 self.vehicle.covs, h_mat, r_mat, gps)
        self.vehicle.weights = upd_weights
        self.vehicle.states = upd_states
        self.vehicle.covs = upd_covs
    
    def update_dvl(self, dvl, mode):
        self.flags.ESTIMATE_IS_VALID = False
        assert mode in ['b', 'w'], "Specify (b)ottom or (w)ater for dvl update"
        if mode == 'b':
            r_mat = self.vars.dvl_b_R #np.array([self.vars.dvl_b_R])
        else:
            r_mat = self.vars.dvl_w_R #np.array([self.vars.dvl_w_R])
        h_mat = self.vars.dvlH #np.array([self.vars.dvlH])
        upd_weights, upd_states, upd_covs = \
                self._kf_update_(self.vehicle.weights, self.vehicle.states, 
                                 self.vehicle.covs, h_mat, r_mat, dvl)
        self.vehicle.weights = upd_weights
        self.vehicle.states = upd_states
        self.vehicle.covs = upd_covs
    
    def update_svs(self, svs):
        self.flags.ESTIMATE_IS_VALID = False
        self.vehicle.states[:, 2] = svs
    
    def update_features(self, features):
        self.flags.ESTIMATE_IS_VALID = False
        if features.shape[0]:
            features_pos = features[:, 0:3].copy()
            features_noise = np.array([np.diag(features[i, 3:6]) 
                for i in range(features.shape[0])])
        else:
            features_pos = np.empty((0, 3))
            features_noise = np.empty((0, 3, 3))
        slam_info = [self.vehicle.maps[i].iterate(features_pos, features_noise) 
            for i in range(self.vars.nparticles)]
        slam_weight_update = np.array([slam_info[i].likelihood])
        self.vehicle.weights *= slam_weight_update/slam_weight_update.sum()
    
    def estimate(self):
        if not self.flags.ESTIMATE_IS_VALID:
            state, cov = misctools.sample_mn_cv(self.vehicle.states, 
                                                self.vehicle.weights)
            self._estimate_.vehicle.ned = SAMPLE(1, state[0: 3], cov[0:3, 0:3])
            max_weight_idx = np.argmax(self.vehicle.weights)
            self._estimate_.map = self.vehicle.maps[max_weight_idx].estimate()
            self.flags.ESTIMATE_IS_VALID = True
        return self._estimate_
    
    def resample(self):
        # Effective number of particles
        nparticles = self.vars.nparticles
        eff_nparticles = 1/np.power(self.vehicle.weights, 2).sum()
        resample_threshold = eff_nparticles/nparticles
        # Check if we have particle depletion
        if (resample_threshold > self.vars.resample_threshold):
            return
        # Otherwise we need to resample
        resample_index = misctools.get_resample_index(self.vehicle.weights)
        # self.states is a numpy array so the indexing operation forces a copy
        self.vehicle.weights = np.ones(nparticles, dtype=float)/nparticles
        self.vehicle.states = self.vehicle.states[resample_index]
        self.vehicle.covs = self.vehicle.covs[resample_index]
        self.vehicle.maps = [self.vehicle.maps[i].copy() 
            for i in resample_index]
        
