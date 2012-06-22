# -*- coding: utf-8 -*-

import blas_tools as blas
import numpy as np
import gmphdfilter
import misctools

class PFG500(object):
    def __init__(self, q_var, gps_var, dvl_bottom_var, dvl_water_var, nparticles=100):
        self.nparticles = nparticles
        ndims = 5
        self.ndims = ndims
        self.x = np.zeros((nparticles, ndims))
        self.P = np.zeros((ndims, ndims))
        self.weights = 1.0/nparticles*np.ones(nparticles)
        self.gps_h = self.gpsH()
        self.gps_r = self.gpsR(gps_var)
        self.dvl_h = self.dvlH()
        self.dvl_bottom_r = self.dvlR(dvl_bottom_var)
        self.dvl_water_r = self.dvlR(dvl_water_var)
        self.Q = self.computeQ(q_var)
        self.init = False
        self.UPDATED = False
        
    def trans_matrices(self, ctrl_input, delta_t):
        # u is assumed to be ordered as [roll, pitch, yaw]
        r, p, y = 0, 1, 2
        # Evaluate cosine and sine of roll, pitch, yaw
        c = np.cos(ctrl_input)
        s = np.sin(ctrl_input)
        
        # Specify the rotation matrix
        # See http://en.wikipedia.org/wiki/Rotation_matrix
        rot_mat = delta_t * np.array(
                [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
                 [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ]] )
                 #[-s[p], s[r]*c[p], c[r]*c[p] ]])
        # Transition matrix
        trans_mat = np.array([ np.vstack(( np.hstack((np.eye(2), rot_mat)),
                                   np.hstack((np.zeros((3,2)), np.eye(3))) )) ])[0]
        ## Add white Gaussian noise to the predicted states
        # Compute scaling for the noise
        scale_matrix = np.array([np.vstack((rot_mat*delta_t/2, delta_t*np.eye(3)))])
        process_noise = np.array([self.Q])
        # Compute the process noise as scale_matrix*process_noise*scale_matrix'
        sc_process_noise = blas.dgemm(scale_matrix, 
                       blas.dgemm(process_noise, scale_matrix, 
                                  TRANSPOSE_B=True, beta=0.0), beta=0.0)[0]
        return trans_mat, sc_process_noise
        
    
    def computeQ(self, q_var):
        """
        Process noise matrix
        """
        Q = np.eye(3)
        return Q*q_var
    
    
    def gpsH(self):
        """
        Observation (or measurement) matrix for gps observations
        """
        gps_h = np.zeros((2,self.ndims))
        gps_h[0,0] = 1.0
        gps_h[1,1] = 1.0
        return gps_h
    
    
    def dvlH(self):
        """
        Observation (or measurement) matrix for dvl observations
        """
        dvl_h = np.zeros((3,self.ndims))
        dvl_h[0,2] = 1.0
        dvl_h[1,3] = 1.0
        dvl_h[2,4] = 1.0
        return dvl_h
    
    
    def gpsR(self, gps_var):
        """
        Observation noise matrix for gps
        """
        gps_r = np.eye(2)
        return gps_r*gps_var
    
    
    def dvlR(self, dvl_var):
        """
        Observation noise matrix for dvl
        """
        dvl_r = np.eye(3)
        return dvl_r*dvl_var
        
        
    def initialize(self, x):
        self.x[:] = x
        self.init = True
#        print "EKF Initialized"
        
        
    def prediction(self, u, t):
        """
        Predict the current state and covariance matrix using control input
        """
        if self.UPDATED:
            # Resample
            self.resample()
            self.UPDATED = False
        trans_mat, sc_process_noise = self.trans_matrices(u, t)
        self._x_ = blas.dgemv(np.array([trans_mat]), self.x)
        awg_noise = np.random.multivariate_normal(np.zeros(self.ndims), sc_process_noise, self.nparticles)
        self._x_ += awg_noise
        self._P_ = np.dot(np.dot(trans_mat, self.P), trans_mat.T) + sc_process_noise
        x_mean, self._P_ = misctools.sample_mn_cv(self.x, self.weights)
#        print "EKF Prediction"
        
        
    def kf_gpsUpdate(self, z):
        ### gps update with z = [north, east] wrt world frame ###
        R = self.gps_r
        upd_state, upd_covariance, kalman_info = \
                gmphdfilter.blas_kf_update(np.array([self._x_]), np.array([self._P_]), 
                               np.array([self.gps_h]), np.array([R]), z, False)
        self.x = upd_state[0]
        self.P = upd_covariance[0]
#        print "EKF GPS update"
    
    def pf_gpsUpdate(self, z):
        R = self.gps_r
        obs_from_state = blas.dgemv(np.array([self.gps_h]), self._x_)
        x_pdf = misctools.mvnpdf(np.array([z]), obs_from_state, np.array([R]))
        self.weights *= x_pdf
        self.weights /= self.weights.sum()
        self.x = self._x_
        x_mean, self._P_ = misctools.sample_mn_cv(self.x, self.weights)
        self.P = self._P_
        self.UPDATED = True
    
    gpsUpdate = pf_gpsUpdate
    
    
    def kf_dvlUpdate(self, z, velocity_respect_to = 'bottom'):
        if velocity_respect_to == 'bottom':
            R = self.dvl_bottom_r
        else:
            R = self.dvl_water_r
        try:
            upd_state, upd_covariance, kalman_info = \
                gmphdfilter.blas_kf_update(np.array([self._x_]), np.array([self._P_]), 
                               np.array([self.dvl_h]), np.array([R]), z, False)
            self.x = upd_state[0]
            self.P = upd_covariance[0]
        except:
            pass
#        print "EKF DVL update"
    
    def pf_dvlUpdate(self, z, velocity_respect_to = 'bottom'):
        if velocity_respect_to == 'bottom':
            R = self.dvl_bottom_r
        else:
            R = self.dvl_water_r
        
        obs_from_state = blas.dgemv(np.array([self.dvl_h]), self._x_)
        x_pdf = misctools.mvnpdf(np.array([z]), obs_from_state, np.array([R]))
        self.weights *= x_pdf
        self.weights /= self.weights.sum()
        self.x = self._x_
        x_mean, self._P_ = misctools.sample_mn_cv(self.x, self.weights)
        self.P = self._P_
        self.UPDATED = True
        
    dvlUpdate = pf_dvlUpdate
    
    
    def updatePrediction(self):
        self.x = self._x_
        self.P = self._P_
        
        
    def getStateVector(self):
        x_mean, x_cov = misctools.sample_mn_cv(self.x, self.weights)
        self.P = x_cov
        return x_mean
    
    def get_covariance(self):
        return self.P
    
    def resample(self):
        resample_index = misctools.get_resample_index(self.weights)
        # self.states is a numpy array so the indexing operation forces a copy
        self.x = self.x[resample_index]
        self.weights = np.ones(self.nparticles, dtype=float)/self.nparticles
        
    