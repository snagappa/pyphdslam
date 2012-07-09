#!/usr/bin/env python

# ROS imports
import math

# Python imports
from numpy import *
import blas_tools as blas
import numpy as np
import code
from gmphdfilter import blas_kf_update
import girona500

class EKFG500 :
    def __init__(self, q_var, gps_var, dvl_bottom_var, dvl_water_var):
        self.x = zeros(5)
        self.P = zeros((5, 5))
        self.gps_h = self.gpsH()
        self.gps_r = self.gpsR(gps_var)
        self.dvl_h = self.dvlH()
        self.dvl_bottom_r = self.dvlR(dvl_bottom_var)
        self.dvl_water_r = self.dvlR(dvl_water_var)
        self.Q = self.computeQ(q_var)
        self.init = False
        
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
        Q = eye(3)
        return Q*q_var
    
    
    def gpsH(self):
        """
        Observation (or measurement) matrix for gps observations
        """
        gps_h = zeros((2,5))
        gps_h[0,0] = 1.0
        gps_h[1,1] = 1.0
        return gps_h
    
    
    def dvlH(self):
        """
        Observation (or measurement) matrix for dvl observations
        """
        dvl_h = zeros((3,5))
        dvl_h[0,2] = 1.0
        dvl_h[1,3] = 1.0
        dvl_h[2,4] = 1.0
        return dvl_h
    
    
    def gpsR(self, gps_var):
        """
        Observation noise matrix for gps
        """
        gps_r = eye(2)
        return gps_r*gps_var
    
    
    def dvlR(self, dvl_var):
        """
        Observation noise matrix for dvl
        """
        dvl_r = eye(3)
        return dvl_r*dvl_var
        
        
    def initialize(self, x):
        self.x = x
        self.init = True
#        print "EKF Initialized"
        
        
    def ekf_prediction(self, u, t):
        """
        Predict the current state and covariance matrix using control input
        """
        trans_mat, sc_process_noise = self.trans_matrices(u, t)
        self._x_ = dot(trans_mat, self.x)
        self._P_ = dot(dot(trans_mat, self.P), trans_mat.T) + sc_process_noise
#        print "EKF Prediction"
        
    def ukf_prediction(self, u, t):
        _alpha_ = 1e-3
        _beta_ = 2.0
        _kappa_ = 0
        
        trans_mat, sc_process_noise = self.trans_matrices(u, t)
        P = self.P + sc_process_noise
        x = self.x
        
        # UKF prediction
        # Create the Sigma points
        (x_sigma, x_weight, P_weight) = girona500.createSigmaPoints(x, P, _alpha_, _beta_, _kappa_)
        
        # Predict Sigma points and multiply by weight
        x_sigma_predicted = blas.dgemv(np.array([trans_mat]), x_sigma, beta=0.0)
        
        blas.dscal(x_weight, x_sigma_predicted)
        
        # Take the weighted mean of the Sigma points to get the predicted mean
        pred_state = np.add.reduce(x_sigma_predicted)
        
        # Generate the weighted Sigma covariance and add Q to get predicted cov
        pred_cov = girona500.evalSigmaCovariance(P_weight, x_sigma_predicted, pred_state) #+ sc_process_noise
        
        self._x_ = pred_state
        self._P_ = pred_cov
    
    prediction = ekf_prediction
    #def prediction(self, u, t):
    #    try:
    #        self.ukf_prediction(u, t)
    #    except:
    #        print "ukf failed, falling back to ekf"
    #        self.ekf_prediction(u, t)
    
    def gpsUpdate(self, z):
        ### gps update with z = [north, east] wrt world frame ###
        R = self.gps_r
        upd_state, upd_covariance, kalman_info = \
                blas_kf_update(np.array([self._x_]), np.array([self._P_]), 
                               np.array([self.gps_h]), np.array([R]), z, False)
        self.x = upd_state[0]
        self.P = upd_covariance[0]
        
        #innovation = z - dot(self.gps_h, self._x_).T
        #temp_K = dot(dot(self.gps_h, self._P_), self.gps_h.T) + self.gps_r
        #temp_K_I = squeeze(asarray(matrix(temp_K).I))
        #K = dot(dot(self._P_, self.gps_h.T), temp_K_I)
        #Ki = dot(K, innovation)
        #self.x = self._x_ + Ki
        #self.P = dot((eye(5)-dot(K, self.gps_h)), self._P_)
#        print "EKF GPS update"
        

    def dvlUpdate(self, z, velocity_respect_to = 'bottom'):
        if velocity_respect_to == 'bottom':
            R = self.dvl_bottom_r
        else:
            R = self.dvl_water_r
        try:
            upd_state, upd_covariance, kalman_info = \
                blas_kf_update(np.array([self._x_]), np.array([self._P_]), 
                               np.array([self.dvl_h]), np.array([R]), z, False)
            self.x = upd_state[0]
            self.P = upd_covariance[0]
        except:
            pass
#        print "EKF DVL update"
    
    
    def updatePrediction(self):
        self.x = self._x_
        self.P = self._P_
        
        
    def getStateVector(self):
        return self.x
            
if __name__ == '__main__':
    ekf_g500 = EKFG500()
    
