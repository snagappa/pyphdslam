# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:22:51 2012

@author: snagappa
"""
import numpy as np
import code
import tf

class camera_fov(object):
    def __init__(self,  fov_x_deg=60, fov_y_deg=45, fov_far_m=3):
        self.fov_x_deg = fov_x_deg
        self.fov_y_deg = fov_y_deg
        self.fov_far_m = fov_far_m
        self.tmp = lambda: 0
        self.precalc()
    
    def set_x_y_far(self, x_deg=None, y_deg=None, far_m=None):
        if not x_deg == None:
            self.fov_x_deg = x_deg
        if not y_deg == None:
            self.fov_y_deg = y_deg
        if not far_m == None:
            self.fov_far_m = far_m
        self.precalc()
        
    def precalc(self):
        self.tmp.fov_x_rad = self.fov_x_deg * np.pi/180.0
        self.tmp.fov_y_rad = self.fov_y_deg * np.pi/180.0
        # Take cosine of half the angle
        self.tmp.tan_x = np.tan(self.tmp.fov_x_rad/2)
        self.tmp.tan_y = np.tan(self.tmp.fov_y_rad/2)
        self.tmp._test_dists_ = np.arange(0.05, self.fov_far_m, 0.05)
        self.tmp._test_dists_area_ = 4*self.get_rect__half_width_height(self.tmp._test_dists_).prod(axis=1)
        self.tmp._proportional_area_ = self.tmp._test_dists_area_/self.tmp._test_dists_area_.sum()
        self.tmp._proportional_vol_ = self.tmp._test_dists_area_*0.05/(0.33*self.tmp._test_dists_area_[-1]*self.fov_far_m)
    
    def is_visible(self, point_xyz):
        if not point_xyz.shape[0]:
            return np.array([], dtype=np.bool)
        test_distances = point_xyz[:, 0].copy()
        xy_limits = self.get_rect__half_width_height(test_distances)
        is_inside_rect = self.__inside_rect__(xy_limits, point_xyz[:,1:3])
        return (is_inside_rect*((0 < test_distances)*(test_distances<self.fov_far_m)))
        
    def __inside_rect__(self, half_rect__width_height, xy):
        bool_is_inside = ((-half_rect__width_height<xy)*(xy<half_rect__width_height)).all(axis=1)
        return bool_is_inside
        
    def fov_vertices_2d(self):
        x_delta = self.fov_far_m*self.tmp.tan_x
        return np.array([[0, 0], [-x_delta, self.fov_far_m], [x_delta, self.fov_far_m]])
        
    def get_rect__half_width_height(self, far):
        return np.array([far*self.tmp.tan_x, far*self.tmp.tan_y]).T
        
    def z_prob(self, far):
        #z_idx = abs(self.tmp._test_dists_[:, np.newaxis] - far).argmin(axis=0)
        #return self.tmp._proportional_vol_[z_idx].copy()
        
        return (1/(self.tmp._test_dists_area_[-1]*self.fov_far_m/3))*np.ones(far.shape[0])
        
    def pdf_detection(self, ref_ned, ref_rpy, features_abs):
        if not features_abs.shape[0]:
            return np.empty(0)
        # Transform points to local frame
        rel_landmarks = tf.relative(ref_ned, ref_rpy, features_abs)
        visible_features_idx = np.array(self.is_visible(rel_landmarks), 
                                        dtype=np.float)
        # Take the distance rather than the square?
        dist_from_ref = np.sum(rel_landmarks[:,[0, 1]]**2, axis=1)
        return np.exp(-dist_from_ref*0.005)*0.99*visible_features_idx
        
    def pdf_clutter(self, z_rel):
        self.z_prob(z_rel[:,0])
        
    def rel_to_abs(self, ref_ned, ref_rpy, features_rel):
        return tf.absolute(ref_ned, ref_rpy, features_rel)
        