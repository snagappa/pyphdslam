# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:27:07 2012

@author: snagappa
"""

import numpy as np
from lib.common import blas

def rotation_matrix(RPY):
    ## See http://en.wikipedia.org/wiki/Rotation_matrix
    r, p, y = 0, 1, 2
    c = np.cos(RPY)
    s = np.sin(RPY)
    rot_matrix = np.array(
                [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
                 [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
                 [-s[p], s[r]*c[p], c[r]*c[p] ]])
    return rot_matrix
    
def relative(vehicle_NED, vehicle_RPY, features_NED):
    if not features_NED.shape[0]: return np.empty(0)
    relative_position = features_NED - vehicle_NED
    #r, p, y = 0, 1, 2
    #c = np.cos(-vehicle_RPY)
    #s = np.sin(-vehicle_RPY)
    #rotation_matrix = np.array([
    #            [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
    #             [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
    #             [-s[p], s[r]*c[p], c[r]*c[p] ]]])
    rot_matrix = np.array([rotation_matrix(-vehicle_RPY)])
    relative_position = blas.dgemv(rot_matrix, relative_position)
    #np.dot(rotation_matrix, relative_position.T).T
    return relative_position

def absolute(vehicle_NED, vehicle_RPY, features_NED):
    if not features_NED.shape[0]: return np.empty(0)
    #r, p, y = 0, 1, 2
    #c = np.cos(vehicle_RPY)
    #s = np.sin(vehicle_RPY)
    #rotation_matrix = np.array([
    #            [[c[p]*c[y], -c[r]*s[y]+s[r]*s[p]*c[y], s[r]*s[y]+c[r]*s[p]*c[y] ],
    #             [c[p]*s[y], c[r]*c[y]+s[r]*s[p]*s[y], -s[r]*c[y]+c[r]*s[p]*s[y] ],
    #             [-s[p], s[r]*c[p], c[r]*c[p] ]]])
    rot_matrix = np.array([rotation_matrix(vehicle_RPY)])
    absolute_position = blas.dgemv(rot_matrix, features_NED) + vehicle_NED
    return absolute_position