# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:14:28 2012

@author: snagappa
"""


from sensor_msgs.msg import PointCloud2, PointField
from lib.common import pc2wrapper
import numpy as np

FLOAT_SIZE = 4
field_offsets = lambda(x): range(0, (x)*FLOAT_SIZE, FLOAT_SIZE)

MSG_XYZ = "xyz"
XYZ_FIELDS = ['x', 'y', 'z']
MSG_XYZ_COV = "xyz_cov"
XYZ_COV_FIELDS = XYZ_FIELDS + ['sigma_x', 'sigma_y', 'sigma_z']

class msgs(object):
    def __init__(self, msg_type):
        self.msg_type = msg_type
        if msg_type == MSG_XYZ:
            self._fields_list_ = XYZ_FIELDS
        elif msg_type == MSG_XYZ_COV:
            self._fields_list_ = XYZ_COV_FIELDS
        else:
            self._fields_list_ = []
        self._num_fields_ = len(self._fields_list_)
        self.pcl_fields = [PointField(_field_name_, _field_offset_, PointField.FLOAT32, 1) for 
                        (_field_name_, _field_offset_) in zip(self._fields_list_, field_offsets(self._num_fields_))]
        self.pcl_header = PointCloud2().header
        self.__FORCE_COMPAT__ = False
    
    def force_compat(self, bool_val):
        self.__FORCE_COMPAT__ = bool_val
        
    
    def to_pcl(self, header_stamp, point_array):
        self.pcl_header.stamp = header_stamp
        return pc2wrapper.create_cloud(self.pcl_header, self.pcl_fields, point_array)
    
    
    def from_pcl(self, pcl_msg):
        if self._fields_list_ is None:
            pcl_points = np.array(list(pc2wrapper.read_points(pcl_msg)))
        else:
            pcl_fields_list = [pcl_msg.fields[i].name for i in range(len(pcl_msg.fields))]
            if not self.__FORCE_COMPAT__:
                if not pcl_fields_list == self._fields_list_:
                    print "pcl fields don't match expected values - ignoring message"
                    pcl_points = np.empty(0)
                else:
                    pcl_points = np.array(list(pc2wrapper.read_points(pcl_msg)))
            else:
                if not (pcl_fields_list in [XYZ_FIELDS, XYZ_COV_FIELDS]):
                    print "could not force combatibility"
                    pcl_points = np.empty(0)
                else:
                    _pcl_points_ = np.array(list(pc2wrapper.read_points(pcl_msg)))
                    pcl_points = np.zeros((_pcl_points_.shape[0], len(self.msg_type)))
                    dims = min((_pcl_points_.shape[1], len(self.msg_type)))
                    pcl_points[:, 0:dims] = _pcl_points_[:, 0:dims]
        return(pcl_points)
        