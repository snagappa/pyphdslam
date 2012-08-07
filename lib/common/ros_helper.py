# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:44:09 2012

@author: snagappa
"""

import rospy
import numpy as np

class STRUCT(object):
    pass

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
    
    
def get_config(ros_param_list):
    config = STRUCT()
    for param in ros_param_list:
        add_ros_param(config, param)
    return config
    
def get_g500_config():
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
    return get_config(ros_param_list)
# End ROS related functions    
##############################################################################
