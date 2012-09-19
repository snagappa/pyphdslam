#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       ros_slam.py
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

# ROS imports
import roslib 
roslib.load_manifest('udg_pandora')
import rospy
import tf
import PyKDL
import math
#import code
import copy
import threading
import sys

# Msgs imports
from cola2_navigation.msg import TeledyneExplorerDvl, ValeportSoundVelocity, \
    FastraxIt500Gps
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from auv_msgs.msg import NavSts
from std_srvs.srv import Empty, EmptyResponse
from cola2_navigation.srv import SetNE, SetNEResponse #, SetNERequest

# import pyximport; pyximport.install()
import lib.slam_worker
#from lib.slam_worker import PHDSLAM
PHDSLAM = lib.slam_worker.PHDSLAM
import numpy as np
from lib.common.ros_helper import get_g500_config
from lib.common.kalmanfilter import sigma_pts
import featuredetector

INVALID_ALTITUDE = -32665
SAVITZKY_GOLAY_COEFFS = [0.2,  0.1,  0. , -0.1, -0.2]

UKF_ALPHA = 0.01
UKF_BETA = 2
UKF_KAPPA = 0

# Profile options
__PROFILE__ = False
__PROFILE_NUM_LOOPS__ = 100
#code.interact(local=locals())

def normalize_angle(np_array):
    return (np_array + (2.0*np.pi*np.floor((np.pi - np_array)/(2.0*np.pi))))

class STRUCT(object):
    pass
    
class G500_SLAM():
    def __init__(self, name):
        # Get config
        self.config = get_g500_config()
        
        # Main SLAM worker
        self.slam_worker = self.init_slam()
        
        # Structure to store vehicle pose from sensors
        self.vehicle = STRUCT()
        # roll, pitch and yaw rates - equivalent to odom.twist.twist.angular
        self.vehicle.twist_angular = np.zeros(3, dtype=float)
        self.vehicle.twist_linear = np.zeros(3, dtype=float)
        # position x, y, z - the state space estimated by the filter.
        self.vehicle.pose_position = np.zeros(3, dtype=float)
        # orientation quaternion x,y,z,w - same as odom.pose.pose.orientation
        self.vehicle.pose_orientation = np.zeros(4, dtype=float)
        # Altitude from dvl
        self.vehicle.altitude = 0.0
        
        # Initialise ROS stuff
        self.init_config()
        self.ros = self.init_ros(name)
        
        self.__LOCK__ = threading.Lock()
        # Set as true to run without gps/imu initialisation
        self.config.init.init = False
        
        
    def init_slam(self):
        slam_worker = PHDSLAM()
        slam_worker.set_parameters(
            Q=np.eye(3)*self.config.model_covariance, 
            gpsH=np.hstack(( np.eye(2), np.zeros((2,4)) )), 
            gpsR=np.eye(2)*self.config.gps_covariance, 
            dvlH=np.hstack(( np.zeros((3,3)), np.eye(3) )), 
            dvl_b_R=np.eye(3)*self.config.dvl_bottom_covariance, 
            dvl_w_R=np.eye(3)*self.config.dvl_water_covariance)
        
        #ndims = slam_worker.vars.ndims
        nparticles = slam_worker.vars.nparticles
        if nparticles == 7:#2*ndims+1:
            sigma_states = self._make_sigma_states_(slam_worker, 
                                                    np.zeros((1, 3)), nparticles)
            slam_worker.set_states(states=sigma_states)
        return slam_worker
        
    def _make_sigma_states_(self, slam_worker, mean_state, nparticles):
        sc_process_noise = \
            slam_worker.trans_matrices(np.zeros(3), 1.0)[1] + \
            slam_worker.trans_matrices(np.zeros(3), 0.01)[1]
        sigma_states = sigma_pts(mean_state, 
                                 sc_process_noise[0:3,0:3].copy(), 
                                _alpha=UKF_ALPHA, _beta=UKF_BETA, 
                                _kappa=UKF_KAPPA)[0]
        sigma_states = np.array(
            np.hstack((sigma_states, np.zeros((nparticles,3)))),
            order='C')
        return sigma_states
            
    
    def init_config(self):
        config = self.config
        
        #Create static transformations
        config.dvl_tf = self.compute_tf(config.dvl_tf_data)
        config.imu_tf = self.compute_tf(config.imu_tf_data)        
        config.svs_tf = self.compute_tf(config.svs_tf_data)
        
        #Initialize flags
        config.init = STRUCT()
        config.init.init = False
        config.init.north = 0.0
        config.init.east = 0.0
        config.init.dvl = False
        config.init.imu = False
        config.init.svs = False
        
        #init last sensor update
        time_now = rospy.Time.now()
        config.last_time = STRUCT()
        config.last_time.init = copy.copy(time_now)
        config.last_time.predict = copy.copy(time_now)
        config.last_time.gps = copy.copy(time_now)
        config.last_time.dvl = copy.copy(time_now)
        config.last_time.imu = copy.copy(time_now)
        config.last_time.svs = copy.copy(time_now)
        config.last_time.map = copy.copy(time_now)
        
        config.imu_data = False
        config.gps_data = not config.gps_update
        config.altitude = INVALID_ALTITUDE
        config.bottom_status = 0
        config.gps_init_samples_list = []
        # Buffer for smoothing the yaw rate
        config.heading_buffer = []
        config.savitzky_golay_coeffs = SAVITZKY_GOLAY_COEFFS
        
    def init_ros(self, name):
        ros = STRUCT()
        ros.name = name
        ros.last_update_time = rospy.Time.now()
        ros.NO_LOCK_ACQUIRE = 0
        
        ros.pcl_helper = \
            featuredetector.msgs.msgs(featuredetector.msgs.MSG_XYZ_COV)
        
        if not __PROFILE__:
            print "Creating ROS subscriptions..."
            # Create Subscriber
            rospy.Subscriber("/cola2_navigation/teledyne_explorer_dvl", 
                TeledyneExplorerDvl, self.update_dvl)
            rospy.Subscriber("/cola2_navigation/valeport_sound_velocity", 
                             ValeportSoundVelocity, self.update_svs)
            rospy.Subscriber("/cola2_navigation/imu", Imu, self.update_imu)
            if self.config.gps_update :
                rospy.Subscriber("/cola2_navigation/fastrax_it_500_gps", 
                                 FastraxIt500Gps, self.update_gps)
            ## Subscribe to visiona slam-features node
            rospy.Subscriber("/slamsim/features", PointCloud2, 
                             self.update_features)
            # Subscribe to sonar slam features node for
            #rospy.Subscriber("/slam_features/fls_pcl", PointCloud2, 
            #                 self.update_features)
            #Create services
            ros.reset_navigation = \
                rospy.Service('/slam_g500/reset_navigation', 
                              Empty, self.reset_navigation)
            ros.reset_navigation = rospy.Service('/slam_g500/set_navigation', 
                                                 SetNE, self.set_navigation)
            
            # Create publishers
            ros.nav = STRUCT()
            ros.nav.msg = NavSts()
            ros.nav.publisher = rospy.Publisher("/phdslam/nav_sts", NavSts)
            # Publish landmarks
            ros.map = STRUCT()
            ros.map.msg = PointCloud2()
            ros.map.publisher = \
                rospy.Publisher("/phdslam/features", PointCloud2)
            ros.map.helper = \
                featuredetector.msgs.msgs(featuredetector.msgs.MSG_XYZ_COV)
            
            # Publish data every 500 ms
            rospy.timer.Timer(rospy.Duration(0.2), self.publish_data)
            # Callback to print vehicle state and weight
            #rospy.timer.Timer(rospy.Duration(10), self.debug_print)
        else:
            print "** RUNNING IN PROFILER MODE **"
        
        return ros
        
    def reset_navigation(self, req):
        print "Resetting navigation..."
        rospy.loginfo("%s: Reset Navigation", self.ros.name)
        #self.slam_worker.states[:,0:2] = 0
        #ndims = self.slam_worker.vars.ndims
        nparticles = self.slam_worker.vars.nparticles
        if nparticles == 7:#2*ndims + 1:
            #pose_angle = tf.transformations.euler_from_quaternion(
            #                                    self.vehicle.pose_orientation)
            sigma_states = self._make_sigma_states_(self.slam_worker, 
                                                    np.zeros((1, 3)), nparticles)
            self.slam_worker.set_states(states=sigma_states)
        else:
            self.slam_worker.vehicle.states[:, :] = 0
        return EmptyResponse()
        
    def set_navigation(self, req):
        print "Setting new navigation..."
        #ndims = self.slam_worker.vars.ndims
        nparticles = self.slam_worker.vars.nparticles
        if nparticles == 7:#2*ndims + 1:
            #pose_angle = tf.transformations.euler_from_quaternion(
            #                                    self.vehicle.pose_orientation)
            mean_state = np.array([[req.north, req.east, 0]])
            sigma_states = self._make_sigma_states_(self.slam_worker, 
                                                    mean_state, nparticles)
            self.slam_worker.set_states(states=sigma_states)
        else:
            self.slam_worker.vehicle.states[:, 0] = req.north
            self.slam_worker.vehicle.states[:, 1] = req.east
        self.slam_worker.vehicle.states[:, 2] = self.vehicle.pose_position[2]
        ret = SetNEResponse()
        ret.success = True
        return ret
    
    def compute_tf(self, transform):
        r = PyKDL.Rotation.RPY(math.radians(transform[3]), 
                               math.radians(transform[4]), 
                               math.radians(transform[5]))
        #rospy.loginfo("Rotation: %s", str(r))
        v = PyKDL.Vector(transform[0], transform[1], transform[2])
        #rospy.loginfo("Vector: %s", str(v))
        frame = PyKDL.Frame(r, v)
        #rospy.loginfo("Frame: %s", str(frame))
        return frame
    
    
    def update_gps(self, gps):
        if (gps.data_quality >= 1) and (gps.latitude_hemisphere >= 0) and \
        (gps.longitude_hemisphere >= 0):
            config = self.config
            config.last_time.gps = copy.copy(gps.header.stamp)
            if not config.gps_data :
                print "gps not set: initialising"
                config.gps_init_samples_list.append([gps.north, gps.east])
                if len(config.gps_init_samples_list) >= config.gps_init_samples:
                    config.gps_data = True
                    [config.init.north, config.init.east] = \
                            np.median(np.array(config.gps_init_samples_list), 
                                      axis=0)
            else:
                slam_estimate = self.slam_worker.estimate()
                est_state = slam_estimate.vehicle.ned.state[0:2]
                distance = np.sqrt((est_state[0] - gps.north)**2 + 
                                (est_state[1] - gps.east)**2)
                #rospy.loginfo("%s, Distance: %s", self.name, distance)
                
                # Right now the GPS is only used to initialize the navigation 
                # not for updating it!!!
                if distance < 0.1:
                    z = np.array([gps.north, gps.east])
                    self.__LOCK__.acquire()
                    try:
                        if self.predict(config.last_time.gps):
                            self.slam_worker.update_gps(z)
                            self.ros.last_update_time = config.last_time.gps
                            self.slam_worker.resample()
                    finally:
                        self.__LOCK__.release()
                    #self.publish_data()
                
        
    def update_dvl(self, dvl):
        #print os.getpid()
        config = self.config
        config.last_time.dvl = copy.copy(dvl.header.stamp)
        config.init.dvl = True
        
        # If dvl_update == 0 --> No update
        # If dvl_update == 1 --> Update wrt bottom
        # If dvl_update == 2 --> Update wrt water
        dvl_update = 0
        
        if dvl.bi_status == "A" and dvl.bi_error > -32.0:
            if (abs(dvl.bi_x_axis) < config.dvl_max_v and 
                abs(dvl.bi_y_axis) < config.dvl_max_v and 
                abs(dvl.bi_z_axis) < config.dvl_max_v) : 
                v = PyKDL.Vector(dvl.bi_x_axis, dvl.bi_y_axis, dvl.bi_z_axis)
                dvl_update = 1
        elif dvl.wi_status == "A" and dvl.wi_error > -32.0:
            if (abs(dvl.wi_x_axis) < config.dvl_max_v and 
                abs(dvl.wi_y_axis) < config.dvl_max_v and 
                abs(dvl.wi_z_axis) < config.dvl_max_v) : 
                v = PyKDL.Vector(dvl.wi_x_axis, dvl.wi_y_axis, dvl.wi_z_axis)
                dvl_update = 2
        
        #Filter to check if the altitude is reliable
        if dvl.bi_status == "A" and dvl.bi_error > -32.0:
            config.bottom_status =  config.bottom_status + 1
        else:
            config.bottom_status = 0
        
        if config.bottom_status > 4:
            self.vehicle.altitude = dvl.bd_range
        else:
            self.vehicle.altitude = INVALID_ALTITUDE
            
        if dvl_update != 0:
            #Rotate DVL velocities and Publish
            #Compte! EL DVL no es dextrogir i s'ha de negar la Y
            vr = config.dvl_tf.M * v
            distance = config.dvl_tf_data[0:3]
            #dvl_reference = "bottom" if dvl_update == 1 else "water"
            mode = 'b' if (dvl_update == 1) else 'w'
            #if self.__LOCK__.locked():
            #    self.ros.NO_LOCK_ACQUIRE += 1
            #    return
            if dvl.header.stamp < config.last_time.predict:
                self.ros.NO_LOCK_ACQUIRE += 1
                return
            self.__LOCK__.acquire()
            try:
                if dvl.header.stamp < config.last_time.predict:
                    self.ros.NO_LOCK_ACQUIRE += 1
                    return
                self.vehicle.twist_linear = np.array([vr[0], -vr[1], vr[2]])
                # Ara ja tenim la velocitat lineal en el DVL representada en 
                # eixos de vehicle falta calcular la velocitat lineal al 
                # centre del vehicle en eixos del vehicle
                angular_velocity = self.vehicle.twist_angular
                self.vehicle.twist_linear -= np.cross(angular_velocity, 
                                                      distance)
                self.predict(config.last_time.dvl)
                
                self.slam_worker.update_dvl(self.vehicle.twist_linear, mode)
                self.ros.last_update_time = config.last_time.dvl
                self.slam_worker.resample()
            finally:
                self.__LOCK__.release()
            #self.publish_data()
        else:
            rospy.loginfo('%s, invalid DVL velocity measurement!', 
                          self.ros.name)
        
    
    def update_svs(self, svs):
        #print os.getpid()
        config = self.config
        config.last_time.svs = copy.copy(svs.header.stamp)
        svs_data = PyKDL.Vector(.0, .0, svs.pressure)
        pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
        vehicle_rpy = PyKDL.Rotation.RPY(*pose_angle)
        svs_trans = config.svs_tf.p
        svs_trans = vehicle_rpy * svs_trans
        svs_data = svs_data - svs_trans
        
        if not config.init.svs:
            config.init.svs = True
            self.__LOCK__.acquire()
            try:
                print "INITIALISING DEPTH to ", str(svs_data[2])
                self.vehicle.pose_position[2] = svs_data[2]
                self.slam_worker.vehicle.states[:, 2] = svs_data[2]
            finally:
                self.__LOCK__.release()
            return
        
        if self.__LOCK__.locked():
            return
        self.__LOCK__.acquire()
        try:
            self.vehicle.pose_position[2] = svs_data[2]
            if self.predict(config.last_time.svs):
                #self.slam_worker.update_svs(self.vehicle.pose_position[2])
                self.slam_worker.vehicle.states[:, 2] = self.vehicle.pose_position[2]
                self.ros.last_update_time = config.last_time.svs
        finally:
            self.__LOCK__.release()
        #self.publish_data()

    
    def update_imu(self, imu):
        #print os.getpid()
        ret_val = None
        config = self.config
        config.init.imu = True
        
        pose_angle = tf.transformations.euler_from_quaternion(
                                       [imu.orientation.x, imu.orientation.y, 
                                       imu.orientation.z, imu.orientation.w])
        imu_data =  PyKDL.Rotation.RPY(*pose_angle)
        imu_data = imu_data*config.imu_tf.M
        pose_angle = imu_data.GetRPY()
        if not config.imu_data :
            config.last_imu_orientation = pose_angle
            config.last_time.imu = copy.copy(imu.header.stamp)
            #config.imu_data = True
            # Initialize heading buffer to apply a savitzky_golay derivation
            if len(config.heading_buffer) == 0:
                config.heading_buffer.append(pose_angle[2])
                
            inc = normalize_angle(pose_angle[2] - config.heading_buffer[-1])
            config.heading_buffer.append(config.heading_buffer[-1] + inc)
            
            if len(config.heading_buffer) == len(config.savitzky_golay_coeffs):
                config.imu_data = True
            
        else:
            period = (imu.header.stamp - config.last_time.imu).to_sec()
            pose_angle_quaternion = \
                tf.transformations.quaternion_from_euler(*pose_angle)
            config.last_imu_orientation = pose_angle
            self.__LOCK__.acquire()
            try:
                self.vehicle.pose_orientation = pose_angle_quaternion
                
                # Derive angular velocities from orientations
                self.vehicle.twist_angular = \
                    normalize_angle(np.array(pose_angle)- 
                        np.array(config.last_imu_orientation))/period
                
                # For yaw rate we apply a savitzky_golay derivation
                inc = normalize_angle(pose_angle[2] - 
                                      config.heading_buffer[-1])
                config.heading_buffer.append(config.heading_buffer[-1] + inc)
                config.heading_buffer.pop(0)
                self.vehicle.twist_angular[2] = \
                    np.convolve(config.heading_buffer, 
                                config.savitzky_golay_coeffs, 
                                mode='valid') / period
                config.last_time.imu = copy.copy(imu.header.stamp)
                
                self.predict(imu.header.stamp)
                self.ros.last_update_time = imu.header.stamp
                ###############################################################
            finally:
                self.__LOCK__.release()
            #self.publish_data()
        return ret_val
        
        
    def update_features(self, pcl_msg):
        init = self.config.init
        if (not init.init) or (not init.dvl):
            return
        self.__LOCK__.acquire()
        try:
            self.config.last_time.map = copy.copy(pcl_msg.header.stamp)
            self.predict(self.config.last_time.map)
            # Convert the pointcloud slam features into normal x,y,z
            # The pointcloud may have uncertainty on the points - this will be
            # the observation noise
            #slam_features = pointclouds.pointcloud2_to_xyz_array(pcl_msg)
            slam_features = self.ros.pcl_helper.from_pcl(pcl_msg)
            # We can now access the points as slam_features[i]
            self.slam_worker.update_features(slam_features)
            self.ros.last_update_time = pcl_msg.header.stamp
        finally:
            self.__LOCK__.release()
        
    
    def predict(self, predict_to_time):
        config = self.config
        if not config.init.init:
            time_now = predict_to_time
            config.last_time.predict = copy.copy(time_now)
            self.slam_worker.last_time.predict = time_now.to_sec()
            if config.imu_data and config.gps_data:                
                # Initialise slam worker with north and east co-ordinates
                init = lambda:0
                init.north = config.init.north
                init.east = config.init.east
                self.slam_worker.reset_states()
                print "Resetting states to ", \
                    str(init.north), ", ", str(init.east)
                self.set_navigation(init)
                self.slam_worker.vehicle.states[:, 2] = self.vehicle.pose_position[2]
                config.init.init = True
            return False
        else:
            pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
            if predict_to_time <= config.last_time.predict:
                self.ros.NO_LOCK_ACQUIRE += 1
                return False
            time_now = predict_to_time
            config.last_time.predict = copy.copy(time_now)
            time_now = time_now.to_sec()
            self.slam_worker.predict(np.array(pose_angle), time_now)
            return True
            
    
    def publish_data(self, *args, **kwargs):
        if not self.config.init.init:
            # self.ros.last_update_time = rospy.Time.now()
            # self.config.init = True
            return
        nav_msg = self.ros.nav.msg
        slam_estimate = self.slam_worker.estimate()
        est_state = slam_estimate.vehicle.ned.state
        est_cov = slam_estimate.vehicle.ned.covariance
        est_state_vel = slam_estimate.vehicle.vel_xyz.state
        angle = tf.transformations.euler_from_quaternion(
            self.vehicle.pose_orientation)
        
        # Create header
        nav_msg.header.stamp = self.ros.last_update_time
        nav_msg.header.frame_id = self.ros.name
        child_frame_id = "world"
                   
        #Fill Nav status topic
        nav_msg.position.north = est_state[0]
        nav_msg.position.east = est_state[1]
        nav_msg.position.depth = est_state[2]
        nav_msg.body_velocity.x = est_state_vel[0]
        nav_msg.body_velocity.y = est_state_vel[1]
        nav_msg.body_velocity.z = est_state_vel[2]
        nav_msg.orientation.roll = angle[0]
        nav_msg.orientation.pitch = angle[1]
        nav_msg.orientation.yaw = angle[2]
        nav_msg.orientation_rate.roll = self.vehicle.twist_angular[0]
        nav_msg.orientation_rate.pitch = self.vehicle.twist_angular[1]
        nav_msg.orientation_rate.yaw = self.vehicle.twist_angular[2]
        nav_msg.altitude = self.vehicle.altitude
        
        # Variance
        nav_msg.position_variance.north = est_cov[0, 0]
        nav_msg.position_variance.east = est_cov[1, 1]
        nav_msg.position_variance.depth = est_cov[2, 2]
        
        # nav_msg.status = np.uint8(np.log10(self.ros.NO_LOCK_ACQUIRE+1))
        
        #Publish topics
        self.ros.nav.publisher.publish(nav_msg)
        
        #Publish TF
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (nav_msg.position.north, 
                nav_msg.position.east, 
                nav_msg.position.depth),
            tf.transformations.quaternion_from_euler(
                nav_msg.orientation.roll, 
                nav_msg.orientation.pitch, 
                nav_msg.orientation.yaw),
            nav_msg.header.stamp, 
            nav_msg.header.frame_id, 
            child_frame_id)
            
        ##
        # Publish landmarks now
        map_estimate = slam_estimate.map
        map_states = map_estimate.state
        map_covs = map_estimate.covariance
        if map_states.shape[0]:
            diag_cov = np.array([np.diag(map_covs[i]) 
                for i in range(map_covs.shape[0])])
            pcl_msg = self.ros.map.helper.to_pcl(rospy.Time.now(), 
                np.hstack((map_states, diag_cov)))
        else:
            pcl_msg = self.ros.map.helper.to_pcl(rospy.Time.now(), 
                                                 np.zeros(0))
        pcl_msg.header.frame_id = self.ros.name
        # and publish visible landmarks
        self.ros.map.publisher.publish(pcl_msg)
        #print "Landmarks at: "
        #print map_states
        """
        print "Tracking ", map_estimate.weight.shape[0], \
            " (", map_estimate.weight.sum(), ") targets."
        #print "Intensity = ", map_estimate.weight.sum()
        dropped_msg_time = \
            (rospy.Time.now()-self.config.last_time.init).to_sec()
        print "Dropped ", self.ros.NO_LOCK_ACQUIRE, " messages in ", \
            int(dropped_msg_time), " seconds."
        """
    def debug_print(self, *args, **kwargs):
        print "Weights: "
        #print self.slam_worker.states
        print self.slam_worker.vehicle.weights
    

def main():
    try:
        # Init node
        rospy.init_node('phdslam')
        girona500_navigator = G500_SLAM(rospy.get_name())
        if not __PROFILE__:
            rospy.spin()
        else:
            import time
            try:
                imu_msg = rospy.wait_for_message("/navigation_g500/imu", 
                                                 Imu, 1)
                last_time = imu_msg.header.stamp
                TEST_IMU = True
            except rospy.ROSException:
                print "*** timeout waiting for imu message! ***"
                imu_msg = None
                TEST_IMU = False
            
            try:
                gps_msg = rospy.wait_for_message(
                        "/navigation_g500/fastrax_it_500_gps", 
                        FastraxIt500Gps, 1)
                last_time = gps_msg.header.stamp
                TEST_GPS = True
            except rospy.ROSException:
                print "*** timeout waiting for gps message! ***"
                gps_msg = None
                TEST_GPS = False
            
            try:
                svs_msg = rospy.wait_for_message(
                    "/navigation_g500/valeport_sound_velocity", 
                    ValeportSoundVelocity)
                last_time = svs_msg.header.stamp
                TEST_SVS = True
            except rospy.ROSException:
                print "*** timeout waiting for svs message! ***"
                svs_msg = None
                TEST_SVS = False
                
            try:
                dvl_msg = rospy.wait_for_message(
                    "/navigation_g500/teledyne_explorer_dvl",
                    TeledyneExplorerDvl)
                last_time = dvl_msg.header.stamp
                TEST_DVL = True
            except rospy.ROSException:
                print "*** timeout waiting for dvl message! ***"
                dvl_msg = None
                TEST_DVL = False
                
            try:
                pcl_msg = rospy.wait_for_message(
                    "/slamsim/features", PointCloud2, 2)
                last_time = pcl_msg.header.stamp
                TEST_PCL = True
            except rospy.ROSException:
                print "*** timeout waiting for pcl message! ***"
                pcl_msg = None
                TEST_PCL = False
            
            print "Pausing for 3 seconds..."
            time.sleep(3)
            test_msg_list = (imu_msg, gps_msg, svs_msg, dvl_msg, pcl_msg)
            test_flag_list = (TEST_IMU, TEST_GPS, TEST_SVS, TEST_DVL, TEST_PCL)
            test_str_list = ("imu", "gps", "svs", "dvl", "pcl")
            test_fn_list = (girona500_navigator.update_imu,
                            girona500_navigator.update_gps, 
                            girona500_navigator.update_svs,
                            girona500_navigator.update_dvl,
                            girona500_navigator.update_features)
            for test_msg, test_flag, test_str, test_fn in \
                    zip(test_msg_list, test_flag_list, 
                        test_str_list, test_fn_list):
                if test_flag:
                    print "\n\nTesting ", test_str
                    for count in range(__PROFILE_NUM_LOOPS__):
                        last_time.secs += 1
                        test_msg.header.stamp = last_time
                        test_fn(test_msg)
                        percent = int(round((count*100.0)/__PROFILE_NUM_LOOPS__))
                        sys.stdout.write("\r%d%% complete" %percent)    # or print >> sys.stdout, "\r%d%%" %i,
                        sys.stdout.flush()
            
            print "\n** Finished profiling **\n"
            rospy.signal_shutdown("Finished profiling.")
    except rospy.ROSInterruptException: 
        pass


if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('phdslam')
        girona500_navigator = G500_SLAM(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException: 
        pass

