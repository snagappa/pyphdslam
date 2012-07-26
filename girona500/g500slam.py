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
roslib.load_manifest('g500slam')
import rospy
import tf
import PyKDL
import math
import code
import copy
import threading

# Msgs imports
from navigation_g500.msg import TeledyneExplorerDvl, ValeportSoundVelocity, \
    FastraxIt500Gps
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from auv_msgs.msg import NavSts
from std_srvs.srv import Empty, EmptyResponse
from navigation_g500.srv import SetNE, SetNEResponse #, SetNERequest

import numpy as np
import girona500
from lib.phdfilter.phdfilter import PARAMETERS
from lib.common import pointclouds
from lib.common.kalmanfilter import sigma_pts
import featuredetector

INVALID_ALTITUDE = -32665
SAVITZKY_GOLAY_COEFFS = [0.2,  0.1,  0. , -0.1, -0.2]

UKF_ALPHA = 2
UKF_BETA = 2
UKF_KAPPA = 1

# Profile options
__PROFILE__ = False
__PROFILE_NUM_LOOPS__ = 1000
#code.interact(local=locals())

def normalizeAngle(np_array):
    return (np_array + (2.0*np.pi*np.floor((np.pi - np_array)/(2.0*np.pi))))

    
class G500_SLAM():
    def __init__(self, name):
        # Get config
        self.config = girona500.get_config()
        
        # Main SLAM worker
        self.slam_worker = self.init_slam()
        
        # Structure to store vehicle pose from sensors
        self.vehicle = PARAMETERS()
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
        self.ros = PARAMETERS()
        self.ros.name = name
        self.ros.last_update_time = rospy.Time.now()
        self.ros.NO_LOCK_ACQUIRE = 0
        self.init_ros()
        self.__LOCK__ = threading.Lock()
        # Set as true to run without gps/imu initialisation
        self.config.init = False
        
        
    def init_slam(self):
        # Get default parameters
        slam_properties = girona500.g500_slam_fn_defs()
        # Add new parameters from ros config
        # Process noise
        slam_properties.state_markov_predict_fn.parameters.process_noise = \
                        np.eye(3)*self.config.model_covariance
        # GPS observation noise
        slam_properties.state_likelihood_fn.parameters.gps_obs_noise = \
                        np.eye(2)*self.config.gps_covariance
        # DVL bottom velocity noise
        slam_properties.state_likelihood_fn.parameters.dvl_bottom_noise = \
                        np.eye(3)*self.config.dvl_bottom_covariance
        # DVL water velocity noise
        slam_properties.state_likelihood_fn.parameters.dvl_water_noise = \
                        np.eye(3)*self.config.dvl_water_covariance
        slam_worker = girona500.G500_PHDSLAM(*slam_properties)
        
        ndims = slam_worker.parameters.state_parameters["ndims"]
        nparticles = slam_worker.parameters.state_parameters["nparticles"]
        if nparticles == 2*ndims+1:
            sc_process_noise = slam_worker.trans_matrices(np.zeros(3), 1.0)[1] + slam_worker.trans_matrices(np.zeros(3), 0.01)[1]
            slam_worker.states = sigma_pts(np.zeros((1, ndims)), sc_process_noise, _alpha=UKF_ALPHA, _beta=UKF_BETA, _kappa=UKF_KAPPA)[0]
        return slam_worker
        
        
    def init_ros(self):
        config = self.config
        
        #Create static transformations
        config.dvl_tf = self.computeTf(config.dvl_tf_data)
        config.imu_tf = self.computeTf(config.imu_tf_data)        
        config.svs_tf = self.computeTf(config.svs_tf_data)
        
        #Initialize flags
        config.init = False
        config.imu_data = False
        config.gps_data = not config.gps_update
        config.init_north = 0.0
        config.init_east = 0.0
        time_now = rospy.Time.now()
        config.last_prediction = copy.copy(time_now)
        config.altitude = INVALID_ALTITUDE
        config.bottom_status = 0
        
        #init last sensor update
        config.init_time = copy.copy(time_now)
        config.gps_last_update = copy.copy(time_now)
        config.dvl_last_update = copy.copy(time_now)
        config.imu_last_update = copy.copy(time_now)
        config.svs_last_update = copy.copy(time_now)
        config.map_last_update = copy.copy(time_now)
        config.dvl_init = False
        config.imu_init = False
        config.svs_init = False
        config.gps_init_samples_list = []
        
        # Buffer for smoothing the yaw rate
        config.heading_buffer = []
        config.savitzky_golay_coeffs = SAVITZKY_GOLAY_COEFFS
        
        self.ros.pcl_helper = featuredetector.msgs.msgs(featuredetector.msgs.MSG_XYZ_COV)
        if not __PROFILE__:
            print "Creating ROS subscriptions..."
            # Create Subscriber
            rospy.Subscriber("/navigation_g500/teledyne_explorer_dvl", 
                             TeledyneExplorerDvl, self.updateTeledyneExplorerDvl,
                             queue_size=1)
            rospy.Subscriber("/navigation_g500/valeport_sound_velocity", 
                             ValeportSoundVelocity, 
                             self.updateValeportSoundVelocity, queue_size=1)
            rospy.Subscriber("/navigation_g500/imu", Imu, self.updateImu)
            if config.gps_update :
                rospy.Subscriber("/navigation_g500/fastrax_it_500_gps", 
                                 FastraxIt500Gps, self.updateGps, queue_size=1)
            ## Subscribe to visiona slam-features node
            rospy.Subscriber("/slamsim/features", PointCloud2, 
                             self.update_features, queue_size=1)
            # Subscribe to sonar slam features node for
            #rospy.Subscriber("/slam_features/fls_pcl", PointCloud2, 
            #                 self.update_features)
            
            #Create services
            self.reset_navigation = rospy.Service('/slam_g500/reset_navigation', 
                                                  Empty, self.resetNavigation)
            self.reset_navigation = rospy.Service('/slam_g500/set_navigation', 
                                                  SetNE, self.setNavigation)
            
        # Create publishers
        self.ros.nav = PARAMETERS()
        self.ros.nav.msg = NavSts()
        self.ros.nav.publisher = rospy.Publisher("/g500slam/nav_sts", NavSts)
        # Publish landmarks
        self.ros.map = PARAMETERS()
        self.ros.map.msg = PointCloud2()
        self.ros.map.publisher = rospy.Publisher("/g500slam/features", PointCloud2)
        self.ros.map.helper = featuredetector.msgs.msgs(featuredetector.msgs.MSG_XYZ_COV)
        
        # Publish data every 500 ms
        rospy.timer.Timer(rospy.Duration(0.5), self.publish_data)
        # Callback to print vehicle state and weight
        #rospy.timer.Timer(rospy.Duration(10), self.debug_print)
        
        
    def resetNavigation(self, req):
        print "Resetting navigation..."
        rospy.loginfo("%s: Reset Navigation", self.name)
        #self.slam_worker.states[:,0:2] = 0
        ndims = self.slam_worker.parameters.state_parameters["ndims"]
        nparticles = self.slam_worker.parameters.state_parameters["nparticles"]
        if nparticles == 2*ndims + 1:
            #pose_angle = tf.transformations.euler_from_quaternion(
            #                                    self.vehicle.pose_orientation)
            sc_process_noise = self.slam_worker.trans_matrices(np.zeros(3), 1.0)[1] + self.slam_worker.trans_matrices(np.zeros(3), 0.01)[1]
            self.slam_worker.states = sigma_pts(np.zeros((1, ndims)), sc_process_noise, _alpha=UKF_ALPHA, _beta=UKF_BETA, _kappa=UKF_KAPPA)[0]
        else:
            self.slam_worker.states[:, :] = 0
        return EmptyResponse()
        
    def setNavigation(self, req):
        print "Setting new navigation..."
        #rospy.loginfo("%s: Set Navigation to: \n%s", self.name, req) 
        #self.slam_worker.states[:,0] = req.north
        #self.slam_worker.states[:,1] = req.east
        ndims = self.slam_worker.parameters.state_parameters["ndims"]
        nparticles = self.slam_worker.parameters.state_parameters["nparticles"]
        if nparticles == 2*ndims + 1:
            #pose_angle = tf.transformations.euler_from_quaternion(
            #                                    self.vehicle.pose_orientation)
            sc_process_noise = self.slam_worker.trans_matrices(np.zeros(3), 1.0)[1] + self.slam_worker.trans_matrices(np.zeros(3), 0.01)[1]
            mean_state = np.array([[req.north, req.east, 0, 0, 0, 0]])
            self.slam_worker.states = sigma_pts(mean_state, sc_process_noise, _alpha=UKF_ALPHA, _beta=UKF_BETA, _kappa=UKF_KAPPA)[0]
        else:
            self.slam_worker.states[:,0] = req.north
            self.slam_worker.states[:,1] = req.east
        self.slam_worker.states[:,2] = self.vehicle.pose_position[2]
        ret = SetNEResponse()
        ret.success = True
        return ret
    
    def computeTf(self, transform):
        r = PyKDL.Rotation.RPY(math.radians(transform[3]), 
                               math.radians(transform[4]), 
                               math.radians(transform[5]))
        #rospy.loginfo("Rotation: %s", str(r))
        v = PyKDL.Vector(transform[0], transform[1], transform[2])
        #rospy.loginfo("Vector: %s", str(v))
        frame = PyKDL.Frame(r, v)
        #rospy.loginfo("Frame: %s", str(frame))
        return frame
    
    
    def updateGps(self, gps):
        if gps.data_quality >= 1 and gps.latitude_hemisphere >= 0 and gps.longitude_hemisphere >= 0:
            config = self.config
            config.gps_last_update = copy.copy(gps.header.stamp)
            if not config.gps_data :
                print "gps not set: initialising"
                config.gps_init_samples_list.append([gps.north, gps.east])
                if len(config.gps_init_samples_list) >= config.gps_init_samples:
                    config.gps_data = True
                    [config.init_north, config.init_east] = \
                            np.median(np.array(config.gps_init_samples_list), 
                                      axis=0)
                    #rospy.loginfo('%s, GPS init data: %sN, %sE', self.name, self.init_north, self.init_east)
            else:
                est_state = self.slam_worker._state_estimate_()
                distance = np.sqrt((est_state[0] - gps.north)**2 + 
                                (est_state[1] - gps.east)**2)
                #rospy.loginfo("%s, Distance: %s", self.name, distance)
                
                #Right now the GPS is only used to initialize the navigation not for updating it!!!
                if distance < 0.1:
                    z = np.array([gps.north, gps.east])
                    self.__LOCK__.acquire()
                    try:
                        if self.makePrediction(config.gps_last_update):
                            self.slam_worker.update_gps(z)
                            self.ros.last_update_time = config.gps_last_update
                            self.slam_worker.resample()
                    finally:
                        self.__LOCK__.release()
                    #self.publish_data()
                
        
    def updateTeledyneExplorerDvl(self, dvl):
        #print os.getpid()
        config = self.config
        config.dvl_last_update = copy.copy(dvl.header.stamp)
        config.dvl_init = True
        
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
            likelihood_params = \
                    self.slam_worker.parameters.state_likelihood_fn.parameters
            likelihood_params.dvl_obs_noise = \
                likelihood_params.dvl_bottom_noise if dvl_update==1 else \
                likelihood_params.dvl_water_noise
            if self.__LOCK__.locked():
                self.ros.NO_LOCK_ACQUIRE += 1
                #print str(self.ros.NO_LOCK_ACQUIRE), ": Could not acquire lock for dvl update"
                return
            self.__LOCK__.acquire()
            try:
                self.vehicle.twist_linear = np.array([vr[0], -vr[1], vr[2]])
                
                #Ara ja tenim la velocitat lineal en el DVL representada en eixos de vehicle
                #falta calcular la velocitat lineal al centre del vehicle en eixos del vehicle
                angular_velocity = self.vehicle.twist_angular
                
                self.vehicle.twist_linear -= np.cross(angular_velocity, distance)
                
                self.makePrediction(config.dvl_last_update)
                
                #self.slam_worker.state_likelihood_fn.parameters.dvl_obs_noise = dvl_reference
                self.slam_worker.update_dvl(self.vehicle.twist_linear)
                self.ros.last_update_time = config.dvl_last_update
                self.slam_worker.resample()
            finally:
                self.__LOCK__.release()
            #self.publish_data()
        else:
            rospy.loginfo('%s, invalid DVL velocity measurement!', self.name)
        
    
    def updateValeportSoundVelocity(self, svs):
        #print os.getpid()
        config = self.config
        config.svs_last_update = copy.copy(svs.header.stamp)
        svs_data = PyKDL.Vector(.0, .0, svs.pressure)
        pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
        vehicle_rpy = PyKDL.Rotation.RPY(*pose_angle)
        svs_trans = config.svs_tf.p
        svs_trans = vehicle_rpy * svs_trans
        svs_data = svs_data - svs_trans
        
        if not config.svs_init:
            config.svs_init = True
            self.__LOCK__.acquire()
            try:
                print "INITIALISING DEPTH to ", str(svs_data[2])
                self.vehicle.pose_position[2] = svs_data[2]
                self.slam_worker.states[:,2] = svs_data[2]
            finally:
                self.__LOCK__.release()
            return
        
        if self.__LOCK__.locked():
            return
        self.__LOCK__.acquire()
        try:
            self.vehicle.pose_position[2] = svs_data[2]
            if self.makePrediction(config.svs_last_update):
                #self.slam_worker.update_svs(self.vehicle.pose_position[2])
                self.slam_worker.states[:,2] = self.vehicle.pose_position[2]
                self.ros.last_update_time = config.svs_last_update
        finally:
            self.__LOCK__.release()
        #self.publish_data()

    
    def updateImu(self, imu):
        #print os.getpid()
        ret_val = None
        config = self.config
        config.imu_init = True
        
        pose_angle = tf.transformations.euler_from_quaternion(
                                       [imu.orientation.x, imu.orientation.y, 
                                       imu.orientation.z, imu.orientation.w])
        imu_data =  PyKDL.Rotation.RPY(*pose_angle)
        imu_data = imu_data*config.imu_tf.M
        pose_angle = imu_data.GetRPY()
        if not config.imu_data :
            config.last_imu_orientation = pose_angle
            config.imu_last_update = copy.copy(imu.header.stamp)
            #config.imu_data = True
            # Initialize heading buffer in order to apply a savitzky_golay derivation
            if len(config.heading_buffer) == 0:
                config.heading_buffer.append(pose_angle[2])
                
            inc = normalizeAngle(pose_angle[2] - config.heading_buffer[-1])
            config.heading_buffer.append(config.heading_buffer[-1] + inc)
            
            if len(config.heading_buffer) == len(config.savitzky_golay_coeffs):
                config.imu_data = True
            
        else:
            period = (imu.header.stamp - config.imu_last_update).to_sec()
            pose_angle_quaternion = tf.transformations.quaternion_from_euler(
                                                                    *pose_angle)
            config.last_imu_orientation = pose_angle
            self.__LOCK__.acquire()
            try:
                
                self.vehicle.pose_orientation = pose_angle_quaternion
                
                # Derive angular velocities from orientations #####################
                self.vehicle.twist_angular = normalizeAngle(
                 np.array(pose_angle)-np.array(config.last_imu_orientation))/period
                
                # For yaw rate we apply a savitzky_golay derivation
                inc = normalizeAngle(pose_angle[2] - config.heading_buffer[-1])
                config.heading_buffer.append(config.heading_buffer[-1] + inc)
                config.heading_buffer.pop(0)
                self.vehicle.twist_angular[2] = np.convolve(config.heading_buffer, 
                                                            config.savitzky_golay_coeffs, 
                                                            mode='valid') / period
                
                
                config.imu_last_update = copy.copy(imu.header.stamp)
                
                self.makePrediction(imu.header.stamp)
                self.ros.last_update_time = imu.header.stamp
                ###############################################################
            finally:
                self.__LOCK__.release()
            #self.publish_data()
        return ret_val
        
        
    def update_features(self, pcl_msg):
        self.__LOCK__.acquire()
        try:
            # Convert the pointcloud slam features into normal x,y,z
            # The pointcloud may have uncertainty on the points - this will be
            # the observation noise
            #slam_features = pointclouds.pointcloud2_to_xyz_array(pcl_msg)
            slam_features = self.ros.pcl_helper.from_pcl(pcl_msg)
            # We can now access the points as slam_features[i]
            self.slam_worker.update_with_features(slam_features, 
                                                  pcl_msg.header.stamp)
            #self.publish_data()
            #self.slam_worker.resample()
            self.config.map_last_update = copy.copy(pcl_msg.header.stamp)
            self.ros.last_update_time = pcl_msg.header.stamp
        finally:
            self.__LOCK__.release()
        
    
    def makePrediction(self, predict_to_time):
        config = self.config
        if not config.init:
            time_now = predict_to_time
            config.last_prediction = copy.copy(time_now)
            self.slam_worker.last_odo_predict_time = time_now.to_sec()
            if config.imu_data and config.gps_data:                
                # Initialise slam worker with north and east co-ordinates
                init = lambda:0
                init.north = config.init_north
                init.east = config.init_east
                self.slam_worker.reset_states()
                print "Resetting states to ", str(init.north), ", ", str(init.east)
                self.setNavigation(init)
                self.slam_worker.states[:,2] = self.vehicle.pose_position[2]
                config.init = True
            return False
        else:
            pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
            if predict_to_time <= config.last_prediction:
                self.ros.NO_LOCK_ACQUIRE += 1
                return False
            time_now = predict_to_time
            config.last_prediction = copy.copy(time_now)
            time_now = time_now.to_sec()
            self.slam_worker.predict(np.array(pose_angle), time_now)
            return True
            
    
    def publish_data(self, *args, **kwargs):
        if self.config.init:
            nav_msg = self.ros.nav.msg
            est_state = self.slam_worker._state_estimate_()
            est_cov = self.slam_worker._state_covariance_()
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
            nav_msg.body_velocity.x = est_state[3]
            nav_msg.body_velocity.y = est_state[4]
            nav_msg.body_velocity.z = est_state[5]
            nav_msg.orientation.roll = angle[0]
            nav_msg.orientation.pitch = angle[1]
            nav_msg.orientation.yaw = angle[2]
            nav_msg.orientation_rate.roll = self.vehicle.twist_angular[0]
            nav_msg.orientation_rate.pitch = self.vehicle.twist_angular[1]
            nav_msg.orientation_rate.yaw = self.vehicle.twist_angular[2]
            nav_msg.altitude = self.vehicle.altitude
            
            # Variance
            nav_msg.position_variance.north = est_cov[0,0]
            nav_msg.position_variance.east = est_cov[1,1]
            nav_msg.position_variance.depth = est_cov[2,2]
            
            nav_msg.status = np.uint8(np.log10(self.ros.NO_LOCK_ACQUIRE+1))
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
            map_estimate = self.slam_worker._map_estimate_()
            map_states = map_estimate.state.state
            map_covs = map_estimate.state.covariance
            if map_states.shape[0]:
                diag_cov = np.array([np.diag(map_covs[i]) for i in range(map_covs.shape[0])])
                pcl_msg = self.ros.map.helper.to_pcl(rospy.Time.now(), np.hstack((map_states, diag_cov)))
            else:
                pcl_msg = self.ros.map.helper.to_pcl(rospy.Time.now(), np.zeros(0))
            pcl_msg.header.frame_id = self.ros.name
            # and publish visible landmarks
            self.ros.map.publisher.publish(pcl_msg)
            #print "Landmarks at: "
            #print map_states
        #else :
        #    self.ros.last_update_time = rospy.Time.now()
        #    self.config.init = True
        
        
    def debug_print(self, *args, **kwargs):
        print "Weights: "
        #print self.slam_worker.states
        print self.slam_worker.weights
    

def main():
    try:
        # Init node
        rospy.init_node('phdslam')
        g500_slam = G500_SLAM(rospy.get_name())
        if not __PROFILE__:
            rospy.spin()
        else:
            imu_msg = rospy.wait_for_message("/navigation_g500/imu", Imu, 1)
            last_time = imu_msg.header.stamp
            #g500_slam.updateImu(imu_msg)
            if 1: #g500_slam.config.gps_update :
                try:
                    gps_msg = rospy.wait_for_message("/navigation_g500/fastrax_it_500_gps", 
                                                     FastraxIt500Gps, 1)
                    print "gps updates..."
                    for count in range(__PROFILE_NUM_LOOPS__):
                        last_time.secs += 1
                        gps_msg.header.stamp = last_time
                        g500_slam.updateGps(gps_msg)
                except rospy.ROSException:
                    print "*** timeout waiting for gps message! ***"
            else:
                print "*** no gps update ***"
            
            print "imu updates..."
            for count in range(__PROFILE_NUM_LOOPS__):
                last_time.secs += 1
                imu_msg.header.stamp = last_time
                g500_slam.updateImu(imu_msg)
            
            svs_msg = rospy.wait_for_message("/navigation_g500/valeport_sound_velocity", 
                                             ValeportSoundVelocity)
            print "svs updates..."
            for count in range(__PROFILE_NUM_LOOPS__):
                last_time.secs += 1
                svs_msg.header.stamp = last_time
                g500_slam.updateValeportSoundVelocity(svs_msg)
            
            dvl_msg = rospy.wait_for_message("/navigation_g500/teledyne_explorer_dvl", 
                                             TeledyneExplorerDvl)
            print "dvl updates..."
            for count in range(__PROFILE_NUM_LOOPS__):
                last_time.secs += 1
                dvl_msg.header.stamp = last_time
                g500_slam.updateTeledyneExplorerDvl(dvl_msg)
            rospy.signal_shutdown("Finished profiling.")
    except rospy.ROSInterruptException: pass


if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('phdslam')
        g500_slam = G500_SLAM(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException: pass

