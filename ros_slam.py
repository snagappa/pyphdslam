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

# Msgs imports
from navigation_g500.msg import TeledyneExplorerDvl, ValeportSoundVelocity, \
    FastraxIt500Gps
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
from auv_msgs.msg import NavSts
from std_srvs.srv import Empty, EmptyResponse
from auv_msgs.srv import SetNE, SetNEResponse

import numpy as np
import girona500
from phdfilter import PARAMETERS

INVALID_ALTITUDE = -32665


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
        self.vehicle.twist_angular = np.zeros(0, dtype=float)
        self.vehicle.twist_linear = np.zeros(0, dtype=float)
        # position x, y, z - the state space estimated by the filter.
        self.vehicle.pose_position = np.zeros(0, dtype=float)
        # orientation quaternion x,y,z,w - same as odom.pose.pose.orientation
        self.vehicle.pose_orientation = np.zeros(0, dtype=float)
        # Altitude from dvl
        self.vehicle.altitude = 0.0
        
        # Initialise ROS stuff
        self.name = name
        self.ros = PARAMETERS()
        self.ros.last_update_time = rospy.Time.now()
        self.init_ros()
        
        
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
        return girona500.G500_PHDSLAM(*slam_properties)
        
        
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
        config.last_prediction = rospy.Time.now()
        config.altitude = -1.0
        
        #init last sensor update
        config.init_time = rospy.Time.now()
        config.gps_last_update = config.init_time
        config.dvl_last_update = config.init_time
        config.imu_last_update = config.init_time
        config.svs_last_update = config.init_time
        config.dvl_init = False
        config.imu_init = False
        config.svs_init = False
        config.gps_init_samples_list = []
        
        # Create Subscriber
        rospy.Subscriber("/navigation_g500/teledyne_explorer_dvl", 
                         TeledyneExplorerDvl, self.updateTeledyneExplorerDvl)
        rospy.Subscriber("/navigation_g500/valeport_sound_velocity", 
                         ValeportSoundVelocity, 
                         self.updateValeportSoundVelocity)
        rospy.Subscriber("/navigation_g500/imu", Imu, self.updateImu)
        if config.gps_update :
            rospy.Subscriber("/navigation_g500/fastrax_it_500_gps", 
                             FastraxIt500Gps, self.updateGps)
        
        # Subscribe to visiona slam-features node
        rospy.Subscriber("/slam_features/vision_pcl", PointCloud2, 
                         self.update_features)
        # Subscribe to sonar slam features node for
        #rospy.Subscriber("/slam_features/fls_pcl", PointCloud2, 
        #                 self.update_features)
        
        # Create publisher
        self.ros.nav_msg = NavSts()
        self.ros.nav_sts_publisher = rospy.Publisher("/g500slam/nav_sts", 
                                                     NavSts)
        
        #Create services
        self.reset_navigation = rospy.Service('/slam_g500/reset_navigation', 
                                              Empty, self.resetNavigation)
        self.reset_navigation = rospy.Service('/slam_g500/set_navigation', 
                                              SetNE, self.setNavigation)
    
    def resetNavigation(self, req):
        rospy.loginfo("%s: Reset Navigation", self.name)
        self.slam_worker.states[:,0:2] = 0
        return EmptyResponse()
        
    def setNavigation(self, req):
        rospy.loginfo("%s: Set Navigation to: \n%s", self.name, req)    
        self.slam_worker.states[:,0] = req.north
        self.slam_worker.states[:,1] = req.east
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
            config.gps_last_update = gps.header.stamp
            if not config.gps_data :
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
                    if self.makePrediction(config.gps_last_update):
                        #z = array([gps.north, gps.east])
                        self.setNavigation(gps)
                        self.ros.last_update_time = config.gps_last_update
                        self.publish_data()
                        
        
        
    def updateTeledyneExplorerDvl(self, dvl):
        config = self.config
        config.dvl_last_update = dvl.header.stamp
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
            self.vehicle.twist_linear = np.array([vr[0], -vr[1], vr[2]])
            
            #Ara ja tenim la velocitat lineal en el DVL representada en eixos de vehicle
            #falta calcular la velocitat lineal al centre del vehicle en eixos del vehicle
            angular_velocity = self.vehicle.twist_angular
            distance = config.dvl_tf_data[0:3]
            self.vehicle.twist_linear -= np.cross(angular_velocity, distance)
            
            self.makePrediction(config.dvl_last_update)
            #dvl_reference = "bottom" if dvl_update == 1 else "water"
            likelihood_params = \
                    self.slam_worker.parameters.state_likelihood_fn.parameters
            likelihood_params.dvl_obs_noise = \
                likelihood_params.dvl_bottom_noise if dvl_update==1 else \
                likelihood_params.dvl_water_noise
            #self.slam_worker.state_likelihood_fn.parameters.dvl_obs_noise = dvl_reference
            self.slam_worker.update_dvl(self.vehicle.twist_linear)
            self.ros.last_update_time = config.dvl_last_update
            self.publish_data()
        else:
            rospy.loginfo('%s, invalid DVL velocity measurement!', self.name)
        
    
    def updateValeportSoundVelocity(self, svs):
        config = self.config
        config.svs_last_update = rospy.Time.now()
        config.svs_init = True
        
        svs_data = PyKDL.Vector(.0, .0, svs.pressure)
        pose_angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)
        vehicle_rpy = PyKDL.Rotation.RPY(*pose_angle)
        svs_trans = config.svs_tf.p
        svs_trans = vehicle_rpy * svs_trans
        svs_data = svs_data + svs_trans
        self.vehicle.pose_position[2] = svs_data[2]
        self.makePrediction(config.dvl_last_update)
        self.slam_worker.update_svs(self.vehicle.pose_position[2])
        self.ros.last_update_time = config.svs_last_update
        self.publish_data()


    def updateImu(self, imu):
        config = self.config
        config.imu_init = True
        
        pose_angle = tf.transformations.euler_from_quaternion(
                                       [imu.orientation.x, imu.orientation.y, 
                                       imu.orientation.z, imu.orientation.w])
        imu_data =  PyKDL.Rotation.RPY(*pose_angle)
        imu_data = config.imu_tf.M * imu_data
        pose_angle = imu_data.GetRPY()
        if not config.imu_data :
            config.last_imu_orientation = pose_angle
            config.imu_last_update = imu.header.stamp
            config.imu_data = True
            
        else:
            pose_angle_quaternion = tf.transformations.quaternion_from_euler(
                                                                *pose_angle)
            self.vehicle.pose_orientation = pose_angle_quaternion
            
            # Derive angular velocities from orientations #####################
            period = (imu.header.stamp - config.imu_last_update).to_sec()
            
            self.vehicle.twist_angular = normalizeAngle(
                                pose_angle-config.last_imu_orientation)/period
            
            config.last_imu_orientation = pose_angle
            config.imu_last_update = imu.header.stamp          
            ###################################################################
            
            self.makePrediction(config.imu_last_update)
            self.ros.last_update_time = config.imu_last_update
            self.publish_data()

        
    def makePrediction(self, predict_to_time):
        config = self.config
        if not config.init:
            if config.imu_data and config.gps_data:
                #time_now = rospy.Time.now()
                time_now = predict_to_time
                config.last_prediction = time_now
                self.slam_worker.last_odo_predict_time = time_now.to_sec()
                # Initialise slam worker with north and east co-ordinates
                init = lambda:0
                init.north = config.init_north
                init.east = config.init_east
                self.slam_worker.reset_states()
                self.setNavigation(init)
                config.init = True
            return False
        else:
            pose_angle = tf.transformations.euler_from_quaternion(
                                                *self.vehicle.pose_orientation)
            #time_now = rospy.Time.now()
            time_now = predict_to_time
            config.last_prediction = time_now
            time_now = time_now.to_sec()
            self.slam_worker.predict_state(pose_angle, time_now)
            return True
            
        
    def publish_data(self):
        if self.init:
            nav_msg = self.ros.nav_msg
            est_state = self.slam_worker._state_estimate_()
            angle = tf.transformations.euler_from_quaternion(
                                                self.vehicle.pose_orientation)            
            
            # Create header
            nav_msg.header.stamp = self.ros.last_update_time
            nav_msg.header.frame_id = "g500slam"
            nav_msg.child_frame_id = "world"
                       
            #Fil Nav status topic
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
            
            #Publish topics
            self.ros.nav_sts_publisher.publish(nav_msg)
            
            #Publish TF
            br = tf.TransformBroadcaster()
            br.sendTransform(
                (nav_msg.position.north, 
                    nav_msg.position.east, 
                    nav_msg.position.depth),
                tf.transformations.quaternion_from_euler(
                    nav_msg.orientation.roll, 
                    self.nav.orientation.pitch, 
                    self.nav.orientation.yaw),
                nav_msg.header.stamp, 
                nav_msg.header.frame_id, 
                nav_msg.child_frame_id)
        else :
            self.ros.last_update_time = rospy.Time.now()
            self.init = True
    


if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('phdslam')
        g500_slam = G500_SLAM(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException: pass
