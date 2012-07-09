# -*- coding: utf-8 -*-

#!/usr/bin/env python

# ROS imports
import roslib 
roslib.load_manifest('navigation_g500')
import rospy
import tf
from tf.transformations import euler_from_quaternion
import tf_conversions.posemath as pm
import PyKDL
import math
import cola2_lib
from ekf_g500 import EKFG500
from pf_g500 import PFG500
import threading

# Msgs imports
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from navigation_g500.msg import *
from sensor_msgs.msg import Imu
from auv_msgs.msg import NavSts
from safety_g500.msg import NavSensorsStatus
from std_srvs.srv import Empty, EmptyResponse
from auv_msgs.srv import SetNE, SetNEResponse, SetNERequest

import girona500

# Python imports
from numpy import *

INVALID_ALTITUDE = -32665
SAVITZKY_GOLAY_COEFFS = [0.2,  0.1,  0. , -0.1, -0.2]
# Larger filter introduce to much delay
# SAVITZKY_GOLAY_COEFFS = [0.10714286, 0.07142857, 0.03571429, 0., -0.03571429, -0.07142857, -0.10714286]
# SAVITZKY_GOLAY_COEFFS = [-0.05827506,  0.05710956,  0.1033411 ,  0.09770785,  0.05749806, 0. , -0.05749806, -0.09770785, -0.1033411 , -0.05710956, 0.05827506]
       
class Navigator :

    
    def __init__(self, name):
        """ Merge different navigation sensor values  """
        self.name = name
        self.config = girona500.get_config()
        config = self.config
        #Init Kalman filter
        self.filter = PFG500(config.model_covariance, 
                           config.gps_covariance, 
                           config.dvl_bottom_covariance,
                           config.dvl_water_covariance)
        self.init_ros()
        self.LOCK = threading.RLock()
   
    def init_ros(self):
        config = self.config
        # To filter yaw rate
        config.heading_buffer = []
        config.savitzky_golay_coeffs = SAVITZKY_GOLAY_COEFFS
       
        # Create Odometry msg
        config.odom = Odometry()
        config.nav = NavSts()
        
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
        config.altitude = INVALID_ALTITUDE
        config.bottom_status = 0
        
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
       
        # Subscribe to sonar slam features node for
        #rospy.Subscriber("/slam_features/fls_pcl", PointCloud2, 
        #                 self.update_features)
        
        # Create publisher
        self.pub_odom = rospy.Publisher("/g500slam/odometry", Odometry)
        self.pub_nav_sts = rospy.Publisher("/g500slam/nav_sts", NavSts)
        self.pub_nav_sensors_status = rospy.Publisher("/g500slam/nav_sensors_status", NavSensorsStatus)
        
        #Create services
        self.reset_navigation = rospy.Service('/slam_g500/reset_navigation', 
                                              Empty, self.resetNavigation)
        self.reset_navigation = rospy.Service('/slam_g500/set_navigation', 
                                              SetNE, self.setNavigation)
    
    
    def resetNavigation(self, req):
        rospy.loginfo("%s: Reset Navigation", self.name)    
        x = self.filter.getStateVector()
        x[0] = 0.0
        x[1] = 0.0
        self.filter.initialize(x)
        return EmptyResponse()
        
        
    def setNavigation(self, req):
        rospy.loginfo("%s: Set Navigation to: \n%s", self.name, req)    
        x = self.filter.getStateVector()
        x[0] = req.north
        x[1] = req.east
        self.filter.initialize(x)
        ret = SetNEResponse()
        ret.success = True
        return ret
    
    
    def updateGps(self, gps):
        try:
            self.LOCK.acquire()
            self._updateGps_(gps)
        finally:
            self.LOCK.release()
    
    def _updateGps_(self, gps):
        config = self.config
        if gps.data_quality >= 1 and gps.latitude_hemisphere >= 0 and gps.longitude_hemisphere >= 0:
            if not config.gps_data :
                config.gps_init_samples_list.append([gps.north, gps.east])
                if len(config.gps_init_samples_list) >= config.gps_init_samples:
                    config.gps_data = True
                    [config.init_north, config.init_east] = median(array(config.gps_init_samples_list), axis=0)
                    rospy.loginfo('%s, GPS init data: %sN, %sE', self.name, config.init_north, config.init_east)
            else:
                distance = sqrt((config.nav.position.north - gps.north)**2 + 
                                (config.nav.position.east - gps.east)**2)
                #rospy.loginfo("%s, Distance: %s", self.name, distance)
                
                #TODO: Roght now the GPS is only used to initialize the navigation not for updating it!!!
                if distance < 0.1:
                    # If the error between the filter and the GPS is small, update the kalman filter
                    config.odom.pose.pose.position.x = gps.north
                    config.odom.pose.pose.position.y = gps.east
                    
                    # Update EKF
                    if self.makePrediction(gps.header.stamp):
                        z = array([gps.north, gps.east])
                        self.filter.gpsUpdate(z)
                        self.publishData()
                        
        
    
    def updateTeledyneExplorerDvl(self, dvl):
        try:
            self.LOCK.acquire()
            self._updateTeledyneExplorerDvl_(dvl)
        finally:
            self.LOCK.release()
            
    def _updateTeledyneExplorerDvl_(self, dvl):
        config = self.config
        config.dvl_last_update = dvl.header.stamp #rospy.Time.now()
        config.dvl_init = True
        
        # If dvl_update == 0 --> No update
        # If dvl_update == 1 --> Update wrt bottom
        # If dvl_update == 2 --> Update wrt water
        dvl_update = 0
        
        if dvl.bi_status == "A" and dvl.bi_error > -32.0:
            if abs(dvl.bi_x_axis) < config.dvl_max_v and abs(dvl.bi_y_axis) < config.dvl_max_v and abs(dvl.bi_z_axis) < config.dvl_max_v : 
                v = PyKDL.Vector(dvl.bi_x_axis, dvl.bi_y_axis, dvl.bi_z_axis)
                dvl_update = 1
        elif dvl.wi_status == "A" and dvl.wi_error > -32.0:
            if abs(dvl.wi_x_axis) < config.dvl_max_v and abs(dvl.wi_y_axis) < config.dvl_max_v and abs(dvl.wi_z_axis) < config.dvl_max_v : 
                v = PyKDL.Vector(dvl.wi_x_axis, dvl.wi_y_axis, dvl.wi_z_axis)
                dvl_update = 2
        
        #Filter to check if the altitude is reliable
        if dvl.bi_status == "A" and dvl.bi_error > -32.0:
            config.bottom_status =  config.bottom_status + 1
        else:
            config.bottom_status = 0
        
        if config.bottom_status > 4:
            config.altitude = dvl.bd_range
        else:
            config.altitude = INVALID_ALTITUDE
            
        if dvl_update != 0:
            #Rotate DVL velocities and Publish
            #Compte! EL DVL no es dextrogir i s'ha de negar la Y
            vr = config.dvl_tf.M * v
            vr2 = array([vr[0], -vr[1], vr[2]])
            
            #Ara ja tenim la velocitat lineal en el DVL representada en eixos de vehicle
            #falta calcular la velocitat lineal al centre del vehicle en eixos del vehicle
            angular_velocity = array([config.odom.twist.twist.angular.x,
                                      config.odom.twist.twist.angular.y,
                                      config.odom.twist.twist.angular.z])
            distance = config.dvl_tf_data[0:3]
            vr2 = vr2 - cross(angular_velocity, distance)
            
            #Copy data to odometry message
            config.odom.twist.twist.linear.x = vr2[0]
            config.odom.twist.twist.linear.y = vr2[1]
            config.odom.twist.twist.linear.z = vr2[2]
            
            # Update EKF
            if self.makePrediction(dvl.header.stamp):
                z = array([vr2[0], vr2[1], vr2[2]])
                if dvl_update == 1: 
                    self.filter.dvlUpdate(z, 'bottom')
                else:
                    self.filter.dvlUpdate(z, 'water')
                self.publishData()
        else:
            rospy.loginfo('%s, invalid DVL velocity measurement!', self.name)
        
       
        
    
    def updateValeportSoundVelocity(self, svs):
        config = self.config
        config.svs_last_update = svs.header.stamp #rospy.Time.now()
        config.svs_init = True
        
        svs_data = PyKDL.Vector(.0, .0, svs.pressure)
        angle = tf.transformations.euler_from_quaternion([config.odom.pose.pose.orientation.x,
                                                          config.odom.pose.pose.orientation.y,
                                                          config.odom.pose.pose.orientation.z,
                                                          config.odom.pose.pose.orientation.w])
        vehicle_rpy = PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
        svs_trans = config.svs_tf.p
        svs_trans = vehicle_rpy * svs_trans
        svs_data = svs_data + svs_trans
        config.odom.pose.pose.position.z = svs_data[2] 


    def updateImu(self, imu):
        config = self.config
        config.imu_last_update = imu.header.stamp #rospy.Time.now()
        config.imu_init = True
        
        if not config.imu_data :
            angle = euler_from_quaternion([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w])
            imu_data =  PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
            imu_data = config.imu_tf.M * imu_data
            angle = imu_data.GetRPY()   
            config.last_imu_orientation = angle
            config.last_imu_update = imu.header.stamp
            
            # Initialize heading buffer in order to apply a savitzky_golay derivation
            if len(config.heading_buffer) == 0:
                config.heading_buffer.append(angle[2])
                
            inc = cola2_lib.normalizeAngle(angle[2] - config.heading_buffer[-1])
            config.heading_buffer.append(config.heading_buffer[-1] + inc)
            
            if len(config.heading_buffer) == len(config.savitzky_golay_coeffs):
                config.imu_data = True
        else:
            angle = euler_from_quaternion([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w])
            imu_data =  PyKDL.Rotation.RPY(angle[0], angle[1], angle[2])
            imu_data = config.imu_tf.M * imu_data
            angle = imu_data.GetRPY()   
            angle_quaternion = tf.transformations.quaternion_from_euler(angle[0], angle[1], angle[2])
            config.odom.pose.pose.orientation.x = angle_quaternion[0]
            config.odom.pose.pose.orientation.y = angle_quaternion[1]
            config.odom.pose.pose.orientation.z = angle_quaternion[2]
            config.odom.pose.pose.orientation.w = angle_quaternion[3]
            
            # Derive angular velocities from orientations ####################### 
            period = (imu.header.stamp - config.last_imu_update).to_sec()
            
            # For yaw rate we apply a savitzky_golay derivation
            inc = cola2_lib.normalizeAngle(angle[2] - config.heading_buffer[-1])
            config.heading_buffer.append(config.heading_buffer[-1] + inc)
            config.heading_buffer.pop(0)
            config.odom.twist.twist.angular.z = convolve(config.heading_buffer, config.savitzky_golay_coeffs, mode='valid') / period

            # TODO: Roll rate and Pitch rate should be also savitzky_golay derivations
            config.odom.twist.twist.angular.x = cola2_lib.normalizeAngle(angle[0] - config.last_imu_orientation[0]) / period 
            config.odom.twist.twist.angular.y = cola2_lib.normalizeAngle(angle[1] - config.last_imu_orientation[1]) / period 

            config.last_imu_orientation = angle
            config.last_imu_update = imu.header.stamp          
            #####################################################################
            
            try:
                self.LOCK.acquire()
                if self.makePrediction(imu.header.stamp):
                    self.filter.updatePrediction()
                    self.publishData()
            finally:
                self.LOCK.release()

        
    def makePrediction(self, time=None):
        config = self.config
        if time==None:
            time = rospy.Time.now()
        if not config.init :
            if config.imu_data and config.gps_data:
                config.last_prediction = time
                self.filter.initialize(array([config.init_north, config.init_east, 0.0, 0.0, 0.0]))
                config.init = True
            return False
        else :
            angle = euler_from_quaternion([config.odom.pose.pose.orientation.x,
                                           config.odom.pose.pose.orientation.y,
                                           config.odom.pose.pose.orientation.z,
                                           config.odom.pose.pose.orientation.w])
            
            t = (time - config.last_prediction).to_sec()
            config.last_prediction = time
            self.filter.prediction(angle, t)
            return True
        
            
    def publishData(self):
        config = self.config
        if config.init:
            x = self.filter.getStateVector()
            angle = euler_from_quaternion([config.odom.pose.pose.orientation.x,
                                           config.odom.pose.pose.orientation.y,
                                           config.odom.pose.pose.orientation.z,
                                           config.odom.pose.pose.orientation.w])            
            
        #   Create header    
            config.odom.header.stamp = rospy.Time.now()
            config.odom.header.frame_id = "girona500"
            config.odom.child_frame_id = "world"
                       
            #Fil Nav status topic
            config.nav.header = config.odom.header
            config.nav.position.north = x[0]
            config.nav.position.east = x[1]
            config.nav.position.depth = config.odom.pose.pose.position.z
            config.nav.body_velocity.x = x[2]
            config.nav.body_velocity.y = x[3]
            config.nav.body_velocity.z = x[4]
            config.nav.orientation.roll = angle[0]
            config.nav.orientation.pitch = angle[1]
            config.nav.orientation.yaw = angle[2]
            config.nav.orientation_rate.roll = config.odom.twist.twist.angular.x
            config.nav.orientation_rate.pitch = config.odom.twist.twist.angular.y
            config.nav.orientation_rate.yaw = config.odom.twist.twist.angular.z
            config.nav.altitude = config.altitude
            
            covariance = self.filter.get_covariance()
            config.nav.position_variance.north = covariance[0,0]
            config.nav.position_variance.east = covariance[1,1]
            
            #Publish topics
            self.pub_nav_sts.publish(config.nav)
            self.pub_odom.publish(config.odom)
            
            #Publish TF
            br = tf.TransformBroadcaster()
            br.sendTransform((config.nav.position.north, config.nav.position.east, config.nav.position.depth),                   
                             tf.transformations.quaternion_from_euler(config.nav.orientation.roll, config.nav.orientation.pitch, config.nav.orientation.yaw),
                             config.odom.header.stamp,
                             config.odom.header.frame_id,
                             config.odom.child_frame_id)
        else :
            config.last_time = rospy.Time.now()
            config.init = True
    
    
    def computeTf(self, tf):
        r = PyKDL.Rotation.RPY(math.radians(tf[3]), math.radians(tf[4]), math.radians(tf[5]))
#        rospy.loginfo("Rotation: %s", str(r))
        v = PyKDL.Vector(tf[0], tf[1], tf[2])
#        rospy.loginfo("Vector: %s", str(v))
        frame = PyKDL.Frame(r, v)
#        rospy.loginfo("Frame: %s", str(frame))
        return frame
        
    
if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('phdslam')
        navigator = Navigator(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException: pass
