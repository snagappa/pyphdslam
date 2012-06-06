# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:42:43 2012

@author: snagappa
"""

# ROS imports
import roslib 
roslib.load_manifest('g500slam')
import rospy
import tf
import PyKDL
import math

# Msgs imports
from navigation_g500.msg import TeledyneExplorerDvl, ValeportSoundVelocity, FastraxIt500Gps
from sensor_msgs.msg import Imu
#from safety_g500.msg import NavSensorsStatus
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
        self.getConfig()
        
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
        
        # Initialise ROS stuff
        self.init_ros(name)
        
        
    def init_slam(self):
        # Get default parameters
        slam_properties = girona500.g500_slam_fn_defs()
        # Add new parameters from ros config
        slam_properties.state_markov_predict_fn.parameters.process_noise = \
                                                   self.config.model_covariance
        slam_properties.state_likelihood_fn.parameters.gps_obs_noise = \
                                            self.config.gps_position_covariance
        slam_properties.state_likelihood_fn.parameters.dvl_obs_noise = \
                                            self.config.dvl_velocity_covariance
        return girona500.G500_PHDSLAM(*slam_properties)
        
        
    def init_ros(self, name):
        self.name = name
        config = getattr(self, "config")
        
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
        config.dvl_last_update = config.init_time
        config.imu_last_update = config.init_time
        config.svs_last_update = config.init_time
        config.dvl_init = False
        config.imu_init = False
        config.svs_init = False
        config.gps_init_samples_list = []
        
        # Create Subscriber
        rospy.Subscriber("/navigation_g500/teledyne_explorer_dvl", TeledyneExplorerDvl, self.updateTeledyneExplorerDvl)
        rospy.Subscriber("/navigation_g500/valeport_sound_velocity", ValeportSoundVelocity, self.updateValeportSoundVelocity)
        rospy.Subscriber("/navigation_g500/imu", Imu, self.updateImu)
        if self.gps_update :
            rospy.Subscriber("/navigation_g500/fastrax_it_500_gps", FastraxIt500Gps, self.updateGps)

        #Create services
        self.reset_navigation = rospy.Service('/slam_g500/reset_navigation', Empty, self.resetNavigation)
        self.reset_navigation = rospy.Service('/slam_g500/set_navigation', SetNE, self.setNavigation)
    
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
    
    def computeTf(self, tf):
        r = PyKDL.Rotation.RPY(math.radians(tf[3]), math.radians(tf[4]), math.radians(tf[5]))
        #rospy.loginfo("Rotation: %s", str(r))
        v = PyKDL.Vector(tf[0], tf[1], tf[2])
        #rospy.loginfo("Vector: %s", str(v))
        frame = PyKDL.Frame(r, v)
        #rospy.loginfo("Frame: %s", str(frame))
        return frame
    
    def getConfig(self):
        self.config = PARAMETERS()
        if rospy.has_param("teledyne_explorer_dvl/tf") :
            self.config.dvl_tf_data = np.array(rospy.get_param("teledyne_explorer_dvl/tf"))
        else:
            rospy.logfatal("teledyne_explorer_dvl/tf param not found")

        if rospy.has_param("tritech_igc_gyro/tf") :
            self.config.imu_tf_data = np.array(rospy.get_param("tritech_igc_gyro/tf"))
        else:
            rospy.logfatal("tritech_igc_gyro/tf param not found")

        if rospy.has_param("valeport_sound_velocity/tf") :
            self.config.svs_tf_data = np.array(rospy.get_param("valeport_sound_velocity/tf"))
        else:
            rospy.logfatal("valeport_sound_velocity/tf param not found")
            
#       Sensors & model covariance
        if rospy.has_param("localization/dvl_covariance") :
            self.config.dvl_velocity_covariance = np.array(rospy.get_param('localization/dvl_covariance'))
        else:
            rospy.logfatal("localization/dvl_covariance param not found")
        
        if rospy.has_param("localization/gps_covariance") :
            self.config.gps_position_covariance = rospy.get_param('localization/gps_covariance')
        else:
            rospy.logfatal("localization/gps_covariance param not found")
            
        if rospy.has_param("localization/model_covariance") :
            self.config.model_covariance = rospy.get_param('localization/model_covariance')
        else:
            rospy.logfatal("localization/model_covariance param not found")
        
        if rospy.has_param("localization/dvl_max_v") :
            self.config.dvl_max_v = rospy.get_param('localization/dvl_max_v')
        else:
            rospy.logfatal("localization/dvl_max_v not found")
            
        if rospy.has_param("localization/gps_update"):
            self.config.gps_update = rospy.get_param('localization/gps_update')
        else:
            rospy.logfatal("localization/gps_update not found")
        
        if rospy.has_param("localization/gps_init_samples"):
            self.config.gps_init_samples = rospy.get_param('localization/gps_init_samples')
        else:
            rospy.logfatal("localization/gps_init_samples not found")
            
        if rospy.has_param("localization/check_sensors_period"):
            self.config.check_sensors_period = rospy.get_param('localization/check_sensors_period')
        else:
            rospy.logfatal("localization/check_sensors_period not found")
            
        if rospy.has_param("localization/dvl_max_period_error"):
            self.config.dvl_max_period_error = rospy.get_param('localization/dvl_max_period_error')
        else:
            rospy.logfatal("localization/dvl_max_period_error not found")

        if rospy.has_param("localization/svs_max_period_error"):
            self.config.svs_max_period_error = rospy.get_param('localization/svs_max_period_error')
        else:
            rospy.logfatal("localization/csvs_max_period_error not found")
            
        if rospy.has_param("localization/imu_max_period_error"):
            self.config.imu_max_period_error = rospy.get_param('localization/imu_max_period_error')
        else:
            rospy.logfatal("localization/imu_max_period_error not found")
            
        if rospy.has_param("localization/max_init_time") :
            self.config.max_init_time = rospy.get_param("localization/max_init_time")
        else:
            rospy.logfatal("localization/max_init_time not found in param list")
            
    
    def updateGps(self, gps):
        if gps.data_quality >= 1 and gps.latitude_hemisphere >= 0 and gps.longitude_hemisphere >= 0:
            config = self.config
            if not config.gps_data :
                config.gps_init_samples_list.append([gps.north, gps.east])
                if len(config.gps_init_samples_list) >= config.gps_init_samples:
                    config.gps_data = True
                    [config.init_north, config.init_east] = np.median(np.array(config.gps_init_samples_list), axis=0)
                    #rospy.loginfo('%s, GPS init data: %sN, %sE', self.name, self.init_north, self.init_east)
            else:
                est_state = self.slam_worker.get_state_estimate()
                distance = np.sqrt((est_state[0] - gps.north)**2 + 
                                (est_state[1] - gps.east)**2)
                #rospy.loginfo("%s, Distance: %s", self.name, distance)
                
                #Right now the GPS is only used to initialize the navigation not for updating it!!!
                if distance < 0.1:
                    if self.makePrediction():
                        #z = array([gps.north, gps.east])
                        self.setNavigation(gps)
                        #self.publishData()
                        
        
        
    def updateTeledyneExplorerDvl(self, dvl):
        config = self.config
        config.dvl_last_update = rospy.Time.now()
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
            config.altitude = dvl.bd_range
        else:
            config.altitude = INVALID_ALTITUDE
            
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
            
            self.makePrediction()
            dvl_reference = "bottom" if dvl_update == 1 else "water"
            self.vehicle.dvlUpdate(self.vehicle.twist_linear, dvl_reference)
            
        else:
            rospy.loginfo('%s, invalid DVL velocity measurement!', self.name)
        
    
    def updateValeportSoundVelocity(self, svs):
        config = self.config
        config.svs_last_update = rospy.Time.now()
        config.svs_init = True
        
        svs_data = PyKDL.Vector(.0, .0, svs.pressure)
        pose_angle = tf.transformations.euler_from_quaternion(self.vehicle.pose_orientation)
        vehicle_rpy = PyKDL.Rotation.RPY(*pose_angle)
        svs_trans = self.svs_tf.p
        svs_trans = vehicle_rpy * svs_trans
        svs_data = svs_data + svs_trans
        self.vehicle.pose_position[2] = svs_data[2]
        self.makePrediction()
        self.vehicle.svsUpdate(self.vehicle.pose_position[2])


    def updateImu(self, imu):
        config = self.config
        config.imu_last_update = rospy.Time.now()
        config.imu_init = True
        
        pose_angle = tf.transformations.euler_from_quaternion(
                                       [imu.orientation.x, imu.orientation.y, 
                                       imu.orientation.z, imu.orientation.w])
        imu_data =  PyKDL.Rotation.RPY(*pose_angle)
        imu_data = config.imu_tf.M * imu_data
        pose_angle = imu_data.GetRPY()
        if not config.imu_data :
            config.last_imu_orientation = pose_angle
            config.last_imu_update = imu.header.stamp
            config.imu_data = True
            
        else:
            pose_angle_quaternion = tf.transformations.quaternion_from_euler(*pose_angle)
            self.vehicle.pose_orientation = pose_angle_quaternion
            
            # Derive angular velocities from orientations ####################### 
            period = (imu.header.stamp - config.last_imu_update).to_sec()
            
            self.vehicle.twist_angular = normalizeAngle(pose_angle-config.last_imu_orientation)/period
            
            self.last_imu_orientation = pose_angle
            self.last_imu_update = imu.header.stamp          
            #####################################################################
            
            self.makePrediction()

        
    def makePrediction(self):
        config = self.config
        if not config.init:
            if config.imu_data and config.gps_data:
                time_now = rospy.Time.now()
                config.last_prediction = time_now
                self.slam_worker.last_odo_predict_time = time_now.to_sec()
                # Initialise slam worker with north and east co-ordinates
                init = lambda:0
                init.north = config.init_north
                init.east = config.east
                self.slam_worker.reset_states()
                config.setNavigation(init)
                config.init = True
            return False
        else:
            pose_angle = tf.transformations.euler_from_quaternion(*self.vehicle.pose_orientation)
            time_now = rospy.Time.now()
            config.last_prediction = time_now
            time_now = time_now.to_sec()
            self.slam_worker.predict_state(pose_angle, time_now)
            return True
            


if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('phdslam')
        g500_slam = G500_SLAM(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException: pass
