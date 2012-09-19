# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:39:16 2012

@author: snagappa
"""

import sys
import os
try:  
    import pygtk  
    pygtk.require("2.0")  
except:  
    pass  
try:  
    import gtk  
    import gtk.glade  
except:  
    print("GTK Not Availible")
    sys.exit(1)
import pango

import matplotlib 
mpl = matplotlib
matplotlib.use('Agg') 
from matplotlib.figure import Figure 
#from matplotlib.axes import Subplot 
from matplotlib.backends.backend_gtkagg import FigureCanvasGTK
#from matplotlib import cm # colormap
#from matplotlib import pylab
import matplotlib.nxutils as nxutils

#pylab.hold(False) # This will avoid memory leak

import roslib
roslib.load_manifest("udg_pandora")
from nav_msgs.msg import Odometry
from auv_msgs.msg import NavSts
from cola2_control.srv import GotoSrv, GotoSrvRequest
import rospy
import tf
from sensor_msgs.msg import PointCloud2
import featuredetector
from lib.common import misctools
import threading
import numpy as np
import copy
import code


class STRUCT(object): pass

LANDMARKS = "landmarks"
WAYPOINTS = "waypoints"

class gtk_slam_sim:
    def __init__(self):
        # True vehicle position
        self.vehicle = STRUCT()
        self.vehicle.north_east_depth = np.zeros(3)
        self.vehicle.roll_pitch_yaw = np.zeros(3)
        self.vehicle.fov = STRUCT()
        self.vehicle.fov.width = 1.5
        self.vehicle.fov.depth = 3.0
        self.vehicle.fov.x_deg = 60
        self.vehicle.fov.y_deg = 45
        self.vehicle.fov.far_m = 3
        self.vehicle.sensor_fov = featuredetector.sensors.camera_fov(self.vehicle.fov.x_deg, 
                                                                     self.vehicle.fov.y_deg, 
                                                                     self.vehicle.fov.far_m)
        self.vehicle.visible_landmarks = STRUCT()
        self.vehicle.visible_landmarks.abs = np.empty(0)
        self.vehicle.visible_landmarks.rel = np.empty(0)
        self.vehicle.visible_landmarks.noise = np.empty(0)
        self.vehicle.visible_landmarks.fov_poly_vertices = np.empty(0)
        self.vehicle.LOCK = threading.Lock()
        
        # Container for estimated vehicle position and landmarks
        self.estimator = STRUCT()
        self.estimator.north_east_depth = np.zeros(3)
        self.estimator.roll_pitch_yaw = np.zeros(3)
        self.estimator.fov_poly_vertices = np.empty(0)
        self.estimator.landmarks = np.empty(0)
        
        # List of landmarks and waypoints
        self.scene = STRUCT()
        self.scene.landmarks = []
        self.scene.waypoints = []
        self.scene.clutter = 2
        
        # variables to enable undo/redo
        self.scene.mode = LANDMARKS
        self.scene.__last_point__ = None
        self.scene.__last_mode__ = None
        
        # Pointer to whatever is currently being modified
        self.scene.__current_list__ = self.scene.landmarks
        
        # Control variables for the simulator
        self.simulator = STRUCT()
        self.simulator.RUNNING = False
        self.simulator.ABORT = False
        
        self.viewer = STRUCT()
        self.viewer.size = STRUCT()
        self.viewer.textview = STRUCT()
        
        self.viewer.NED_spinbutton = STRUCT()
        self.viewer.NED_spinbutton.east = 0.0
        self.viewer.NED_spinbutton.north = 0.0
        self.viewer.NED_spinbutton.depth = 0.0
        self.viewer.NED_spinbutton.noise = 0.03
        
        self.viewer.size.width = 10
        self.viewer.size.height = 10
        
        self.viewer.DAMAGE = True
        self.viewer.LOCK = threading.Lock()
        self.viewer.DRAW_COUNT = 0
        self.viewer.DRAW_CANVAS = True
        
        # Set up GUI
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        self.gladefile = os.path.join(pkg_dir, "glade", "sim_gui.xml")
        self.glade = gtk.Builder()
        self.glade.add_from_file(self.gladefile)
        
        self.viewer.textview.vehicle_xyz = self.glade.get_object("vehicle_xyz")
        self.viewer.textview.vehicle_rpy = self.glade.get_object("vehicle_rpy")
        fontdesc = pango.FontDescription("monospace 10")
        self.viewer.textview.vehicle_xyz.modify_font(fontdesc)
        self.viewer.textview.vehicle_rpy.modify_font(fontdesc)

        self.glade.connect_signals(self)
        
        self.ros = STRUCT()
        self.init_ros()
        self.viewer.figure = Figure(figsize=(512, 512), dpi=75)
        self.viewer.axis = self.viewer.figure.add_subplot(111) 
        self.viewer.canvas = FigureCanvasGTK(self.viewer.figure) # a gtk.DrawingArea
        self.viewer.canvas.show() 
        self.viewer.graphview = self.glade.get_object("viewer_drawing_box")
        self.viewer.graphview.pack_start(self.viewer.canvas, True, True)
        
        self.viewer.cursor = mpl.widgets.Cursor(self.viewer.axis, useblit=False, color='red', linewidth=2 )
        self.viewer._cid_ = self.viewer.figure.canvas.mpl_connect(
                                        'button_press_event', self.add_cursor_point)
        
        self.print_position()
        
        self.timers = STRUCT()
        # Drawing timer
        self.timers.update_image = self.viewer.figure.canvas.new_timer(interval=50)
        self.timers.update_image.add_callback(self.draw)
        self.timers.update_image.start()
        # Update visible landmarks
        self.timers.update_visible_landmarks = self.viewer.figure.canvas.new_timer(interval=100)
        self.timers.update_visible_landmarks.add_callback(self.update_visible_landmarks)
        self.timers.update_visible_landmarks.start()
        # ROS publisher
        self.timers.publisher = self.viewer.figure.canvas.new_timer(interval=1000)
        self.timers.publisher.add_callback(self.publish_visible_landmarks)
        self.timers.publisher.start()
        
        self.glade.get_object("MainWindow").show_all()
        
    def init_ros(self):
        self.ros.name = "slamsim"
        rospy.init_node(self.ros.name)
        # Create Subscriber
        rospy.Subscriber("/dataNavigator", Odometry, self.update_position)
        # Create Publisher
        self.ros.pcl_publisher = rospy.Publisher("/slamsim/features", PointCloud2)
        self.ros.pcl_helper = featuredetector.msgs.msgs(featuredetector.msgs.MSG_XYZ_COV)
        #field_name = ['x', 'y', 'z', 'sigma_x', 'sigma_y', 'sigma_z']
        #field_offset = range(0, 24, 4)
        #self.ros.pcl_fields = [PointField(_field_name_, _field_offset_, 
        #                                  PointField.FLOAT32, 1) for 
        #                        (_field_name_, _field_offset_) in zip(field_name, field_offset)]
        #self.ros.pcl_header = PointCloud2().header
        
        # Subscribe to vehicle NavSts as well as the published estimated landmarks
        rospy.Subscriber("/phdslam/nav_sts", NavSts, self.estimator_update_position)
        rospy.Subscriber("/phdslam/features", PointCloud2, self.estimator_update_landmarks)
        
        
    def on_MainWindow_delete_event(self, widget, event):
            gtk.main_quit()
    
    def set_mode_landmarks(self, widget):
        self.scene.__current_list__ = self.scene.landmarks
        self.scene.mode = LANDMARKS
        print "set mode to landmarks"
        
    def set_mode_waypoints(self, widget):
        self.scene.__current_list__ = self.scene.waypoints
        self.scene.mode = WAYPOINTS
        print "set mode to waypoints"
        
    def set_spinbutton_north(self, widget):
        self.viewer.NED_spinbutton.north = widget.get_value()
        p = [self.viewer.NED_spinbutton.north, 
             self.viewer.NED_spinbutton.east, 
             self.viewer.NED_spinbutton.depth]
        print "new point set to (N,E,D)", str(p)
    
    def set_spinbutton_east(self, widget):
        self.viewer.NED_spinbutton.east = widget.get_value()
        p = [self.viewer.NED_spinbutton.north, 
             self.viewer.NED_spinbutton.east, 
             self.viewer.NED_spinbutton.depth]
        print "new point set to (N,E,D)", str(p)
        
    def set_spinbutton_depth(self, widget):
        self.viewer.NED_spinbutton.depth = widget.get_value()
        p = [self.viewer.NED_spinbutton.north, 
             self.viewer.NED_spinbutton.east, 
             self.viewer.NED_spinbutton.depth]
        print "new point set to (N,E,D)", str(p)
        
    def set_spinbutton_noise(self, widget):
        self.viewer.NED_spinbutton.noise = widget.get_value()
        p = self.viewer.NED_spinbutton.noise
        print "new observation noise set to ", str(p)
        
    def set_spinbutton_viewer_width(self, widget):
        self.viewer.size.width = widget.get_value()
        self.viewer.size.height = self.viewer.size.width
        p = [self.viewer.size.width, self.viewer.size.height]
        print "set new viewer size to ", str(p)
        self.set_damage()
        
    def set_spinbutton_viewer_height(self, widget):
        self.viewer.size.height = widget.get_value()
        p = [self.viewer.size.width, self.viewer.size.height]
        print "set new viewer size to ", str(p)
        self.set_damage()
        
    def set_spinbutton_fov_width(self, widget):
        self.vehicle.fov.width = widget.get_value()
        self.set_damage()
        
    def set_spinbutton_fov_depth(self, widget):
        self.vehicle.fov.depth = widget.get_value()
        self.set_damage()
        
    def set_spinbutton_fov_x(self, widget):
        self.vehicle.fov.x_deg = widget.get_value()
        self.set_sensor_fov()
        self.set_damage()
        
    def set_spinbutton_fov_y(self, widget):
        self.vehicle.fov.y_deg = widget.get_value()
        self.set_sensor_fov()
        self.set_damage()
    
    def set_spinbutton_fov_far(self, widget):
        self.vehicle.fov.far_m = widget.get_value()
        self.set_sensor_fov()
        self.set_damage()
        
    def set_sensor_fov(self):
        self.vehicle.sensor_fov.set_x_y_far(self.vehicle.fov.x_deg, 
                                            self.vehicle.fov.y_deg, 
                                            self.vehicle.fov.far_m)
    
    def set_spinbutton_clutter(self, widget):
        self.scene.clutter = widget.get_value()
        print "Set number of FOV clutter points to ", str(self.scene.clutter)
    
    def undo(self, widget):
        if len(self.scene.__current_list__):
            self.scene.__last_point__ = self.scene.__current_list__.pop()
            self.scene.__last_mode__ = self.scene.mode
            print "popped point at : ", str(self.scene.__last_point__)
            self.set_damage()
        
    def redo(self, widget):
        if not self.scene.__last_point__ == None:
            if self.scene.mode == self.scene.__last_mode__:
                self.scene.__current_list__.append(self.scene.__last_point__)
                print "pushed point at : ", str(self.scene.__last_point__)
                self.scene.__last_point__ = None
                self.set_damage()
                
    def pop(self, widget):
        if len(self.scene.__current_list__):
            pop_point = self.scene.__current_list__.pop(0)
            print "popped point at : ", str(pop_point)
        self.set_damage()
        
    def add_spinbutton_point(self, widget):
        NED_point = [self.viewer.NED_spinbutton.north, 
                     self.viewer.NED_spinbutton.east, 
                     self.viewer.NED_spinbutton.depth]
        self.add_point(*NED_point)
        print "added new point at (N,E,D) : ", str(NED_point)
        self.set_damage()
        
    def add_cursor_point(self, event):
        if event.inaxes != self.viewer.axis: return
        NED_point = [event.ydata, event.xdata, self.viewer.NED_spinbutton.depth]
        self.add_point(*NED_point)
        print "added new point at (N,E,D) : ", str(NED_point)
        self.set_damage()
        
    def add_point(self, north, east, depth):
        point = [north, east, depth]
        self.scene.__current_list__.append(point)
    
    def start_sim(self, widget):
        if self.simulator.RUNNING:
            print "simulator already running..."
            return
        self.simulator.RUNNING = True
        self.simulator.sim_thread = threading.Thread(target=self._start_sim_)
        self.simulator.sim_thread.start()
    
    def _start_sim_(self):
        try:
            rospy.wait_for_service("/cola2_control/goto", timeout=5)
        except rospy.ROSException:
            print "Could not execute path"
            return
        try:
            goto_wp = rospy.ServiceProxy("/cola2_control/goto", GotoSrv)
            waypoints = self.scene.waypoints
            waypoint_index = 0
            
            while waypoint_index < len(waypoints):
                this_wp = waypoints[waypoint_index]
                if self.simulator.ABORT:
                    self.simulator.RUNNING = False
                    self.simulator.ABORT = False
                    return
                goto_wp_req = GotoSrvRequest()
                goto_wp_req.north = this_wp[0]
                goto_wp_req.east = this_wp[1]
                response = goto_wp(goto_wp_req)
                print response
                waypoint_index += 1
        except rospy.ROSException:
            self.simulator.RUNNING = False
            self.simulator.ABORT = False
        self.simulator.RUNNING = False
        
    def stop_sim(self, widget):
        self.simulator.ABORT = True
        print "Simulation will end when vehicle reaches next waypoint..."
    
    def update_position(self, odom):
        #print "updating position"
        # received a new position, update the viewer
        position = np.array([odom.pose.pose.position.x,
                             odom.pose.pose.position.y,
                             odom.pose.pose.position.z])
        
        euler_from_quaternion = tf.transformations.euler_from_quaternion
        orientation = euler_from_quaternion([odom.pose.pose.orientation.x,
                                             odom.pose.pose.orientation.y,
                                             odom.pose.pose.orientation.z,
                                             odom.pose.pose.orientation.w])
        self.vehicle.LOCK.acquire()
        try:
            self.vehicle.north_east_depth = position
            self.vehicle.roll_pitch_yaw = np.array(orientation)
        finally:
            self.vehicle.LOCK.release()
        self.print_position()
        self.update_visible_landmarks()
        
    def estimator_update_position(self, nav):
        #print "updating position"
        # received a new position, update the viewer
        position = np.array([nav.position.north,
                             nav.position.east,
                             nav.position.depth])
        self.estimator.north_east_depth = position
        
        orientation = np.array([nav.orientation.roll, 
                                nav.orientation.pitch, 
                                nav.orientation.yaw])
        self.estimator.roll_pitch_yaw = orientation
        self.print_position()
        
        # Update field of view
        vertices = self.vehicle.sensor_fov.fov_vertices_2d()
        # Rotate by the yaw
        cy = np.cos(self.estimator.roll_pitch_yaw[2])
        sy = np.sin(self.estimator.roll_pitch_yaw[2])
        vertices = np.dot(np.array([[cy, sy], [-sy, cy]]), vertices.T).T
        # Translation to vehicle position
        north, east, depth = position
        vertices += np.array([east, north])
        self.estimator.fov_poly_vertices = vertices
        
    def print_position(self):
        north, east, depth = np.round(self.vehicle.north_east_depth, 2)
        e_north, e_east, e_depth = np.round(self.estimator.north_east_depth, 2)
        text_xyz = ("True:\n" +
                    "north : " + "%.2f" % north + "\n" +
                    "east  : " + "%.2f" % east + "\n" +
                    "depth : " + "%.2f" % depth + "\n" + "\n\n"
                    "Estimated:\n" + 
                    "north : " + "%.2f" % e_north + "\n" +
                    "east  : " + "%.2f" % e_east + "\n" +
                    "depth : " + "%.2f" % e_depth)
        roll, pitch, yaw = np.round(self.vehicle.roll_pitch_yaw, 2)
        e_roll, e_pitch, e_yaw = np.round(self.estimator.roll_pitch_yaw, 2)
        text_rpy = ("True:\n" +
                    "roll  : " + "%.2f" % roll + "\n" +
                    "pitch : " + "%.2f" % pitch + "\n" +
                    "yaw   : " + "%.2f" % yaw + "\n" + "\n\n" +
                    "Estimated:\n" +
                    "roll  : " + "%.2f" % e_roll + "\n" +
                    "pitch : " + "%.2f" % e_pitch + "\n" +
                    "yaw   : " + "%.2f" % e_yaw)
        
        self.viewer.textview.vehicle_xyz.get_buffer().set_text(text_xyz)
        self.viewer.textview.vehicle_rpy.get_buffer().set_text(text_rpy)
    
    def update_visible_landmarks(self, *args, **kwargs):
        if self.vehicle.LOCK.locked():
            return
        self.vehicle.LOCK.acquire()
        try:
            landmarks = np.array(self.scene.landmarks)
            relative_landmarks = featuredetector.tf.relative(self.vehicle.north_east_depth, 
                                                             self.vehicle.roll_pitch_yaw, 
                                                             landmarks)
            visible_landmarks_idx = self.vehicle.sensor_fov.is_visible(relative_landmarks)
            self.vehicle.visible_landmarks.abs = landmarks[visible_landmarks_idx]
            self.vehicle.visible_landmarks.rel = relative_landmarks[visible_landmarks_idx]
            if self.vehicle.visible_landmarks.rel.shape[0]:
                self.vehicle.visible_landmarks.noise = (
                    np.sqrt((self.vehicle.visible_landmarks.rel**2).sum(axis=1))*
                    self.viewer.NED_spinbutton.noise)
                self.vehicle.visible_landmarks.noise[self.vehicle.visible_landmarks.noise<0.001] = 0.001
            else:
                self.vehicle.visible_landmarks.noise = np.empty(0)
            """
            if self.vehicle.visible_landmarks.rel.shape[0]:
                inverse_landmarks = featuredetector.tf.absolute(self.vehicle.north_east_depth, 
                                                                self.vehicle.roll_pitch_yaw, 
                                                                self.vehicle.visible_landmarks.rel)
            
                if not (np.abs(inverse_landmarks - self.vehicle.visible_landmarks.abs) < 1e-4).all():
                    print "Inverse not equal"
                    print "Diff = ", (inverse_landmarks - self.vehicle.visible_landmarks.abs)
                    #code.interact(local=locals())
            """
            # Set vertices
            vertices = self.vehicle.sensor_fov.fov_vertices_2d()
            
            # Rotate by the yaw
            cy = np.cos(self.vehicle.roll_pitch_yaw[2])
            sy = np.sin(self.vehicle.roll_pitch_yaw[2])
            vertices = np.dot(np.array([[cy, sy], [-sy, cy]]), vertices.T).T
            # Translation to vehicle position
            north, east, depth = self.vehicle.north_east_depth
            vertices += np.array([east, north])
            self.vehicle.visible_landmarks.fov_poly_vertices = vertices
            
            """
            north, east, depth = self.vehicle.north_east_depth
            yaw = self.vehicle.roll_pitch_yaw[2]
            vis_width = self.vehicle.fov.width
            vis_depth = self.vehicle.fov.depth
            
            # Set vertices
            vertices = np.array([[-vis_width/2, 0], [-vis_width/2, vis_depth], 
                                 [vis_width/2, vis_depth], [vis_width/2, 0]])
            
            # Rotate by the yaw
            cy = np.cos(yaw)
            sy = np.sin(yaw)
            vertices = np.dot(np.array([[cy, sy], [-sy, cy]]), vertices.T).T
            # Translation to vehicle position
            vertices += np.array([east, north])
            self.vehicle.visible_landmarks.fov_poly_vertices = vertices
            
            # copy visible landmarks to self.vehicle.visible_landmarks.abs
            landmarks = np.array(self.scene.landmarks)
            if landmarks.shape[0]:
                landmarks_mask = nxutils.points_inside_poly(landmarks[:, [1,0]], vertices)
                self.vehicle.visible_landmarks.abs = landmarks[landmarks_mask]
                self.vehicle.visible_landmarks.rel = \
                    featuredetector.tf.relative(np.array([north, east, depth]), 
                                                        self.vehicle.roll_pitch_yaw, 
                                                        self.vehicle.visible_landmarks.abs)
            else:
                self.vehicle.visible_landmarks.abs = np.empty(0)
                self.vehicle.visible_landmarks.rel = np.empty(0)
            # Perform translation/rotation of visible landmarks to
            # self.vehicle.visible_landmarks.rel
            """
        finally:
            self.vehicle.LOCK.release()
        # Publish the landmarks
        ##features = np.array(np.hstack((features, np.zeros((features.shape[0], 1)))), order='C')
        # Convert north-east-z to x-y-z
        ##vehicle_xyz = self.vehicle.position
        ##relative_position = girona500.feature_relative_position(vehicle_xyz, self.vehicle.orientation, features)
        #self.publish_visible_landmarks()
    
    def estimator_update_landmarks(self, pcl_msg):
        self.estimator.landmarks = self.ros.pcl_helper.from_pcl(pcl_msg)
        
    def publish_visible_landmarks(self, *args, **kwargs):
        self.update_visible_landmarks()
        rel_landmarks = self.vehicle.visible_landmarks.rel
        clutter_pts = self.scene.clutter #np.random.poisson(self.scene.clutter)
        if clutter_pts:
            # Draw uniform samples for r, theta, phi
            clutter_r_theta_phi = np.random.rand(3, clutter_pts)
            clutter_r_theta_phi[0] *= (self.vehicle.fov.far_m - 0.3)
            clutter_r_theta_phi[0] += 0.3
            clutter_r_theta_phi[1:3] -= 0.5
            #clutter_r_theta_phi[2,:] = 0
            clutter_r_theta_phi[1:3] *= (np.array([[self.vehicle.fov.x_deg-4], [self.vehicle.fov.y_deg-4]])*np.pi/180)
            clutter_xyz = np.zeros(clutter_r_theta_phi.shape)
            misctools.spherical_to_cartesian(clutter_r_theta_phi, clutter_xyz)
            #clutter_xyz = clutter_xyz.T
            clutter_landmarks = (clutter_xyz[[2, 0, 1]]).T
            clutter_noise = (np.sqrt((clutter_xyz.T**2).sum(axis=1))*
                             self.viewer.NED_spinbutton.noise)
            clutter_noise[clutter_noise<0.001] = 0.001
            clutter_noise[:] = self.viewer.NED_spinbutton.noise
            clutter_landmarks = np.hstack((clutter_landmarks, 
                clutter_noise[:,np.newaxis]*np.ones(clutter_landmarks.shape)))
        else:
            clutter_landmarks = np.empty(0)
        
        detected = np.random.rand(rel_landmarks.shape[0])<0.99
        if detected.shape[0]:
            rel_landmarks = rel_landmarks[detected]
        else:
            rel_landmarks = np.zeros(0)
        if rel_landmarks.shape[0]:
            # Convert xyz to PointCloud message
            #pcl_msg = pointclouds.xyz_array_to_pointcloud2(rel_landmarks, rospy.Time.now(), "slamsim")
            # Convert xyz to PointCloud message with (diagonal) covariance
            noise = self.vehicle.visible_landmarks.noise[detected]
            std_dev = np.sqrt(noise)
            #awgn = std_dev[:,np.newaxis]*np.random.normal(size=rel_landmarks.shape)
            awgn = self.viewer.NED_spinbutton.noise*np.random.normal(size=rel_landmarks.shape)
            rel_landmarks = np.hstack((rel_landmarks+awgn, noise[:,np.newaxis]*np.ones(rel_landmarks.shape)))
            #self.ros.pcl_header.stamp = rospy.Time.now()
            #self.ros.pcl_header.frame_id = "slamsim"
            #pcl_msg = pc2wrapper.create_cloud(self.ros.pcl_header, self.ros.pcl_fields, rel_landmarks)
        #else:
        #    rel_landmarks = np.empty(0)
        
        if rel_landmarks.shape[0] and clutter_landmarks.shape[0]:
            rel_landmarks = np.vstack((rel_landmarks, clutter_landmarks))
        elif clutter_landmarks.shape[0]:
            rel_landmarks = clutter_landmarks
        pcl_msg = self.ros.pcl_helper.to_pcl(rospy.Time.now(), rel_landmarks)
        pcl_msg.header.frame_id = self.ros.name
        # and publish visible landmarks
        self.ros.pcl_publisher.publish(pcl_msg)
        print "Visible Landmarks (abs):"
        print self.vehicle.visible_landmarks.abs
        print "Publishing:"
        print rel_landmarks #self.vehicle.visible_landmarks.rel
    
    def set_damage(self):
        self.viewer.LOCK.acquire()
        self.viewer.DAMAGE = True
        self.viewer.LOCK.release()
        
    def toggle_canvas_draw(self, widget):
        self.viewer.DRAW_CANVAS = not self.viewer.DRAW_CANVAS
        
    def draw_vehicle(self):
        # True position from simulator
        yaw = self.vehicle.roll_pitch_yaw[2]
        north, east, depth = self.vehicle.north_east_depth
        arrow_width = 0.015*self.viewer.size.width
        arrow_length = 0.05*self.viewer.size.height
        self.viewer.axis.arrow(east, north, arrow_length*np.sin(yaw), 
                               arrow_length*np.cos(yaw), width=arrow_width,
                               length_includes_head=True, color='b')
        # Draw fov
        fov_vertices = self.vehicle.visible_landmarks.fov_poly_vertices.copy()
        if fov_vertices.shape[0]:
            fov_vertices = np.vstack((fov_vertices, fov_vertices[0]))
            self.viewer.axis.plot(fov_vertices[:,0], fov_vertices[:,1], c='b')
        
        # Estimation from navigator
        yaw = self.estimator.roll_pitch_yaw[2]
        north, east, depth = self.estimator.north_east_depth
        self.viewer.axis.arrow(east, north, arrow_length*np.sin(yaw), 
                               arrow_length*np.cos(yaw), width=arrow_width,
                               length_includes_head=True, color='r')
        fov_vertices = self.estimator.fov_poly_vertices.copy()
        if fov_vertices.shape[0]:
            fov_vertices = np.vstack((fov_vertices, fov_vertices[0]))
            self.viewer.axis.plot(fov_vertices[:,0], fov_vertices[:,1], c='r')
            
    def draw_visible_landmarks(self):
        # Plot visible landmarks
        points = np.array(self.vehicle.visible_landmarks.abs)
        if points.shape[0]:
            self.viewer.axis.scatter(points[:,1], points[:,0], s=36, marker='o')
        # Plot estimated landmarks
        points = self.estimator.landmarks
        if points.shape[0]:
            self.viewer.axis.scatter(points[:,1], points[:,0], s=36, marker='*', c='g')
        
    def draw(self, *args, **kwargs):
        if not self.viewer.DRAW_CANVAS and self.viewer.DRAW_COUNT < 500:
            #self.viewer.canvas.draw_idle()
            self.viewer.DRAW_COUNT += 1
            return
        self.viewer.LOCK.acquire()
        if not self.viewer.DAMAGE and self.viewer.DRAW_COUNT < 10:
            self.viewer.DRAW_COUNT += 1
            self.viewer.LOCK.release()
            return
        self.viewer.DRAW_COUNT = 0
        self.viewer.figure.sca(self.viewer.axis)
        axis = self.viewer.axis
        axis.cla()
        axis.set_title("Click to create landmarks and set waypoints")
        
        points = np.array(self.scene.waypoints)
        if points.shape[0]:
            axis.plot(points[:,1], points[:,0], '-x')
        
        points = np.array(self.scene.landmarks)
        if points.shape[0]:
            axis.scatter(points[:,1], points[:,0], c='r', s=16, marker='o')
        
        self.draw_vehicle()
        self.draw_visible_landmarks()
        
        xlim = np.round(self.viewer.size.width*np.array([-1, 1]) + self.vehicle.north_east_depth[1])
        ylim = np.round(self.viewer.size.height*np.array([-1, 1]) + self.vehicle.north_east_depth[0])
        if xlim[0] < -100:
            xlim += -100 - xlim[0]
        elif xlim[1] > 100:
            xlim += 100 - xlim[1]
        if ylim[0] < -100:
            ylim += -100 - ylim[0]
        elif ylim[1] > 100:
            ylim += 100 - ylim[1]
        
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        
        self.viewer.canvas.draw_idle()
        self.DAMAGE = False
        self.viewer.LOCK.release()
    
    def quit(self, *args, **kwargs):
        gtk.main_quit()

if __name__ == '__main__':
    try:
        slamsim = gtk_slam_sim()
        gtk.main()
    except KeyboardInterrupt:
        pass
