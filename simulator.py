# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:49:04 2012

@author: snagappa
"""

import matplotlib
import matplotlib.pyplot
mpl = matplotlib
import matplotlib.nxutils as nxutils
import numpy as np
import copy
import scipy.interpolate

import roslib
roslib.load_manifest("g500slam")
from nav_msgs.msg import Odometry
import rospy
import tf
from sensor_msgs.msg import PointCloud2
import pc2wrapper

class STRUCT(object): pass

class SLAM_MAP(object):
    def __init__(self):
        # List of landmarks
        self._landmarks_ = []
        # List of vehicle waypoints
        self._waypoints_ = []
        self._splinepath_ = np.zeros(0)
        # Alias for the current list
        self._current_list_ = None
        # Stores the current mode
        self.mode = None
        # Variables for undo/redo
        self.__last_point__ = None
        self.__last_mode__ = None
        
        
    def _setmode_landmarks_(self):
        self._current_list_ = self._landmarks_
        self.mode = "landmarks"
        
    def _setmode_waypoints_(self):
        self._current_list_ = self._waypoints_
        self.mode = "waypoints"
        
    def setmode(self, label):
        if label == "landmarks":
            self._setmode_landmarks_()
        elif label == "waypoints":
            self._setmode_waypoints_()
        
    def addpoint(self, event):
        point = [event.xdata, event.ydata]
        self._current_list_.append(point)
        self.draw()
        
    def undo(self, event):
        if len(self._current_list_):
            self.__last_point__ = self._current_list_.pop()
            self.__last_mode__ = self.mode
        self.draw()
        
    def redo(self, event):
        if not self.__last_point__ == None:
            if self.mode == self.__last_mode__:
                self._current_list_.append(self.__last_point__)
                self.__last_point__ = None
        self.draw()
    
    def draw(self, *args):
        pass
        
    def landmarks(self):
        return np.array(self._landmarks_)
        
    def waypoints(self):
        return np.array(self._waypoints_)
        
    def splinepath(self):
        wp = self.waypoints()
        if wp.shape[0] < 2: return
        x = wp[:,0]
        y = wp[:,1]
        distance = (np.diff(wp, 1, 0)**2).sum(axis=1).sum()
        num_interp_points = np.ceil(distance/0.5)
        k=np.min([3, wp.shape[0]-1])
        tck,u = scipy.interpolate.splprep([x,y],s=0, k=k)
        unew = np.arange(0,1+1/num_interp_points,1/num_interp_points)
        out = scipy.interpolate.splev(unew,tck)
        self._splinepath_ = np.array(zip(out[0], out[1]))
        return self._splinepath_

    def save_scene(self, event):
        Scene._landmarks_ = copy.deepcopy(self._landmarks_)
        Scene._waypoints_ = copy.deepcopy(self._waypoints_)
        
Scene = SLAM_MAP()


class SLAM_MAP_BUILDER(SLAM_MAP):
    def __init__(self):
        self.fig = mpl.pyplot.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.2)
        self.points_collection = self.ax.scatter(np.empty(0), np.empty(0))
        self._cid_ = self.points_collection.figure.canvas.mpl_connect(
                                        'button_press_event', self.addpoint)
        super(SLAM_MAP_BUILDER, self).__init__()
        self.xlim = [-10, 10]
        self.ylim = [-10, 10]
        
        # Undo button
        self.buttons = STRUCT()
        self.buttons.undo = STRUCT()
        self.buttons.redo = STRUCT()
        self.buttons.save = STRUCT()
        self.buttons.radio = STRUCT()
        
        # Undo button
        self.buttons.undo.ax = mpl.pyplot.axes([0.7, 0.05, 0.1, 0.075])
        self.buttons.undo.object = mpl.widgets.Button(self.buttons.undo.ax, 'Undo')
        self.buttons.undo.object.on_clicked(self.undo)
        
        # Redo button
        self.buttons.redo.ax = mpl.pyplot.axes([0.81, 0.05, 0.1, 0.075])
        self.buttons.redo.object = mpl.widgets.Button(self.buttons.redo.ax, 'Redo')
        self.buttons.redo.object.on_clicked(self.redo)
        
        # Save button
        self.buttons.save.ax = mpl.pyplot.axes([0.51, 0.05, 0.1, 0.075])
        self.buttons.save.object = mpl.widgets.Button(self.buttons.save.ax, 'Save')
        self.buttons.save.object.on_clicked(self.save_scene)
        
        # Radio buttons
        self.buttons.radio.axcolor = 'lightgoldenrodyellow'
        self.buttons.radio.ax = mpl.pyplot.axes([0.1, 0.025, 0.2, 0.1], 
                                            axisbg=self.buttons.radio.axcolor)
        self.buttons.radio.object = mpl.widgets.RadioButtons(self.buttons.radio.ax, 
                                                             ('landmarks', 'waypoints'))
        self.buttons.radio.object.on_clicked(self.setmode)
        self.setmode("landmarks")
        
        self.cursor = mpl.widgets.Cursor(self.ax, useblit=True, color='red', linewidth=2 )
        
        self.fig.sca(self.ax)
        #self.ax.set_axes(self.points_collection.axes)
        self.draw()
        #mpl.pyplot.show()
        
        
    def addpoint(self, event):
        if event.inaxes != self.points_collection.axes: return
        super(SLAM_MAP_BUILDER, self).addpoint(event)
        
    def draw(self, *args, **kwargs):
        self.fig.sca(self.ax)
        self.ax.cla()
        self.ax.set_title("Click to create landmarks and set waypoints")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        
        points = self.landmarks()
        if points.shape[0]:
            mpl.pyplot.scatter(points[:,0], points[:,1])
        
        points = self.waypoints()
        if points.shape[0]:
            #line = self.ax.plot(points[:,0], points[:,1])
            mpl.pyplot.plot(points[:,0], points[:,1], '-x')
            wp_spline = self.splinepath()
            if not wp_spline==None:
                mpl.pyplot.plot(wp_spline[:,0], wp_spline[:,1])
        self.points_collection.figure.canvas.draw()
        


class SLAM_SIMULATOR(object):
    def __init__(self, name="slamsim"):
        self.name = name
        self.map_builder = SLAM_MAP_BUILDER()
        self.vehicle = STRUCT()
        self.vehicle.position = np.zeros(3)
        self.vehicle.orientation = np.zeros(3)
        self.viewer = self.init_viewer()
        self.init_ros(name)
        self.last_update = rospy.Time.now()
        self.timers = STRUCT()
        self.timers.update_image = self.viewer.fig.canvas.new_timer(interval=50)
        self.timers.update_image.add_callback(self.draw)
        self.timers.update_image.start()
        self.timers.publisher = self.viewer.fig.canvas.new_timer(interval=1000)
        self.timers.publisher.add_callback(self.publish_visible)
        self.timers.publisher.start()
        self.pcl_msg = PointCloud2()
        mpl.pyplot.show()
        
    def init_ros(self, name):
        #rospy.init_node(name)
        # Create Subscriber
        rospy.Subscriber("/uwsim/girona500_odom", Odometry, self.update_position)
        # Create Publisher
        self.pcl_publisher = rospy.Publisher("/slamsim/features", PointCloud2)
        
    def init_viewer(self):
        viewer = STRUCT()
        viewer.fig = mpl.pyplot.figure()
        viewer.ax = viewer.fig.add_subplot(111)
        viewer.xlim = [-10, 10]
        viewer.ylim = [-10, 10]
        return viewer
    
    def update_position(self, odom):
        # received a new position, update the viewer
        position = np.array([odom.pose.pose.position.x,
                             odom.pose.pose.position.y,
                             odom.pose.pose.position.z])
        self.vehicle.position = position
        
        euler_from_quaternion = tf.transformations.euler_from_quaternion
        orientation = euler_from_quaternion([odom.pose.pose.orientation.x,
                                             odom.pose.pose.orientation.y,
                                             odom.pose.pose.orientation.z,
                                             odom.pose.pose.orientation.w])
        self.vehicle.orientation = orientation
        
    def visible_landmarks(self, width=2.0, length=1.0):
        x = self.vehicle.position[1]
        y = self.vehicle.position[0]
        orientation = self.vehicle.orientation[2]
        
        delta_x = width/2*np.cos(np.pi-(np.pi/2-orientation)-np.pi/2)
        delta_y = width/2*np.sin(np.pi-(np.pi/2-orientation)-np.pi/2)
        extra_x = length*np.cos(np.pi-orientation-np.pi/2)
        extra_y = length*np.sin(np.pi-orientation-np.pi/2)
        """
        vertices = np.array([[x-width/2, y],
                             [x-width/2, y+length/2],
                             [x+width/2, y+length/2],
                             [x+width/2, y]])
        """
        vertices = np.array([[x+delta_x, y-delta_y], 
                             [x-delta_x, y+delta_y],
                             [x+extra_x-delta_x, y+extra_y+delta_y],
                             [x+extra_x+delta_x, y+extra_y-delta_y]])
        
        landmarks = self.map_builder.landmarks()
        if landmarks.shape[0]:
            landmarks_mask = nxutils.points_inside_poly(landmarks, vertices)
            vis_landmarks = landmarks[landmarks_mask]
        else:
            vis_landmarks = np.empty(0)
        return vis_landmarks, vertices
        
    def draw(self, *args, **kwargs):
        viewer = self.viewer
        viewer.ax.clear()
        viewer.ax.cla()
        viewer.ax.set_title("Vehicle position")
        viewer.ax.set_xlim(self.viewer.xlim)
        viewer.ax.set_ylim(self.viewer.ylim)
        yaw = self.vehicle.orientation[2]
        position = self.vehicle.position
        viewer.ax.arrow(position[1], position[0], 
                        0.5*np.sin(yaw), 0.5*np.cos(yaw), width=0.15,
                        length_includes_head=True)
        
        points = self.map_builder.landmarks()
        if points.shape[0]:
            viewer.ax.scatter(points[:,0], points[:,1], c='r')
        landmarks, vertices = self.visible_landmarks()
        
        #mpl.patches.Polygon(vertices)
        vertices = np.vstack((vertices, vertices[0]))
        viewer.ax.plot(vertices[:,0], vertices[:,1])
        if landmarks.shape[0]:
            viewer.ax.scatter(landmarks[:,0], landmarks[:,1])
        viewer.fig.canvas.draw()
    
    def publish_visible(self, *args, **kwargs):
        features = self.visible_landmarks()[0]
        if not features.shape[0]: return
        x = self.vehicle.position[1]
        y = self.vehicle.position[0]
        relative_position = features - np.array([x, y])
        yaw = self.vehicle.orientation[2]
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])
        relative_position = np.dot(rotation_matrix, relative_position.T).T
        relative_position = np.hstack((relative_position, np.zeros((relative_position.shape[0], 1))))
        # Convert to a pointcloud and publish
        pcl_msg = self.pcl_msg
        pcl_msg.header.stamp = rospy.Time.now()
        pcl_msg.header.frame_id = "slamsim"
        pcl_msg = pc2wrapper.create_cloud_xyz32(pcl_msg.header, relative_position)
        self.pcl_publisher.publish(pcl_msg)
        
if __name__ == '__main__':
    try:
        #   Init node
        rospy.init_node('slamsim')
        slamsim = SLAM_SIMULATOR(rospy.get_name())
        mpl.pyplot.show()
    except rospy.ROSInterruptException: pass