# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:39:16 2012

@author: snagappa
"""

import sys
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
from matplotlib.axes import Subplot 
from matplotlib.backends.backend_gtkagg import FigureCanvasGTK
from matplotlib import cm # colormap
from matplotlib import pylab
#pylab.hold(False) # This will avoid memory leak

import roslib
roslib.load_manifest("g500slam")
from nav_msgs.msg import Odometry
from control_g500.srv import GotoSrv, GotoSrvRequest
import rospy
import tf
from sensor_msgs.msg import PointCloud2
#import pc2wrapper
import pointclouds
import girona500

import threading
import numpy as np
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
        
        # List of landmarks and waypoints
        self.scene = STRUCT()
        self.scene.landmarks = []
        self.scene.waypoints = []
        
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
        
        self.viewer.NE_spinbutton = STRUCT()
        self.viewer.NE_spinbutton.east = 0.0
        self.viewer.NE_spinbutton.north = 0.0
        
        self.viewer.size.width = 10
        self.viewer.size.height = 10
        
        self.viewer.DAMAGE = True
        self.viewer.LOCK = threading.Lock()
        self.viewer.DRAW_COUNT = 0
        self.viewer.DRAW_CANVAS = True
        
        # Set up GUI
        self.gladefile = "glade/sim_gui.xml"
        self.glade = gtk.Builder()
        self.glade.add_from_file(self.gladefile)
        
        self.viewer.textview.vehicle_xyz = self.glade.get_object("vehicle_xyz")
        self.viewer.textview.vehicle_rpy = self.glade.get_object("vehicle_rpy")
        fontdesc = pango.FontDescription("monospace 10")
        self.viewer.textview.vehicle_xyz.modify_font(fontdesc)
        self.viewer.textview.vehicle_rpy.modify_font(fontdesc)

        self.glade.connect_signals(self)
        
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
        self.timers.update_image = self.viewer.figure.canvas.new_timer(interval=200)
        self.timers.update_image.add_callback(self.draw)
        self.timers.update_image.start()
        self.timers.publisher = self.viewer.figure.canvas.new_timer(interval=1000)
        self.timers.publisher.add_callback(self.publish_visible)
        self.timers.publisher.start()
        self.pcl_msg = PointCloud2()
        
        self.glade.get_object("MainWindow").show_all()
        
    def init_ros(self):
        self.name = "slamsim"
        rospy.init_node(self.name)
        # Create Subscriber
        rospy.Subscriber("/uwsim/girona500_odom", Odometry, self.update_position)
        # Create Publisher
        self.pcl_publisher = rospy.Publisher("/slamsim/features", PointCloud2)
        
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
        self.viewer.NE_spinbutton.north = widget.get_value()
        p = [self.viewer.NE_spinbutton.east, self.viewer.NE_spinbutton.north]
        print "new point set to ", str(p)
    
    def set_spinbutton_east(self, widget):
        self.viewer.NE_spinbutton.east = widget.get_value()
        p = [self.viewer.NE_spinbutton.east, self.viewer.NE_spinbutton.north]
        print "new point set to ", str(p)
        
    def set_spinbutton_viewer_width(self, widget):
        self.viewer.size.width = widget.get_value()
        p = [self.viewer.size.width, self.viewer.size.height]
        print "set new viewer size to ", str(p)
        self.set_damage()
        
    def set_spinbutton_viewer_height(self, widget):
        self.viewer.size.height = widget.get_value()
        p = [self.viewer.size.width, self.viewer.size.height]
        print "set new viewer size to ", str(p)
        self.set_damage()
        
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
        NE_point = [self.viewer.NE_spinbutton.north, self.viewer.NE_spinbutton.east]
        self.add_point(*NE_point)
        print "added new point at (N,E) : ", str(NE_point)
        self.set_damage()
        
    def add_cursor_point(self, event):
        if event.inaxes != self.viewer.axis: return
        NE_point = [event.ydata, event.xdata]
        self.add_point(*NE_point)
        print "added new point at (N,E) : ", str(NE_point)
        self.set_damage()
        
    def add_point(self, north, east):
        point = [east, north]
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
            rospy.wait_for_service("/control_g500/goto", timeout=5)
        except rospy.ROSException:
            print "Could not execute path"
            return
        try:
            goto_wp = rospy.ServiceProxy("/control_g500/goto", GotoSrv)
            waypoints = self.scene.waypoints
            waypoint_index = 0
            
            while waypoint_index < len(waypoints):
                this_wp = waypoints[waypoint_index]
                if self.simulator.ABORT:
                    self.simulator.RUNNING = False
                    self.simulator.ABORT = False
                    return
                goto_wp_req = GotoSrvRequest()
                goto_wp_req.north = this_wp[1]
                goto_wp_req.east = this_wp[0]
                response = goto_wp(goto_wp_req)
                print response
        except rospy.ROSException:
            self.simulator.RUNNING = False
            self.simulator.ABORT = False
        self.simulator.RUNNING = False
        
    def stop_sim(self, widget):
        self.simulator.ABORT = True
        print "Simulation will end when vehicle reaches next waypoint..."
    
    def update_position(self, odom):
        print "updating position"
        # received a new position, update the viewer
        position = np.array([odom.pose.pose.position.x,
                             odom.pose.pose.position.y,
                             odom.pose.pose.position.z])
        self.vehicle.north_east_depth = position
        
        euler_from_quaternion = tf.transformations.euler_from_quaternion
        orientation = euler_from_quaternion([odom.pose.pose.orientation.x,
                                             odom.pose.pose.orientation.y,
                                             odom.pose.pose.orientation.z,
                                             odom.pose.pose.orientation.w])
        self.vehicle.roll_pitch_yaw = np.array(orientation)
        self.print_position()
        
    def print_position(self):
        north, east, depth = np.round(self.vehicle.north_east_depth, 2)
        text_xyz = ("north : " + "%.2f" % north + "\n" +
                    "east  : " + "%.2f" % east + "\n" +
                    "depth : " + "%.2f" % depth)
        roll, pitch, yaw = np.round(self.vehicle.roll_pitch_yaw, 2)
        text_rpy = ("roll  : " + "%.2f" % roll + "\n" +
                    "pitch : " + "%.2f" % pitch + "\n" +
                    "yaw   : " + "%.2f" % yaw)
        
        self.viewer.textview.vehicle_xyz.get_buffer().set_text(text_xyz)
        self.viewer.textview.vehicle_rpy.get_buffer().set_text(text_rpy)
        
    def publish_visible(self, *args, **kwargs):
        pass
    
    def set_damage(self):
        self.viewer.LOCK.acquire()
        self.viewer.DAMAGE = True
        self.viewer.LOCK.release()
        
    def toggle_canvas_draw(self, widget):
        self.viewer.DRAW_CANVAS = not self.viewer.DRAW_CANVAS
        
    def draw_vehicle(self):
        yaw = self.vehicle.roll_pitch_yaw[2]
        north, east, down = self.vehicle.north_east_depth
        arrow_width = 0.015*self.viewer.size.width
        arrow_length = 0.05*self.viewer.size.height
        self.viewer.axis.arrow(east, north, arrow_length*np.sin(yaw), 
                               arrow_length*np.cos(yaw), width=arrow_width,
                               length_includes_head=True)
    
    def draw(self, *args, **kwargs):
        if not self.viewer.DRAW_CANVAS:
            self.viewer.canvas.draw_idle()
            return
        self.viewer.LOCK.acquire()
        if not self.viewer.DAMAGE and self.viewer.DRAW_COUNT < 10:
            self.viewer.DRAW_COUNT += 1
            self.viewer.LOCK.release()
            return
        
        self.viewer.figure.sca(self.viewer.axis)
        axis = self.viewer.axis
        axis.cla()
        axis.set_title("Click to create landmarks and set waypoints")
        
        points = np.array(self.scene.waypoints)
        if points.shape[0]:
            axis.plot(points[:,0], points[:,1], '-x')
        
        points = np.array(self.scene.landmarks)
        if points.shape[0]:
            axis.scatter(points[:,0], points[:,1])
        
        self.draw_vehicle()
        
        xlim = self.viewer.size.width*np.array([-1, 1])
        ylim = self.viewer.size.height*np.array([-1, 1])
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
