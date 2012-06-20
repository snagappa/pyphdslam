# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:49:04 2012

@author: snagappa
"""

import matplotlib as mpl
import matplotlib.nxutils as nxutils
import numpy as np
import copy
import scipy.interpolate


class SLAM_POINTS(object):
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
        
        
class STRUCT(object): pass
class SCENARIO(object):
    def __init__(self):
        self.vehicle = STRUCT()
        self.landmarks = STRUCT()
        
    def visible_landmarks(self, x, y, orientation, width=1, length=1):
        delta_x = width/2*np.cos(np.pi/2-orientation)
        delta_y = width/2*np.sin(np.pi/2-orientation)
        extra_x = length*np.cos(orientation)
        extra_y = length*np.sin(orientation)
        vertices = np.array([[x+delta_x, y-delta_y], 
                             [x-delta_x, y+delta_y],
                             [x+extra_x-delta_x, y+extra_y+delta_y],
                             [x+extra_x+delta_x, y+extra_y-delta_y]])
        landmarks_mask = nxutils.points_inside_poly(self.landmarks, vertices)
        return landmarks_mask
        
    def make_observations(self, width=1, length=1):
        mask = []
        observations = []
        for i in range(self.vehicle.orientation.shape[0]):
            this_mask = self.visible_landmarks(self.vehicle.path[i,0], 
                                               self.vehicle.path[i,1],
                                               self.vehicle.orientation[i],
                                               width, length)
            this_obs = self.landmarks[this_mask]
            mask.append(this_mask)
            observations.append(this_obs)
        return observations
        

class SLAM_SCENE(SLAM_POINTS):
    def __init__(self):
        self.fig = mpl.pyplot.figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.2)
        self.points_collection = self.ax.scatter(np.empty(0), np.empty(0))
        self._cid_ = self.points_collection.figure.canvas.mpl_connect(
                                        'button_press_event', self.addpoint)
        super(SLAM_SCENE, self).__init__()
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
        mpl.pyplot.show()
        
        
    def addpoint(self, event):
        if event.inaxes != self.points_collection.axes: return
        super(SLAM_SCENE, self).addpoint(event)
        
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
        
    def scenario(self):
        slam_scenario = SCENARIO()
        slam_scenario.vehicle = STRUCT()
        slam_scenario.vehicle.path = self.splinepath()
        path_delta = np.diff(slam_scenario.vehicle.path, 1, 0)
        orientation = [np.math.atan2(path_delta[i,1], path_delta[i,0]) 
                                        for i in range(path_delta.shape[0])]
        orientation.append(orientation[-1])
        slam_scenario.vehicle.orientation = np.array(orientation)
        slam_scenario.landmarks = self.landmarks()
        return slam_scenario


Scene = SLAM_POINTS()


def visible_landmarks(xy, orientation, width=1, length=1):
    x, y = xy[0], xy[1]
    delta_x = width/2*np.cos(np.pi/2-orientation)
    delta_y = width/2*np.sin(np.pi/2-orientation)
    extra_x = length*np.cos(orientation)
    extra_y = length*np.sin(orientation)
    vertices = np.array([[x+delta_x, y-delta_y], 
                         [x-delta_x, y+delta_y],
                         [x+extra_x-delta_x, y+extra_y+delta_y],
                         [x+extra_x+delta_x, y+extra_y-delta_y]])
    mpl.pyplot.plot(vertices[:,0], vertices[:,1])
    return vertices

"""
#def simulator():
pyplot = mpl.pyplot
Button = mpl.widgets.Button
sim_fig = pyplot.figure()
ax = sim_fig.add_subplot(111)
sim_fig.subplots_adjust(bottom=0.2)
# Select points from the graph
scatter_plot = ax.scatter(np.empty(0), np.empty(0))
callback = SLAM_SCENE(scatter_plot)
ax.set_title('click to create landmarks and set waypoints')
pyplot.xlim([-1, 1])
pyplot.ylim([-1, 1])

# Undo button
axundo = pyplot.axes([0.7, 0.05, 0.1, 0.075])
b_undo = Button(axundo, 'Undo')
b_undo.on_clicked(callback.undo)

# Redo button
axredo = pyplot.axes([0.81, 0.05, 0.1, 0.075])
b_redo = Button(axredo, 'Redo')
b_redo.on_clicked(callback.redo)

# Save button
axsave = pyplot.axes([0.51, 0.05, 0.1, 0.075])
b_save = Button(axsave, 'Save')
b_save.on_clicked(callback.save_scene)

# Radio buttons
axcolor = 'lightgoldenrodyellow'
rax = pyplot.axes([0.1, 0.025, 0.2, 0.1], axisbg=axcolor)
radio = mpl.widgets.RadioButtons(rax, ('landmarks', 'waypoints'))
radio.on_clicked(callback.setmode)
callback.setmode("landmarks")

cursor = mpl.widgets.Cursor(ax, useblit=True, color='red', linewidth=2 )

pyplot.show()
"""