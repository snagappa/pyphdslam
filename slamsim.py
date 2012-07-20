#!/usr/bin/python

import gtk
from simulator.gtk_simulator import gtk_slam_sim

try:
    slamsim = gtk_slam_sim()
    gtk.main()
except KeyboardInterrupt:
    pass
