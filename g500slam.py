#!/usr/bin/python

from girona500 import g500slam2
g500slam = g500slam2
g500slam.__PROFILE__ = False
g500slam.__PROFILE_NUM_LOOPS__ = 100
g500slam.main()
