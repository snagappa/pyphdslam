#!/usr/bin/python

from girona500 import g500slam2
g500slam = g500slam2
g500slam.__PROFILE__ = True
g500slam.__PROFILE_NUM_LOOPS__ = 1000
g500slam.main()
