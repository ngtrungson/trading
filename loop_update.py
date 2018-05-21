# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:23:49 2018

@author: sonng
"""
from threading import Timer
from VNINDEX import portfolio_management
    
t = Timer(100.0, portfolio_management())
t.start()