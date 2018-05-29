# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:21:12 2018

@author: sonng
"""

import pandas_datareader.data as web
import datetime as dt

from datetime import date
today = '2018-05-28'
data = web.DataReader('AAPL',  'yahoo', dt.datetime(2018,5,25), dt.datetime(2018,5,25))
