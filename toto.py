# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:21:12 2018

@author: sonng
"""
import requests

#This URL will be the URL that your login form points to with the "action" tag.
POST_LOGIN_URL = 'http://www.cophieu68.vn/account/login.php'

#This URL is the page you actually want to pull down with requests.
REQUEST_URL = 'http://www.cophieu68.vn/export/excelfull.php?id=ACB'

payload = {
    'username': 'sonngtrung@gmail.com',
    'pass': '29011985'
}

with requests.Session() as session:
    post = session.post(POST_LOGIN_URL, data=payload)
    r = session.get(REQUEST_URL)
    print(r.text)   #or whatever else you want to do with the request data!
