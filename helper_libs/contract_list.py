#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:39:06 2023

@author: gregg
"""

def cl(l):
    new_l = [l[0]]
    last_element=l[0]
    for element in l:
        if element != last_element:
            new_l.append(element)
        last_element=element
        
    for i in reversed(range(len(new_l))):
        if new_l[i] in new_l[0:i]:
            del new_l[i]
        
    return new_l