#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:28:08 2022

@author: gregg
"""
import pickle

def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

# call this as pickle_all(dir(),globals(),save_name)
def pickle_all(dir_,globals_,save_name):
    bk = {}
    for k in dir_:
        obj = globals_[k]
        if is_picklable(obj):
            try:
                bk.update({k: obj})
            except TypeError:
                pass
        
    with open(save_name, 'wb') as f:
        pickle.dump(bk, f)
        
def pickle_var_list(var_list,globals_,save_name):
    bk = {}
    for k in var_list:
        obj = globals_[k]
        if is_picklable(obj):
            try:
                bk.update({k: obj})
            except TypeError:
                pass
        
    with open(save_name, 'wb') as f:
        pickle.dump(bk, f)
        
def load_pickle(save_name):
    with open(save_name, 'rb') as f:
        bk = pickle.load(f)
    return bk