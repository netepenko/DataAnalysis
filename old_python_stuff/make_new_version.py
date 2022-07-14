#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:27:11 2021

@author: boeglinw
"""
def make_new_version(file_spec):
    import os, shutil

    if os.path.isfile(file_spec):
        # Determine root filename so the extension doesn't get longer
        n, e = os.path.splitext(file_spec)
        # Is e an integer?
        try:
             num = int(e)
             root = n
        except ValueError:
             root = file_spec

        # Find next available file version
        for i in range(1000):
             new_file = f'{root}.{i:d}'
             if not os.path.isfile(new_file):
                      shutil.copy(file_spec, new_file)
                      return new_file
        return ''

