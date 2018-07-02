#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is for experimenting with multiprocessing.
"""


def f(list):

    point_object = list[0]
    setting = list[1]

    point = point_object(location='images_bulk/', debug=True, increase_factor=1)

    # Crop and build queue
    point.new_queue()
    # point.add_to_queue(point.crop, {'aspect': [1920, 1080], 'resize': True}, 1)
    point.add_to_queue(point.resize, {'ratio': 0, 'min_size': 2200}, 1)
    point.add_to_queue(point.plot, {'setting': setting}, 1)

    # Run and save
    point.run_pile_images(location='images_bulk_out', suffix=setting)

    return setting
