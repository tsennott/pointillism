#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single derived class for pointillism package, handles gifs
"""

from .main import main
import numpy as np
import imageio
import inspect


class gif(main):
    """Subclass of pointillize for making stacks of images.
    Only supports single images currently"""

    def __init__(self, *args, **kwargs):

        pointillize.__init__(self, *args, **kwargs)

    def new_queue(self):
        """Builds a new set of lists for the queue"""

        self.queue = {'methods': [], 'names': [], 'args': [], 'repeats': []}

    def add_to_queue(self, method, args, n):
        """Adds a new method to the queue, to be run with args, n times"""

        self.queue['methods'].append(method)
        self.queue['names'].append(method.__name__)
        self.queue['args'].append(args)
        self.queue['repeats'].append(n)

    def print_queue(self):
        """Prints current status of the queue"""
        for i, method in enumerate(self.queue['methods']):
            print(self.queue['names'][i], self.queue['args']
                  [i], self.queue['repeats'][i])

    def run_queue(self, **kwargs):
        """Runs queue, primarily for build_stacks()"""

        # Make new image
        self._newImage(self.border)

        # Set some parameters
        save_steps = kwargs.get('save_steps', False)
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')

        to_print = True if self.debug & (frame_is_top | save_steps) else False

        if save_steps:
            if not dir().__contains__('self.image_stack'):
                self.image_stack = []

        for i, method in enumerate(self.queue['methods']):
            in_kwargs = self.queue['args'][i]
            n = self.queue['repeats'][i]

            if to_print:
                print(method.__name__ + ':', end=' ')

            for i in range(0, n):
                method(**in_kwargs)
                if save_steps:
                    self.image_stack.append(self.out.copy())
                if to_print:
                    print(i + 1, end=' ')
            if to_print:
                print("done")

    def build_stacks(self, n, save_steps):
        """Makes an image stack by running the pipeline n times,
        saving intermediate steps if save_steps is true"""

        self.image_stack = []

        to_print = True if (self.debug & save_steps is not True) else False

        if to_print:
            print('Building image: ', end=' ')
        for j in range(0, n):
            if to_print:
                print(j + 1, end=' ')
            self._newImage(self.border)
            self.run_queue(save_steps=save_steps)
            self.image_stack.append(self.out)
        if to_print:
            print('done')

    def build_multipliers(self, plot_list, **kwargs):
        """Plots the point queue repeatedly with multipliers from list set"""
        self.image_stack = []

        to_print = self.debug
        reverse = kwargs.get('reverse', True)
        reverse_list = kwargs.get('reverse_list', False)
        if reverse_list:
            self.pointQueue = sorted(self.pointQueue, key=lambda k: k['r'])
        n = len(plot_list)
        if to_print:
            print('Building image: ', end=' ')
        for j in range(0, n):
            if to_print:
                print(j + 1, end=' ')
            self._plotQueue(plot_list[j])
            self.image_stack.append(self.out)

        if reverse:
            self.image_stack += self.image_stack[::-1]

        if to_print:
            print('done')

    def save_gif(self, location, step_duration, **kwargs):
        """Save a gif of the image stack with step_duration"""

        arrays = []
        for out in self.image_stack:
            arrays.append(np.array(out))

        imageio.mimsave(location, arrays, format='gif', duration=step_duration)