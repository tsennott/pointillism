#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single derived class for pointillism package, handles gifs
"""

from .image import image
import numpy as np
import imageio
import inspect
import os


class pipeline(image):
    """Subclass of pointillize for making stacks of images.
    Only supports single images currently"""

    def __init__(self, *args, **kwargs):

        image.__init__(self, *args, **kwargs)

    def new_queue(self):
        """Builds a new set of lists for the queue"""

        self.queue = {'methods': [], 'names': [], 'args': [], 'repeats': []}

    def add_to_queue(self, method, args, n):
        """Adds a new method to the queue, to be run with args, n times"""

        self.queue['methods'].append(method)
        self.queue['names'].append(method.__name__)
        self.queue['args'].append(args)
        self.queue['repeats'].append(n)

    def _print_queue(self):
        """Prints current status of the queue"""
        for i, method in enumerate(self.queue['methods']):
            print(self.queue['names'][i], self.queue['args']
                  [i], self.queue['repeats'][i])

    def _run_queue(self, **kwargs):
        """Runs queue, primarily for build_stacks()"""

        # Make new image
        self._newImage(self.border)

        # Set some parameters
        save_steps = kwargs.get('save_steps', False)
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')

        to_print = True if self.debug & (frame_is_top) else False

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

        to_print = True if self.debug else False

        if to_print:
            print('Building image: ', end=' ')
        for j in range(0, n):
            if to_print:
                print(j + 1, end=' ')
            self._newImage(self.border)
            self._run_queue(save_steps=save_steps)
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

    def make_gif(self, location='./', name=None, kind='multiplier', crop=False, **kwargs):
        """ Makes a gif of kind multiplier, assembly, or loop

            Optional kwargs:
                bulk=False
                    set to True for bulk operation
                size=[1000, 500] or size=0.25
                    for crop=True should be [width, height], otherwise should be ratio
                n_total=1e4
                    for kind 'stacking', determines total points to use
                setting='balanced'
                    for kinds multiplier and loop, determines preset to use
        """
        # Get kwargs
        bulk = kwargs.get('bulk', False)
        n_total = kwargs.get('n_total', 1e4)
        setting = kwargs.get('setting', 'balanced')
        size_default = [1000, 500] if crop else 0.25
        size = kwargs.get('size', size_default)

        if kind == 'assembly':

            # Construct queue
            self.new_queue()

            # Resize
            if not bulk:
                if crop:
                    self.add_to_queue(self.crop, {'aspect': size, 'resize': True}, 1)
                else:
                    self.add_to_queue(self.resize, {'ratio': size, 'min_size': 200}, 1)

            # Make
            self.add_to_queue(self.plotRecPoints, {'n': 40, 'multiplier': 1, 'fill': True}, 1)
            self.add_to_queue(self.plotRandomPointsComplexity, {'n': n_total / 10, 'constant': 0.012,
                                                                'power': 3, 'grad_size': .015,
                                                                'min_size': 0.002}, 10)
            # Build image stacks
            self.build_stacks(n=1, save_steps=True)

        elif kind == 'loop':
            # Construct queue
            self.new_queue()

            # Resize
            if not bulk:
                if crop:
                    self.add_to_queue(self.crop, {'aspect': size, 'resize': True}, 1)
                else:
                    self.add_to_queue(self.resize, {'ratio': size, 'min_size': 200}, 1)

            # Make
            self.add_to_queue(self.make, {'setting': setting}, 1)

            # Build image stacks
            self.build_stacks(n=10, save_steps=False)

        elif kind == 'multiplier':
            # Set up queue
            self.point_queue = True
            self._initQueue()

            # Resize
            if not bulk:
                if crop:
                    self.crop(aspect=size, resize=True)
                else:
                    self.resize(ratio=size, min_size=200)

            # Queue up points
            self.make('balanced')

            # Build multipliers
            multipliers = [10, 8, 6, 5, 4.5, 4, 3.5, 3, 2.6, 2.3,
                           2, 1.75, 1.5, 1.25, 1.1, 1, 1, 1, 1, 1]
            self.build_multipliers(multipliers, reverse=True)

        else:
            raise ValueError('Invalid kind')

        # Save
        if not bulk: self.save_gif(location=location, name=name, step_duration=0.1)

    def save_gif(self, location='./', name=None, step_duration=0.1, **kwargs):
        """Save a gif of the image stack with step_duration. If no name
        is given, uses name of source image with optional suffix and prefix.

            Optional kwargs:
                suffix='pointillized'
                    optional suffix to add to file
                prefix=''
                    optional prefix to add to file
        """

        suffix = kwargs.get('suffix', 'pointillized')
        prefix = kwargs.get('prefix', '')

        arrays = []
        for out in self.image_stack:
            arrays.append(np.array(out))

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        if name is None:
            name = self.filename.split('/')[-1]
            file_to_save = os.path.join(location, (prefix + name + ' - ' + suffix + '.gif'))
        else:
            file_to_save = os.path.join(location, name)

        imageio.mimsave(file_to_save, arrays, format='gif', duration=step_duration)
