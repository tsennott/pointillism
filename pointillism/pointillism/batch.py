#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single derived class for pointillism package, handles batches of files
"""

from .main import main
from IPython.display import display
import os
import time


class batch(main):
    """Subclass of pointillizeStack for operating serially on images"""

    def __init__(self, *args, **kwargs):

        location = kwargs.get('location', False)
        if location is False:
            raise ValueError('Must declare directory to initialize')
        else:
            self.pile_filenames = []
            if os.path.isdir(location):
                for file in os.listdir(location):
                    if file.endswith(".jpg") | file.endswith(".JPG") | file.endswith(".jpeg") | file.endswith(".JPEG"):
                        self.pile_filenames.append(location + file)
            else:
                raise ValueError('Must declare directory to initialize')

        self.outputs_store = []
        self.inputs_store = []
        self.filenames_store = []
        self._kwargs = kwargs
        self._args = args
        self._init_pointilize(index=0)

    def _init_pointilize(self, index):

        args = self._args
        kwargs = self._kwargs
        kwargs['location'] = self.pile_filenames[index]
        pointillize.__init__(self, *args, **kwargs)

    def display(self, **kwargs):
        """Displays browser-size version of outputs, or original images
        if original=True"""

        original = kwargs.get('original', False)
        for i in range(len(self.inputs_store)):
            image = self.inputs_store[i] if original else self.outputs_store[i]
            print(self.filenames_store[i])
            ratio = 500/(image.size[0]**2 + image.size[1]**2)**0.5
            display(image.resize(
                    [int(image.size[0] * ratio), int(image.size[1] * ratio)]))

    def run_pile_images(self, location, **kwargs):
        """Process and save files to location"""

        print('Batch processing image:', end=' ')
        start=time.time()
        for i in range(0, len(self.pile_filenames)):
            print(i + 1, end=' ')
            self._init_pointilize(i)
            self.run_queue()
            self.save_out(location, **kwargs)
            self.filenames_store.append(self.filename)
            self.inputs_store.append(self.image)
            self.outputs_store.append(self.out)
        print('done....took %0.2f seconds' % (time.time()-start))


    def run_pile_gifs(self, location, n, save_steps, step_duration, **kwargs):

        suffix = kwargs.get('suffix', '')

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        for i in range(0, len(self.pile_filenames)):
            print(i + 1, end=' ')
            self._init_pointilize(i)
            self.build_stacks(n, save_steps)
            self.save_gif(location + '/' + self.filename.split('/')[1] +
                          ' ' + suffix + '.gif', step_duration, **kwargs)

    def run_pile_multipliers(self, location, multipliers,
                             step_duration, **kwargs):

        suffix = kwargs.get('suffix', '')
        reverse = kwargs.get('reverse', False)

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        for i in range(0, len(self.pile_filenames)):
            print(i + 1, end=' ')
            self._init_pointilize(i)
            self.run_queue()
            self.build_multipliers(multipliers, reverse=reverse)
            self.save_gif(location + '/' + self.filename.split('/')[1] +
                          ' ' + suffix + '.gif', step_duration, **kwargs)
