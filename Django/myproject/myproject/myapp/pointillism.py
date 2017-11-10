#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes that help create pointillized images.
"""

import numpy as np
from PIL import Image, ImageDraw
import imageio
from IPython.display import display
from random import random
import os
import time
import inspect


# Base class definitions, handles files and image manipulations


class pointillize:
    """Base class for pointillzation project"""

    def __init__(self, *args, **kwargs):
        """Initiialize with image or directory"""

        image = kwargs.get('image', False)
        if image is False:

            # Build list of filenames
            location = kwargs.get('location', False)

            if location is False:
                raise ValueError('Must declare image or dir to initialize')

            else:
                self.filenames = []

            if os.path.isdir(location):
                for file in os.listdir(location):
                    if (file.endswith(".jpg") | file.endswith(".JPG") |
                       file.endswith(".png") | file.endswith(".PNG")):
                        self.filenames.append(location + file)

            else:
                self.filenames.append(location)

        else:
            self.images = [image]
            self.filenames = ['none']
            self._build_arrays()

        # Make blank canvases with borders
        self.border = kwargs.get('border', 100)
        self._newImage(self.border)

        # Set debug state and initialize default params
        self.debug = kwargs.get('debug', False)
        self.params = {}
        self.params['complexity_radius'] = kwargs.get('complexity_radius', 10)

    def _open_images(self):
        """Opens images"""
        self.images = []
        for file in self.filenames:
            image = Image.open(file)
            self.images.append(image)
        self._build_arrays()

    def _build_arrays(self):
        """Builds np arrays of self.images"""
        self.arrays = []
        for image in self.images:
            self.arrays.append(np.array(image).astype('float'))

    def _newImage(self, border):
        """Creates new blank canvas with border"""

        self.outs = []
        for image in self.images:
            h = image.size[1]
            w = image.size[0]
            self.outs.append(Image.new(
                'RGBA',
                [w + (border * 2), h + (border * 2)],
                (255, 255, 255, 0)))

    def print_attributes(self):
        """Prints non-hidden object parameters"""

        variables = vars(self)
        for var in variables:
            if var[0] != '_':
                if var == 'arrays':
                    print(var, ':', len(variables[var]), ' numpy array(s)   ')
                else:
                    print(var, ': ', variables[var])

    def crop_Y(self, aspect, resize):
        """Crops and resizes in the height dimension to match aspect ratio"""

        for i, image in enumerate(self.images):
            w = image.size[0]
            h = image.size[1]
            h_new = w * aspect[1] // aspect[0]
            image = image.crop((0, h // 2 - h_new // 2,
                                w, h // 2 + h_new // 2))
            if resize:
                self.images[i] = image.resize([aspect[0], aspect[1]])
            else:
                self.images[i] = image

        self._build_arrays()
        self._newImage(self.border)

    def display(self, **kwargs):
        """Displays browser-size version of outputs, or original images
        if original=True"""

        original = kwargs.get('original', False)
        images = self.images if original else self.outs
        for i, image in enumerate(images):
            print(self.filenames[i])
            display(image.resize(
                [1000, image.size[1] * 1000 // image.size[0]]))

    def _getColorOfPixel(self, array, loc, r):
        """Returns RGB tuple [0,255] of average color of the np array
        of an image within a square of width 2r at location loc=[x,y]"""
        left = int(max(loc[0] - r, 0))
        right = int(min(loc[0] + r, array.shape[1]))
        bottom = int(max(loc[1] - r, 0))
        top = int(min(loc[1] + r, array.shape[0]))
        x = range(left, right)
        y = range(bottom, top)
        if len(x) == 0 | len(y) == 0:
            return (255, 255, 255)
        R = int(array[np.ix_(y, x, [0])].mean())
        G = int(array[np.ix_(y, x, [1])].mean())
        B = int(array[np.ix_(y, x, [2])].mean())
        return (R, G, B)

    def _plotColorPoint(self, image, array, loc, r):
        """Plots point at loc with size r with average color from
        same in array"""
        border = self.border
        color = self._getColorOfPixel(array, loc, r)
        draw = ImageDraw.Draw(image)
        draw.ellipse((border + loc[0] - r, (border + loc[1] - r),
                      border + loc[0] + r, (border + loc[1] + r)),
                     color + (255,))

    def plotRecPoints(self, step, r, fill):
        """Plots rectangular array of points over an image array,
        where step is the step size in pixels, r is the radius in pixels,
        and if fill is True, fills frame, otherwise leaves border"""
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRecPoints:', end=' ')
        start = time.time()
        for i, image in enumerate(self.outs):
            array = self.arrays[i]
            h = array.shape[0]
            w = array.shape[1]
            if fill:
                for x in [int(x) for x in np.linspace(0, w, w // step)]:
                    for y in [int(y) for y in np.linspace(0, h, h // step)]:
                        self._plotColorPoint(image, array, [x, y], r)
            else:

                for x in [int(x) for x in np.linspace(r, w - r, w // step)]:
                    for y in [int(y) for y in np.linspace(r, h - r,
                                                          h // step)]:
                        self._plotColorPoint(image, array, [x, y], r)
            self.outs[i] = image
            if to_print:
                print(i + 1, end=' ')
        end = time.time()
        frame_is_top = (inspect.currentframe()
                        .f_back.f_code.co_name == '<module>')
        if to_print:
            print('done...took %0.2f sec' % (end - start))

    def plotRecPointsFill(self, n, fill):
        """Plots symmetrical array of points over an image array,
        where n is the number of points across the horizontal,
        and if fill is True, fills frame, otherwise leaves border"""
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRecPoints:', end=' ')
        start = time.time()
        for i, image in enumerate(self.outs):
            array = self.arrays[i]
            h = array.shape[0]
            w = array.shape[1]
            step = w/n
            r = step
            if fill:
                for x in [int(x) for x in np.linspace(0, w, w // step)]:
                    for y in [int(y) for y in np.linspace(0, h, h // step)]:
                        self._plotColorPoint(image, array, [x, y], r)
            else:

                for x in [int(x) for x in np.linspace(r, w - r, w // step)]:
                    for y in [int(y) for y in np.linspace(r, h - r,
                                                          h // step)]:
                        self._plotColorPoint(image, array, [x, y], r)
            self.outs[i] = image
            if to_print:
                print(i + 1, end=' ')
        end = time.time()
        frame_is_top = (inspect.currentframe()
                        .f_back.f_code.co_name == '<module>')
        if to_print:
            print('done...took %0.2f sec' % (end - start))

    def plotRandomPoints(self, n, constant, power):
        """Plots n random points over image, where constant is the portion
        of the image width for the max size of the bubble, and power > 1
        pushing the distribution towards smaller bubbles for increasing
        complexity, and power [0,1] making the distribution flatter"""
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRandomPoints:', end=' ')
        start = time.time()
        for i, image in enumerate(self.outs):
            array = self.arrays[i]
            h = array.shape[0]
            w = array.shape[1]
            for j in range(0, int(n)):
                loc = [int(random() * w), int(random() * h)]
                r = int((random() / 2)**(power) * w * constant) * 2**power + 1
                self._plotColorPoint(image, array, loc, r)
            self.outs[i] = image
            if to_print:
                print(i + 1, end=' ')
        end = time.time()
        frame_is_top = (inspect.currentframe()
                        .f_back.f_code.co_name == '<module>')
        if to_print:
            print('done...took %0.2f sec' % (end - start))

    def _getComplexityOfPixel(self, array, loc, r):
        """Returns value [0,1] of average color of the np array
        of an image within a square of width 2r at location loc=[x,y]"""
        left = max(loc[0] - r, 0)
        right = min(loc[0] + r, array.shape[1])
        bottom = max(loc[1] - r, 0)
        top = min(loc[1] + r, array.shape[0])
        x = range(left, right)
        y = range(bottom, top)
        if len(x) == 0 | len(y) == 0:
            return 0
        R = array[np.ix_(y, x, [0])].max() - array[np.ix_(y, x, [0])].min()
        G = array[np.ix_(y, x, [1])].max() - array[np.ix_(y, x, [1])].min()
        B = array[np.ix_(y, x, [2])].max() - array[np.ix_(y, x, [2])].min()
        if (np.isnan(R) | np.isnan(G) | np.isnan(B)):
            R, G, B = 0, 0, 0
        return 1 - (R + G + B) / (255 * 3.0)

    def plotRandomPointsComplexity(self, n, constant, power):
        """plots random points over image, where constant is
        the portion of the width for the max size of the bubble,
        and power pushes the distribution towards smaller bubbles"""
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRandomPointsComplexity:', end=' ')
        start = time.time()
        for i, image in enumerate(self.outs):
            array = self.arrays[i]
            h = array.shape[0]
            w = array.shape[1]
            for j in range(0, int(n)):
                loc = [int(random() * w), int(random() * h)]
                complexity = self._getComplexityOfPixel(
                    array, loc, self.params['complexity_radius'])
                r = int((complexity / 2)**(power) *
                        w * constant * 2**power + 5)
                self._plotColorPoint(image, array, loc, r)
            self.outs[i] = image
            if to_print:
                print(i + 1, end=' ')

        end = time.time()
        if to_print:
            print('done...took %0.2f sec' % (end - start))

    def save_out(self, location, **kwargs):
        """Saves files to location"""

        suffix = kwargs.get('suffix', '')

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        for i, image in enumerate(self.outs):
            image.save(
                location + '/' + self.filenames[i].split('/')[1:][0] +
                ' - ' + suffix + '.png')


# Subclass adding workflows and image stack (gif) handling

class pointillizeStack(pointillize):
    """Subclass of pointillize for making stacks of images.
    Only supports single images currently"""

    def __init__(self, *args, **kwargs):

        pointillize.__init__(self, *args, **kwargs)

        assert len(self.images) == 1, "Only single images are supported"

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
            args = self.queue['args'][i]
            n = self.queue['repeats'][i]

            if to_print:
                print(method.__name__ + ':', end=' ')

            for i in range(0, n):
                method(*args)
                if save_steps:
                    self.image_stack.append(self.outs[0].copy())
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
            self.image_stack.append(self.outs[0])
        if to_print:
            print('done')

    def save_gif(self, location, step_duration, **kwargs):
        """Save a gif of the image stack with step_duration"""

        arrays = []
        for out in self.image_stack:
            arrays.append(np.array(out))

        imageio.mimsave(location, arrays, duration=step_duration)

        # Deprecated, using PIL
        # image = self.image_stack[0]
        # image.save(fp=location, format='gif', save_all=True,
        #           append_images=self.image_stack[1:])


class pointillizePile(pointillizeStack):
    """Subclass of pointillizeStack for operating serially on images, for
    savings gifs of whole directories or processing large batches of files
    where pointillize operating in parallel would be undesirable"""

    def __init__(self, *args, **kwargs):

        location = kwargs.get('location', False)
        if location is False:
            raise ValueError('Must declare directory to initialize')
        else:
            self.pile_filenames = []
            if os.path.isdir(location):
                for file in os.listdir(location):
                    if file.endswith(".jpg") | file.endswith(".JPG"):
                        self.pile_filenames.append(location + file)
            else:
                raise ValueError('Must declare directory to initialize')
        self._kwargs = kwargs
        self._args = args
        self._init_pointilize(index=0)

    def _init_pointilize(self, index):

        args = self._args
        kwargs = self._kwargs
        kwargs['location'] = self.pile_filenames[index]
        pointillize.__init__(self, *args, **kwargs)

    def run_pile_images(self, location, **kwargs):
        """Process and save files to location"""

        print('Batch processing image:', end=' ')
        for i in range(0, len(self.pile_filenames)):
            print(i + 1, end=' ')
            self._init_pointilize(i)
            self.run_queue()
            self.save_out(location, **kwargs)
        print('done')

    def run_pile_gifs(self, location, n, save_steps, step_duration, **kwargs):

        suffix = kwargs.get('suffix', '')

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        for i in range(0, len(self.pile_filenames)):
            print(i + 1, end=' ')
            self._init_pointilize(i)
            self.build_stacks(n, save_steps)
            self.save_gif(location + '/' + self.filenames[0].split('/')[1] +
                          ' ' + suffix + '.gif', step_duration, **kwargs)
