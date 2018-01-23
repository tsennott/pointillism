#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes that help create pointillized images.
"""

import numpy as np
from PIL import Image, ImageDraw, ExifTags
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

        # Set debug state and initialize default params
        self.debug = kwargs.get('debug', False)
        self.params = {}
        self.params['reduce_factor'] = kwargs.get('reduce_factor', 2)
        self.point_queue = kwargs.get('queue', False)
        if self.point_queue:
            self._initQueue()

        # Get image if passed, or get from location
        image = kwargs.get('image', False)
        if image is False:
            location = kwargs.get('location', False)
            if location is False:
                raise ValueError('Must pass image or location')
            self.filename = location
            self.image = Image.open(self.filename)
        else:
            self.image = image
            self.filename = ['none']

        # Fix orientation if image is rotated
        if hasattr(self.image, '_getexif'):  # only present in JPEGs
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            e = self.image._getexif()       # returns None if no EXIF data
            if e is not None:
                exif = dict(e.items())
                if orientation in exif:
                    orientation = exif[orientation]
                    if orientation == 3:
                        self.image = self.image.transpose(Image.ROTATE_180)
                    elif orientation == 6:
                        self.image = self.image.transpose(Image.ROTATE_270)
                    elif orientation == 8:
                        self.image = self.image.transpose(Image.ROTATE_90)

        # Make array and blank canvas with borders
        self._build_array()
        self.border = kwargs.get('border', 100)
        self._newImage(self.border)

    def _build_array(self):
        """Builds np arrays of self.images"""

        d = (self.image.size[0]**2 + self.image.size[1]**2)**0.5
        self.params['reduce_factor'] = max(min(self.params['reduce_factor'],
                                           d / 1000), 1)
        w = int(self.image.size[0]/self.params['reduce_factor'])
        h = int(self.image.size[1]/self.params['reduce_factor'])
        resized = self.image.resize([w, h])
        self.array = np.array(resized).astype('float')

    def _newImage(self, border):
        """Creates new blank canvas with border"""

        h = self.image.size[1]
        w = self.image.size[0]
        self.out = Image.new(
                'RGB',
                [w + (border * 2), h + (border * 2)],
                (255, 255, 255))

    def print_attributes(self):
        """Prints non-hidden object parameters"""

        variables = vars(self)
        for var in variables:
            if var[0] != '_':
                if var == 'array':
                    print(var, ': ', '1 numpy array   ')
                else:
                    print(var, ': ', variables[var])

    def crop_Y(self, aspect, resize):
        """Crops and resizes in the height dimension to match aspect ratio"""

        w = self.image.size[0]
        h = self.image.size[1]
        h_new = w * aspect[1] // aspect[0]
        self.image = self.image.crop((0, h // 2 - h_new // 2,
                                      w, h // 2 + h_new // 2))
        if resize:
            self.image = self.image.resize([aspect[0], aspect[1]])

        self._build_array()
        self._newImage(self.border)

    def resize(self, ratio, min_size):
        """Resizes by ratio, or to min diagonal size in pixels"""

        w = self.image.size[0]
        h = self.image.size[1]
        d = (h**2 + w**2)**0.5
        ratio = max(ratio, min_size / d)

        self.image = self.image.resize([int(w * ratio),
                                       int(h * ratio)])

        self._build_array()
        self._newImage(self.border)

    def display(self, **kwargs):
        """Displays browser-size version of outputs, or original images
        if original=True"""

        original = kwargs.get('original', False)
        image = self.image if original else self.out
        print(self.filename)
        ratio = 1000/(image.size[0]**2 + image.size[1]**2)**0.5
        display(image.resize(
                [int(image.size[0] * ratio), int(image.size[1] * ratio)]))

    def _getColorOfPixel(self, loc, r):
        """Returns RGB tuple [0,255] of average color of the np array
        of an image within a square of width 2r at location loc=[x,y]"""

        # Redefine location to array based on reduce factor
        loc = [int(loc[0]/self.params['reduce_factor']),
               int(loc[1]/self.params['reduce_factor'])]
        r = int(r/self.params['reduce_factor'])

        left = int(max(loc[0] - r, 0))
        right = int(min(loc[0] + r, self.array.shape[1]))
        bottom = int(max(loc[1] - r, 0))
        top = int(min(loc[1] + r, self.array.shape[0]))
        x = range(left, right)
        y = range(bottom, top)
        if len(x) == 0 | len(y) == 0:
            return (255, 255, 255)
        R = int(self.array[np.ix_(y, x, [0])].mean())
        G = int(self.array[np.ix_(y, x, [1])].mean())
        B = int(self.array[np.ix_(y, x, [2])].mean())
        return (R, G, B)

    def _plotColorPoint(self, loc, r, **kwargs):
        """Plots point at loc with size r with average color from
        same in array"""

        alpha = kwargs.get('alpha', 255)
        border = self.border
        color = self._getColorOfPixel(loc, r)
        if self.point_queue:
            self._queueColorPoint(loc, r, color)
        else:
            # Hacking together transparency here for now
            # TODO handle more generally
            new_layer = Image.new('RGBA', (int(2*r), int(2*r)), (0, 0, 0, 0))

            draw = ImageDraw.Draw(new_layer)
            draw.ellipse((0, 0, 2*r, 2*r),
                         color + (int((random() * 0.5)**3 * 255 * 2**3),),)
            self.out.paste(new_layer, (border + loc[0] - int(r),
                                       border + loc[1] - int(r)),
                           new_layer)

    def plotRecPoints(self, n, multiplier, fill):
        """Plots symmetrical array of points over an image array,
        where n is the number of points across the diagonal,
        and multiplier is the ratio of the radius to the step
        and if fill is True, fills frame, otherwise leaves border"""

        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRecPoints:', end=' ')
        start = time.time()

        array = self.array
        h = array.shape[0]*self.params['reduce_factor']
        w = array.shape[1]*self.params['reduce_factor']
        step = (w**2 + h**2)**0.5/n
        r = step*multiplier
        if fill:
            for x in [int(x) for x in np.linspace(0, w, w // step)]:
                for y in [int(y) for y in np.linspace(0, h, h // step)]:
                    self._plotColorPoint([x, y], r)
        else:

            for x in [int(x) for x in np.linspace(r, w - r, w // step)]:
                for y in [int(y) for y in np.linspace(r, h - r,
                                                      h // step)]:
                    self._plotColorPoint([x, y], r)

        end = time.time()
        frame_is_top = (inspect.currentframe()
                        .f_back.f_code.co_name == '<module>')
        if to_print:
            print('done...took %0.2f sec' % (end - start))

    def _getComplexityOfPixel(self, array, loc, r):
        """Returns value [0,1] of average complexity of the np array
        of an image within a square of width 2r at location loc=[x,y]"""
        loc = [int(loc[0]/self.params['reduce_factor']),
               int(loc[1]/self.params['reduce_factor'])]
        r = int(r/self.params['reduce_factor'])

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

    def _getRadiusFromComplexity(self, d, power, constant, complexity):
        """Returns radius based on complexity calculation"""
        return np.ceil((complexity / 2)**(power) *
                        d * constant * 2**power + d/1000)

    def plotRandomPointsComplexity(self, n, constant, power, **kwargs):
        """plots random points over image, where constant is
        the portion of the diagonal for the max size of the bubble,
        and power pushes the distribution towards smaller bubbles"""

        alpha = kwargs.get('alpha', 255)
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRandomPointsComplexity:', end=' ')
        start = time.time()

        h = self.array.shape[0]*self.params['reduce_factor']
        w = self.array.shape[1]*self.params['reduce_factor']
        d = (h**2 + w**2)**0.5
        j = 0 
        while j < int(n):
            loc = [int(random() * w), int(random() * h)]
            # compare with probability matrix
            if random() < self._testProbability(loc):
                complexity = self._getComplexityOfPixel(
                self.array, loc, int(d * constant / 2))
                r = self._getRadiusFromComplexity(d, power, constant, complexity)
                self._plotColorPoint(loc, r, alpha=alpha)
                j+=1

        end = time.time()
        if to_print:
            print('done...took %0.2f sec' % (end - start))

    def _queueColorPoint(self, loc, r, color):
        """Builds queue of color points"""
        self.pointQueue.append({'loc': loc, 'r': r, 'color': color})

    def _initQueue(self):
        """Builds new point queue"""
        self.pointQueue = []

    def _plotQueue(self, multiplier):
        """Plots point queue"""
        self._newImage(self.border)
        border = self.border
        for point in self.pointQueue:
            loc = point['loc']
            r = int(point['r'] * multiplier)
            color = point['color']
            draw = ImageDraw.Draw(self.out)
            draw.ellipse((border + loc[0] - r, (border + loc[1] - r),
                          border + loc[0] + r, (border + loc[1] + r)),
                         color + (255,))

    def _makeProbabilityMask(self, constant, power):
        """Makes a mask of image complexity for improving
        distribution of dots to where they are needed"""
        
        self.probability_is_defined = True
        h = self.array.shape[0]*self.params['reduce_factor']
        w = self.array.shape[1]*self.params['reduce_factor']
        d = (h**2 + w**2)**0.5
        d_mask = int(1/constant)
        h_mask = int(d_mask/d * h)
        w_mask = int(d_mask/d * w)

        # TODO probably a faster way to compute this
        self.probabilityMask = np.empty([h_mask,w_mask])
        for i in range(0, h_mask):
            for j in range(0, w_mask):
                complexity = self._getComplexityOfPixel(self.array, 
                                                        [int(j*w/w_mask),
                                                        int(i*h/h_mask)],
                                                        int(constant*d))
                radius = self._getRadiusFromComplexity(d, power, constant, complexity)
                area_ratio = radius**2 / (constant*d)**2
                self.probabilityMask[i,j] =  1/area_ratio
        self.probabilityMask /= self.probabilityMask.max()

    def _testProbability(self, loc):
        if self.probability_is_defined:
            h = self.array.shape[0]*self.params['reduce_factor']
            w = self.array.shape[1]*self.params['reduce_factor']
            h_mask = self.probabilityMask.shape[0]
            w_mask = self.probabilityMask.shape[1]
            probability = self.probabilityMask[int(loc[1]*h_mask/h),int(loc[0]*w_mask/w)]
        else:
            probability = 1

        return probability

    def save_out(self, location, **kwargs):
        """Saves files to location"""

        suffix = kwargs.get('suffix', '')

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        self.out.save(
            location + '/' + self.filename.split('/')[1:][0] +
            ' - ' + suffix + '.png')


# Subclass adding workflows and image stack (gif) handling

class pointillizeStack(pointillize):
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

    def build_multipliers(self, set, **kwargs):
        """Plots the point queue repeatedly with multipliers from list set"""
        self.image_stack = []

        to_print = self.debug
        reverse = kwargs.get('reverse', False)
        n = len(set)
        if to_print:
            print('Building image: ', end=' ')
        for j in range(0, n):
            if to_print:
                print(j + 1, end=' ')
            self._plotQueue(set[j])
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


class pointillizePile(pointillizeStack):
    """Subclass of pointillizeStack for operating serially on images"""

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
        for i in range(0, len(self.pile_filenames)):
            print(i + 1, end=' ')
            self._init_pointilize(i)
            self.run_queue()
            self.save_out(location, **kwargs)
            self.filenames_store.append(self.filename)
            self.inputs_store.append(self.image)
            self.outputs_store.append(self.out)
        print('done')

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
