#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single base class for pointillism package, handles all basic image functions
"""

import numpy as np
from PIL import Image, ImageDraw, ExifTags, ImageEnhance
from scipy import ndimage
from IPython.display import display
from random import random
import os
import time
import inspect
from matplotlib import pyplot as plt


class image:
    """Base class for pointillizer"""

    def __init__(self, location=None, image=None, debug=False, **kwargs):
        """Initiialize with image or directory

            Optional kwargs:
                reduce_factor=2
                    reduces size of lookup array
                increase_factor=1
                    increases size of output image
                queue=False
                    instead of plotting, queues points for later plotting
                plot_coverage=True
                    plots black and white mask of image coverage
        """

        # Set debug state and initialize default params
        self.debug = debug
        self.params = {}
        self.params['reduce_factor'] = kwargs.get('reduce_factor', 2)
        self.params['increase_factor'] = kwargs.get('increase_factor', 1)
        self.point_queue = kwargs.get('queue', False)
        self.plot_coverage = kwargs.get('plot_coverage', True)
        if self.point_queue:
            self._initQueue()

        # Get image if passed, or get from location
        if image is None:
            if location is None:
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

        # Make meta settings
        self._makeMetaSettings()

    def _build_array(self):
        """Builds np arrays of self.images"""

        d = (self.image.size[0]**2 + self.image.size[1]**2)**0.5
        self.params['reduce_factor'] = max(min(self.params['reduce_factor'],
                                           d / 1000), 1)
        self.params['net_factor'] = (self.params['reduce_factor'] *
                                     self.params['increase_factor'])
        w = int(self.image.size[0] / self.params['reduce_factor'])
        h = int(self.image.size[1] / self.params['reduce_factor'])
        resized = self.image.resize([w, h])
        self.array = np.array(resized).astype('float')

    def _newImage(self, border):
        """Creates new blank canvas with border"""

        h = self.image.size[1] * self.params['increase_factor']
        w = self.image.size[0] * self.params['increase_factor']
        self.out = Image.new(
            'RGB',
            [w + (border * 2), h + (border * 2)],
            (255, 255, 255))
        if self.plot_coverage:
            self.out_coverage = Image.new(
                'L',
                [w + (border * 2), h + (border * 2)],
                (0,))

    def _makeMetaSettings(self):

        self.settings = {
            'uniform': {
                'PlotRecPoints': {'n': 40, 'fill': (self.border == 0)},
                'PlotPointsComplexity': {'constant': 0.004, 'power': 1, 'grad_size': .005, 'min_size': 0.004}
            },
            'coarse': {
                'PlotRecPoints': {'n': 20, 'fill': (self.border == 0)},
                'PlotPointsComplexity': {'constant': 0.016, 'power': 2, 'grad_size': .019, 'min_size': 0.004}
            },
            'balanced': {
                'PlotRecPoints': {'n': 100, 'fill': (self.border == 0)},
                'PlotPointsComplexity': {'constant': 0.012, 'power': 3, 'grad_size': .015, 'min_size': 0.002}
            },
            'fine': {
                'PlotRecPoints': {'n': 100, 'fill': (self.border == 0)},
                'PlotPointsComplexity': {'constant': 0.008, 'power': 3, 'grad_size': .01, 'min_size': 0.001}
            },
            'ultrafine': {
                'PlotRecPoints': {'n': 100, 'fill': (self.border == 0)},
                'PlotPointsComplexity': {'constant': 0.005, 'power': 3, 'grad_size': .006, 'min_size': 0.0005}
            },
        }

    def crop(self, aspect, resize=False, direction='height'):
        """Crops and resizes in the dimension specified ('height' or 'width')"""

        w = self.image.size[0]
        h = self.image.size[1]
        if direction == 'height':
            h_new = w * aspect[1] // aspect[0]
            self.image = self.image.crop((0, h // 2 - h_new // 2,
                                          w, h // 2 + h_new // 2))
        elif direction == 'width':
            w_new = h * aspect[0] // aspect[1]
            self.image = self.image.crop((w // 2 - w_new // 2, 0,
                                          w // 2 + w_new // 2, h))
        else:
            raise ValueError('Invalid direction argument')

        if resize:
            self.image = self.image.resize([aspect[0], aspect[1]])

        self._build_array()
        self._newImage(self.border)

    def enhance(self, kind='contrast', amount=1):
        """Multiplies kind ('contrast', 'sharpness', 'color') by amount"""

        if kind == 'contrast':
            self.image = ImageEnhance.Contrast(self.image).enhance(amount)
        elif kind == 'sharpness':
            self.image = ImageEnhance.Sharpness(self.image).enhance(amount)
        elif kind == 'color':
            self.image = ImageEnhance.Color(self.image).enhance(amount)
        else:
            raise Exception('Invalid Type')

        self._build_array()
        self._newImage(self.border)

    def resize(self, ratio, min_size):
        """Resizes by ratio, or to min diagonal size in pixels,
        whichever is larger"""

        w = self.image.size[0]
        h = self.image.size[1]
        d = (h**2 + w**2)**0.5
        ratio = max(ratio, float(min_size / d))

        self.image = self.image.resize([int(w * ratio),
                                       int(h * ratio)])

        self._build_array()
        self._newImage(self.border)

    def display(self, original=False, coverage=False, gradient=False):
        """Displays browser-size version of outputs, or original images
        or coverage map or gradient"""

        if original:
            image = self.image
        elif coverage:
            image = self.out_coverage
        elif gradient:
            image = Image.fromarray((self.array_complexity * 255).astype('uint8'))
        else:
            image = self.out

        print(self.filename)
        ratio = 1000 / (image.size[0]**2 + image.size[1]**2)**0.5
        display(image.resize(
                [int(image.size[0] * ratio), int(image.size[1] * ratio)]))

    def _getColorOfPixel(self, loc, r):
        """Returns RGB tuple [0,255] of average color of the np array
        of an image within a square of width 2r at location loc=[x,y]"""

        # Redefine location to array based on reduce factor
        loc = [int(loc[0] / self.params['net_factor']),
               int(loc[1] / self.params['net_factor'])]
        r = max(int(r / self.params['net_factor']), 1)

        left = int(max(loc[0] - r, 0))
        right = int(min(loc[0] + r, self.array.shape[1]))
        bottom = int(max(loc[1] - r, 0))
        top = int(min(loc[1] + r, self.array.shape[0]))
        x = range(left, right)
        y = range(bottom, top)
        if len(x) == 0 | len(y) == 0:
            return (255, 255, 255)

        R = int(self.array[bottom:top, left:right, [0]].mean())
        G = int(self.array[bottom:top, left:right, [1]].mean())
        B = int(self.array[bottom:top, left:right, [2]].mean())

        return (R, G, B)

    def _plotColorPoint(self, loc, r, mask=False, **kwargs):
        """Plots point at loc with size r with average color from
        same in array"""

        use_transparency = kwargs.get('use_transparency', False)
        alpha_fcn = kwargs.get('alpha_fcn', lambda: 255)
        border = self.border
        color = self._getColorOfPixel(loc, r)

        # Hacking together transparency here for now
        # TODO handle more generally
        if use_transparency:
            alpha = int(alpha_fcn())
            self.alpha_list.append(alpha)
        else:
            alpha = 255

        if self.point_queue:
            self._queueColorPoint(loc, r, color)
        else:
            new_layer = Image.new('RGBA', (int(3 * r), int(3 * r)), (0, 0, 0, 0))
            draw = ImageDraw.Draw(new_layer)
            draw.ellipse((0, 0, 2 * r, 2 * r),
                         color + (alpha,))
            self.out.paste(new_layer, (border + loc[0] - int(r),
                                       border + loc[1] - int(r)),
                           new_layer)

        if self.plot_coverage & mask:
            color = (255, 255, 255)
            new_layer = Image.new('RGBA', (int(3 * r), int(3 * r)), (0, 0, 0, 0))
            draw = ImageDraw.Draw(new_layer)
            draw.ellipse((0, 0, 2 * r, 2 * r),
                         color + (alpha,))
            self.out_coverage.paste(new_layer, (border + loc[0] - int(r),
                                                border + loc[1] - int(r)),
                                    new_layer)

    def plotRecPoints(self, n=40, multiplier=1, fill=False):
        """Plots symmetrical array of points over an image array,
        where n is the number of points across the diagonal,
        and multiplier is the ratio of the radius to the step.
        If fill is True, fills frame, otherwise leaves border"""

        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRecPoints:', end=' ')
        start = time.time()
        count = 0
        h = self.array.shape[0] * self.params['net_factor']
        w = self.array.shape[1] * self.params['net_factor']
        step = (w**2 + h**2)**0.5 / n
        r = step * multiplier
        if fill:
            for x in [int(x) for x in np.linspace(0, w, w // step)]:
                for y in [int(y) for y in np.linspace(0, h, h // step)]:
                    self._plotColorPoint([x, y], r)
                    count += 1
        else:

            for x in [int(x) for x in np.linspace(r, w - r, w // step)]:
                for y in [int(y) for y in np.linspace(r, h - r,
                                                      h // step)]:
                    self._plotColorPoint([x, y], r)
                    count += 1

        end = time.time()
        frame_is_top = (inspect.currentframe()
                        .f_back.f_code.co_name == '<module>')
        if to_print:
            print('done...took %0.2f sec for %d points' % ((end - start), count))

    def _getComplexityOfPixel(self, array, loc, r, use_complexity=True):
        """DEPRECATED: Returns value [0,1] of average complexity of the np array
        of an image within a square of width 2r at location loc=[x,y]"""

        if use_complexity:
            loc = [int(loc[0] / self.params['net_factor']),
                   int(loc[1] / self.params['net_factor'])]
            r = int(r / self.params['net_factor'])

            left = max(loc[0] - r, 0)
            right = min(loc[0] + r, array.shape[1])
            bottom = max(loc[1] - r, 0)
            top = min(loc[1] + r, array.shape[0])
            x = range(left, right)
            y = range(bottom, top)
            if len(x) == 0 | len(y) == 0:
                return 0
            R = array[bottom:top, left:right, [0]].max() - array[bottom:top, left:right, [0]].min()
            G = array[bottom:top, left:right, [1]].max() - array[bottom:top, left:right, [1]].min()
            B = array[bottom:top, left:right, [2]].max() - array[bottom:top, left:right, [2]].min()
            if (np.isnan(R) | np.isnan(G) | np.isnan(B)):
                R, G, B = 0, 0, 0
            return 1 - (R + G + B) / (255 * 3.0)
        else:
            return np.random.random()

    def _generateRandomPoints(self, n):
        h = self.out.size[1]
        w = self.out.size[0]
        locations = []
        for i in range(0, int(n)):
            locations.append([int(random() * w), int(random() * h)])
        return locations

    def _getRadiusFromComplexity(self, d, power, constant, min_size, complexity):
        """Returns radius based on complexity calculation"""
        return np.ceil((complexity / 2)**(power) *
                       d * constant * 2**power + max(d * min_size, 2))

    def plotRandomPointsComplexity(self, n=int(1e5), max_skips=int(2e3), constant=1e-2,
                                   power=1, min_size=1e-3, **kwargs):
        """Plots n random points over image using smaller points for more complex areas.
        If n is high enough, image will be saturated and max_skips will be used to
        determine when the image is sufficiently filled.
        Constant is the portion of the diagonal for the max size of the bubble,
        and power pushes the distribution towards smaller bubbles.
        Min_size is the portion of the diagonal for the min bubble size.

        Optional kwargs:
            use_transparency=False
                allows transparency for points
            alpha_fcn=(lambda: ((random() * 0.5)**3 * 255 * 2**3))
                function for determining transparency
            use_complexity=True
                uses complexity to determine point size
            use_gradient=True
                pre-computes gradient for complexity lookup (faster)
            grad_size=20
                size of gradient maximum filter
            grad_multiplier=1
                tweaks distribution of gradient
        """

        use_transparency = kwargs.get('use_transparency', False)
        alpha_fcn = kwargs.get('alpha_fcn', lambda: ((random() * 0.5)**3 * 255 * 2**3))
        use_coverage = kwargs.get('use_coverage', True)
        use_complexity = kwargs.get('use_complexity', True)
        use_gradient = kwargs.get('use_gradient', True)
        grad_size = kwargs.get('grad_size', 20)
        grad_mult = kwargs.get('grad_mult', 1)

        if use_gradient:
            self._makeComplexityArray(1, grad_size, grad_mult)

        locations = kwargs.get('locations', False)
        if locations is not False:
            n = len(locations)
        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        if to_print:
            print('plotRandomPointsComplexity:', end=' ')
        start = time.time()

        h = self.array.shape[0] * self.params['net_factor']
        w = self.array.shape[1] * self.params['net_factor']
        d = (h**2 + w**2)**0.5
        j = 0
        count = 0
        points = 0
        if self.debug:
            self.count_list = []
            self.point_list = []
            self.time_list = []
            self.radius_list = []
            self.complexity_list = []
            self.alpha_list = []
        while True:
            count += 1
            j += 1

            if points > int(n):
                if to_print: print('\nWarning: max iterations reached\n')
                break
            if count > max_skips:
                break

            if self.debug:
                self.count_list.append(count)
                self.point_list.append(points)
                self.time_list.append(time.time() - start)

            if locations is not False:
                loc = locations[points]  # TODO, check if this is used and delete if not
            else:
                loc = [int(random() * w), int(random() * h)]
            # compare with probability matrix
            if random() < self._testProbability(loc, use_coverage):
                if use_gradient:
                    complexity = self.array_complexity[(int(loc[1] / self.params['net_factor']),
                                                       int(loc[0] / self.params['net_factor']))]
                else:
                    complexity = self._getComplexityOfPixel(
                        self.array, loc, int(d * constant / 2), use_complexity)
                r = self._getRadiusFromComplexity(d, power, constant, min_size, complexity)
                self._plotColorPoint(loc, r, use_transparency=use_transparency,
                                     alpha_fcn=alpha_fcn, mask=True)
                if self.debug:
                    self.radius_list.append(r)
                    self.complexity_list.append(complexity)
                points += 1
                count = 0

        end = time.time()
        if to_print:
            print('done...took %0.2f sec for %d points' % ((end - start), points))

    def make(self, setting='balanced', **kwargs):
        """Makes plots with present settings and optional arguments

            Possible setting values:
                balanced, fine, ultrafine, coarse, uniform

        """

        if setting not in self.settings.keys():
            raise Exception("Invalid setting")

        frame_is_top = (inspect.currentframe().
                        f_back.f_code.co_name == '<module>')
        to_print = True if self.debug & frame_is_top else False
        start = time.time()

        self.plotRecPoints(**self.settings[setting]['PlotRecPoints'])
        self.plotRandomPointsComplexity(**self.settings[setting]['PlotPointsComplexity'])

        if to_print: print('done in %0.2f seconds' % (time.time() - start))

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

    def _makeComplexityArray(self, sigma1, sigma2, multiplier=.8):

        h = self.array.shape[0]
        w = self.array.shape[1]
        d = (h**2 + w**2)**0.5
        gradient = ndimage.gaussian_gradient_magnitude(self.array.sum(axis=2), sigma=sigma1)
        gradient_max = ndimage.maximum_filter(gradient, size=d * sigma2)

        self.array_complexity = 1 - gradient_max / gradient_max.max() * multiplier - (1 - multiplier)

    def _testProbability(self, loc, use_coverage):
        if use_coverage:
            location = (loc[0] + self.border, loc[1] + self.border)
            probability = max(1 - self.out_coverage.getdata().getpixel(location) / 255, 0)

        else:
            probability = 1

        return probability

    def _plotIterations(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(self.count_list)
        ax2.plot(self.time_list, self.count_list)
        ax1.set_ylabel('Consecutive non-plots')
        ax1.set_xlabel('Iteration')
        ax2.set_xlabel('Seconds')
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        f.set_figwidth(10)

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(self.point_list)
        ax2.plot(self.time_list, self.point_list)
        ax1.set_ylabel('Plotted points')
        ax1.set_xlabel('Iterations')
        ax2.set_xlabel('Seconds')
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        f.set_figwidth(10)
        plt.show()

    def _plotBubbleSize(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.hist(self.radius_list, bins=100, orientation='horizontal')
        ax2.plot(self.radius_list, '.', alpha=min(0.5, 3e4 / len(self.point_list)))
        ax1.set_ylabel('Radius')
        ax1.set_xlabel('Count')
        ax2.set_xlabel('Iteration')
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        f.set_figwidth(10)
        plt.show()

    def _plotComplexity(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.hist(self.complexity_list, bins=100, orientation='horizontal')
        ax2.scatter(x=self.radius_list, y=self.complexity_list, s=0.5)
        ax1.set_ylabel('Complexity')
        ax1.set_xlabel('Count')
        ax2.set_xlabel('Radius')
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        f.set_figwidth(10)
        plt.show()

    def _plotAlpha(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.hist(self.alpha_list, bins=100, orientation='horizontal')
        ax2.plot(self.alpha_list, '.', alpha=min(0.5, 3e4 / len(self.point_list)))
        ax1.set_ylabel('Alpha')
        ax1.set_xlabel('Count')
        ax2.set_xlabel('Iteration')
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        f.set_figwidth(10)
        plt.show()

    def save_out(self, location='./', name=None, **kwargs):
        """Saves output image to location (directory). If no name
        is given, uses name of source image plus optional suffix and prefix.

            Optional kwargs:
                suffix='pointillized'
                    optional suffix to add to file
                prefix=''
                    optional prefix to add to file
        """

        suffix = kwargs.get('suffix', 'pointillized')
        prefix = kwargs.get('prefix', '')

        if os.path.isdir(location) is not True:
            os.makedirs(location)

        if name is None:
            name = self.filename.split('/')[-1]
            file_to_save = os.path.join(location, (prefix + name + ' - ' + suffix + '.png'))
        else:
            file_to_save = os.path.join(location, name)

        self.out.save(file_to_save)
