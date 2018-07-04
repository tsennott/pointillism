from unittest import TestCase
import os
import shutil

import pointillism as pt


class TestImage(TestCase):

    def setUp(self):
        self.point = pt.image('../media/images/pfieffer.jpg', border=0)
        self.directory = 'test_image_results'

    def tearDown(self):
        try:
            shutil.rmtree(self.directory)
        except Exception as e:
            pass

    def test_resize_and_crop(self):

        # resize
        self.point.resize(ratio=0.5, min_size=1000)
        self.point.crop(aspect=(1000, 500), resize=True)

        # test
        self.assertEqual(self.point.image.size, (1000, 500), "image wrong size")

    def test_basic_make_method_and_save_options(self):

        # make
        self.point.crop(aspect=[1000, 500], resize=True)
        self.point.make('balanced')

        # test
        self.assertEqual(self.point.out.size, (1000, 500), "output image wrong size")

        # save
        self.point.save_out(location=self.directory, suffix='suffix', prefix='prefix_')
        self.point.save_out(location=self.directory, name='test_image.png')
        files = os.listdir(self.directory).sort()
        desired_files = [
            'prefix_pfieffer.jpg - suffix.png',
            'test_image.png'
        ].sort()

        # test
        self.assertEqual(files, desired_files, "file names don't match")


class TestPipeline(TestCase):

    def setUp(self):
        self.point = pt.pipeline('../media/images/pfieffer.jpg', border=0)
        self.directory = 'test_image_results'

    def tearDown(self):
        try:
            shutil.rmtree(self.directory)
        except Exception as e:
            pass

    def test_basic_make_gif_method(self):

        size = [500, 250]
        ratio = 0.1

        # make assembly and save with specific name
        self.point.make_gif(kind='assembly', location=self.directory,
                            name='animated_assembly.gif', size=size, crop=True)

        # make loop and save with default name,
        self.point.make_gif(kind='loop', location=self.directory,
                            size=size, crop=True)

        # make multiplier and save with default name with prefix, using resize instead of crop
        self.point.make_gif(kind='loop', location=self.directory,
                            size=ratio, crop=False, prefix='multiplier_')

        # get filenames
        files = os.listdir(self.directory).sort()
        desired_files = [
            'animated_assembly.gif',
            'pfieffer.jpg - pointillized.gif',
            'multiplier_pfieffer.jpg - pointillized.gif',
        ].sort()

        # test
        self.assertEqual(files, desired_files, "file names don't match")
