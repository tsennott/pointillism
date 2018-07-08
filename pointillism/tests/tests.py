from unittest import TestCase
import os
import shutil
import pointillism as pt
from mock import patch


class TestImage(TestCase):

    def setUp(self):
        self.point = pt.image('tests/images/pfieffer.jpg', border=0, debug=True)
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

    def test_generate_random_points(self):

        n = 1000
        # resize
        locations = self.point._generateRandomPoints(n)
        self.assertEqual(len(locations), n, "wrong number of points")

    def test_basic_make_method_and_save_options(self):

        # make
        self.point.crop(aspect=[1000, 500], resize=True)
        self.point.make('balanced')

        # test
        self.assertEqual(self.point.out.size, (1000, 500), "output image wrong size")

        # save
        self.point.save_out(location=self.directory, suffix='suffix', prefix='prefix_')
        self.point.save_out(location=self.directory, name='test_image.png')
        files = sorted(os.listdir(self.directory))
        desired_files = sorted([
            'prefix_pfieffer.jpg - suffix.png',
            'test_image.png'
        ])

        # test
        self.assertEqual(files, desired_files, "file names don't match")

    @patch("matplotlib.pyplot.show")
    def test_display_and_plot_methods_plus_alpha_and_nongradient(self, mock_show):

        # test enhance
        self.point.enhance('contrast', 1.1)
        self.point.enhance('sharpness', 1.1)
        self.point.enhance('color', 1.1)

        # make, including use of deprecated _getComplexity method
        self.point.crop(aspect=[1000, 500], resize=True)
        self.point.plotRecPoints(fill=True)
        self.point.plotRandomPointsComplexity(
            n=1e4, constant=0.012, power=3,
            use_transparency=True, use_gradient=False, min_size=0.002)

        # display
        self.point.display()
        self.point.display('original')
        self.point.display('coverage')
        self.point.display('gradient')

        # debug plot methods
        self.point._plotComplexity()
        self.point._plotIterations()
        self.point._plotBubbleSize()
        self.point._plotAlpha()

    def test_colormap(self):

        # test colormap
        self.point.colormap()
        self.point.colormap(setting='noir')
        self.point.colormap(setting='b&w')


class TestPipeline(TestCase):

    def setUp(self):
        self.point = pt.pipeline('tests/images/pfieffer.jpg', border=0, debug=True)
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


class TestBatch(TestCase):

    def setUp(self):
        self.point = pt.batch('tests/images', border=0, debug=True)
        self.directory = 'test_image_results'

    def tearDown(self):
        try:
            shutil.rmtree(self.directory)
        except Exception as e:
            pass

    def test_batch_images(self):

        # make images and test showing them
        self.point.new_queue()
        self.point.add_to_queue(self.point.crop, {'aspect': [1000, 500], 'resize': True}, 1)
        self.point.add_to_queue(self.point.make, {'setting': 'balanced'}, 1)
        self.point.run_pile_images(location=self.directory, suffix='bulk')
        self.point.display()

        # get filenames
        files = sorted(os.listdir(self.directory))
        desired_files = sorted([
            'IMG_0116.jpg - bulk.png',
            'pfieffer.jpg - bulk.png',
        ])

        # test
        self.assertEqual(files, desired_files, "file names don't match")

    def test_batch_gifs(self):

        # make asssembly gifs
        self.point.new_queue()
        self.point.add_to_queue(self.point.crop, {'aspect': [1000, 500], 'resize': True}, 1)
        self.point.add_to_queue(self.point.make_gif, {'kind': 'assembly',
                                                      'location': 'media/gifs',
                                                      'bulk': True}, 1)
        self.point.run_pile_gifs(location=self.directory, save_steps=True,
                                 suffix='bulk_assembly')

        # make multiplier gifs
        self.point.new_queue()
        self.point.add_to_queue(self.point.crop, {'aspect': [1000, 500], 'resize': True}, 1)
        self.point.add_to_queue(self.point.make_gif, {'kind': 'multiplier',
                                                      'location': 'media/gifs',
                                                      'bulk': True}, 1)
        self.point.run_pile_gifs(location=self.directory, save_steps=True,
                                 suffix='bulk_multiplier')

        # get filenames
        files = sorted(os.listdir(self.directory))
        desired_files = sorted([
            'IMG_0116.jpg - bulk_assembly.gif',
            'IMG_0116.jpg - bulk_multiplier.gif',
            'pfieffer.jpg - bulk_assembly.gif',
            'pfieffer.jpg - bulk_multiplier.gif',
        ])

        # test
        self.assertEqual(files, desired_files, "file names don't match")
