# Pointillism
Image manipulation for various "pointillism" effects, built from scratch in Python. Image handling courtesy of Pillow. 

Under active development. Please let me know if you have feedback.

## Examples and usage
See detailed example usage in `Pointillism Example Usage.ipynb` [notebook](https://github.com/tsennott/pointillism/blob/master/Pointillism%20Example%20Usage.ipynb)

See example images in google [album](https://photos.app.goo.gl/Dv6IObEJnsxKI3bn1)

## Deployment
Currently deployed via Django on AWS Elastic beanstalk at [pointillizer.com](http://www.pointillizer.com)

## Installation
Install with `pip install pointillism`

## Modules
* `pointillism.image`: core image methods for pointilling
* `pointillism.pipeline`: methods for making image manipulation pipelines and gifs
* `pointillism.batch`: batch processing of images or gifs in parallel
* `pointillism.movies`: (coming soon)

## Basic usage
Making a pointillized image with default presets 
```
# Import 
import pointillism as pt

# Initialize
point = pt.image(image_location)

# Render
# optional setting can be 'balanced', 'fine', 'ultrafine', 'coarse', or 'uniform'
point.make() 

# Save
# or point.display() if using IPython
point.save_out(image_location) 
```

