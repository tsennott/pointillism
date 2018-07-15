Image manipulation for various "pointillism" effects, built from scratch in Python. Image handling courtesy of Pillow. 

Under active development. Please let me know if you have feedback.

## Web app version
Currently deployed at [pointillizer.com](http://www.pointillizer.com), check it out to try out the effect or see examples.

## Examples and usage
See examples and detailed usage in [example notebook](https://github.com/tsennott/pointillism/blob/master/Pointillism%20Example%20Usage.ipynb)

See other example images in [google album](https://photos.app.goo.gl/Dv6IObEJnsxKI3bn1)

## Installation
Install with `pip install pointillism`

## Modules
* `pointillism.image` - core image methods for pointillizing
* `pointillism.pipeline` - methods for making image manipulation pipelines and gifs
* `pointillism.batch` - batch processing of images or gifs in series (parallel coming soon)
* `pointillism.movies` - coming soon!

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

