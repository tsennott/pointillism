# TODO

## Current

* Features and Github
  * Clean up working notebook and add more complex methods in the main notebook
  * Consider sphynx documentation?
* `pointillism.image`
  * add sepia setting
  * possible dummy base class for easier readability?
* `pointillism.bulk`
  * implement multiprocessing add to example file
* `pointillism.movie`
  * make this! with multiprocessing
* Later
  * organize `image` params into dicts (like radius_list, etc etc)
  * for `image.make()` add ability to override settings
  * change `frame_is_top` to verbose being passed in?
* Pointillizer.com
  * (later) add django tests
  * (later) fix scipy issue
  * (later) add image detail page with source image and add/edit/delete
    * must change data model to allow source files with child result files

## Deployment notes
* For deployment with correct markdown to PyPI
  * `python setup.py sdist`
  * `twine upload dist/{latest version of package`
* For running tests locally
  * `coverage run --source pointillism setup.py test`
  * `coverage html`
  * `open htmlcov/index.html`
* For clearing stupid github cached badge images
  * `curl -X PURGE {url of cached badge image}`



## Algorithm notes
Notes on possible automatic coverage stop (time and implement one of these)
* Implement a stop method using a the coverage plot
  * Percentage black? Or rate of change of coverage vs attempted iteration?
    * Both would need to be evaluated only every x loops as they are expensive
* Could look at slope of time vs points or interations vs points as well, that would be much cheaper


