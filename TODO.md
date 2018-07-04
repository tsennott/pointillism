# TODO

## Current

Remaining
* Github
  * Setup tests and CI
  * Clean up working .ipynb and make more detailed docs on more complex methods in the main example
* `pointillism.bulk`
  * implement MP in Bulk processing and add to example file
* `pointillism.movie`
  * make this! with multiprocessing
* Later
  * organize `image` params into dicts (like radius_list, etc etc)
  * for `image.make()` add ability to override settings
  * change `frame_is_top` to verbose being passed in?
* Pointillizer.com
  * convert to new modularized code
  * (later) fix scipy issue
  * (later) change data model for website to allow source files with child result files
  * (later) 


## Algorithm notes
Notes on possible automatic coverage stop (time and implement one of these)
* Implement a stop method using a the coverage plot
  * Percentage black? Or rate of change of coverage vs attempted iteration?
    * Both would need to be evaluated only every x loops as they are expensive
* Could look at slope of time vs points or interations vs points as well, that would be much cheaper



