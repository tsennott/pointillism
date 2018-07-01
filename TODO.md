# TODO

## Current


* CLEAN UP GITHUB MASTER 


* Better resizing function for website for speed...

* Expose meta grouped build method with a few presets, make a public reset function
  * make rect a function of diagonal as well, and dial that in
  * add ability to override settings like size (e.g. for uniform)
  * handle fill=True for border=0...how? Think I've solved it but check again
  * and maybe a public debug plot grouped method
  * Also default grad_mult to 1 and adjust settings accordlingly
  * Link gradient size and plotrecpoints to these settings

* Massive cleanup, drop unnecesary methods, add lots of comments, etc
  * Change all optional args to defaulted args!!
  * organize params into dicts (like radius_list, etc etc)
  * Wrap lots of stuff in if debug, like plotting coverage and the like
  * Maybe drop PointillizePile? Or at least change names to be clearer
  * Definitely drop probablity mask stuff, 
  * Drop plotComplexityGrid and Point, in favor of displaying raw array_complexity
  * Get rid of frame is top in favor of verbose=self.verbose?

* Add explicit movie functions, and chunked movie functions, probably to second file



## Algorithm notes
Notes on possible automatic coverage stop (time and implement one of these)
* Implement a stop method using a the coverage plot
  * Percentage black? Or rate of change of coverage vs attempted iteration?
    * Both would need to be evaluated only every x loops as they are expensive
* Could look at slope of time vs points or interations vs points as well, that would be much cheaper



