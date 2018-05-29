# TODO

Really current
* Fix border issue (DONE)
* See below, how make automatic cutoff for coveerage
* Change all optional args to defaulted args
* MAssive cleanup
* Possibly massive refactor or from scratch

Current
* Finish mask and transparency work
  * Implement alpha properly using global settings, including setting the distribution
  * Make prob function cutoff also a passable parameter
  * HOW DEAL WITH PLOT REC POINTS FOR CUTOFF??
* Implement gradient version into `plotRandomPointsComplexity`
  * DONE, need to pass parameters for sigma somehow
  * Point count is lower for automatic cutoff, need to increase contrast maybe
* Make the debug plots a function
* Delete `_makeProbabilityMask` and the complexity plotting functions, or kick into experimental subclass
* Split off the experiments into a new notebook (maybe)
  * Save some example images, including point cloud, the edge images and the edgeless images

Cleanup
* 


Optimization
* Use gradient function instead of complexity of pixel
  * OK, WORKING BUT NEED TO TEST QUALITY
* Consider better way to distribute dots
  * Use a coverage mask to only pick uncovered areas?
    * DONE, working great
  * Segment into sections to do the randomness in?

Coverage stop (time and implement one of these)
* Implement a stop method using a partially transparent (uniform) black coverage test
  * Percentage black? Or rate of change of coverage vs attempted iteration?
    * Both would need to be evaluated only every x loops as they are expensive
  * Or number of misses?
* Analytical a priori method for comaring coverage?


