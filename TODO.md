# TODO

Current
* Get mask working for transparent images
  * Implement alpha properly using global settings
  * Make prob function cutoff also a passable parameter
* Remove `_make_ProbabilityMask`, implement gradient version into `plotRandomPointsComplexity`
* Split off the experiments into a new notebook (maybe)
  * Save some example images, including point cloud

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


