# TODO


Cleanup
* Properly implement prob test with test for the existence of a mask (or two loops?)

Optimization
* Use gradient function instead of complexity of pixel
  * OK, WORKING BUT NEED TO TEST QUALITY
* Consider better way to distribute dots
  * Use a coverage mask to only pick uncovered areas?
  * Segment into sections to do the randomness in?

Coverage stop (time and implement one of these)
* Implement a stop method using a partially transparent (uniform) black coverage test
  * Percentage black? Or rate of change of coverage vs attempted iteration?
    * Both would need to be evaluated only every x loops as they are expensive
  * Or number of misses?
* Analytical a priori method for comaring coverage?


