# Phase 13: Error analysis (E010, test set, 15 errors in 2,400 images)

- Most errors (8/15) are leaves with mild or no visible symptoms, so the label is hard to support from the image.
- 4/15 are large dead-tissue lesions that look alike across diseases (Early/Late Blight confusion); 1/15 many small spots (Early Blight vs Bacterial Spot); 2/15 unusual framing or leaf shape.
- Early Blight is the hardest class (4/150 errors); all 3 Target Spot errors were predicted as Spider Mites.
- Class size is associated with error rate (three of the four smallest training classes have the highest error rates) but Tomato Mosaic Virus is an exception; class weights did not help (E008).
- Errors lean toward softer images within their class (median sharpness percentile 23); our sharpness measure is texture-dependent, so this may reflect low symptom texture rather than blur. Weak evidence (n = 15).
- All 15 test errors have confidence below 0.95; 44/2,400 test images are below 0.95.
- Label correctness of the most confident errors was not verified by an expert.
- No model change and no relabeling or removal of images was made (decision 36).