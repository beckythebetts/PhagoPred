---
title: Tracking
layout: single
sidebar: 
    nav: "navigation"
toc: true
---
`PhagoPred.tracking.tracker.Tracker()`
Match cells between each frame to form tracks.
# Find Tracklets
Between each pair of consecutive frames, pair cells in order to minimise the total distance between all cell pairs.
For each time step:
1. `Tracker().get_cell_info()`
    Find centroids (average x, and y coordinates) of each cell, see [Feature Extraction](/FeatureExtraction/).
    
2. `Tracker().frame_to_frame_matching()`
    Create 'cost matrix' of distances between each possible cell pairing. Apply Jonker-Volgenant algorithm to match cells in order to minimise sum of distances between all cells. (Discard matching if distance is below `SETTINGS.MAXIMUM_DISTANCE_THRESHOLD`)

3. `Tracker().apply_lut()`
    Update the stored segmentation masks and cell datasets to assign matched cells the same cell index, see [Dataset Structure](/DatasetStructure/).

# Join Tracklets
`Tracker().join_tracklets()`

Cell segmentations may be missing/wrong in some frames, so the tracklets formed above are matched up again using the Jonker-Volgenant algorithm to minimise distances. The distances between the start and end of each tracklet are used to form the cost matrix, provided that the `start_frame` - `end_frame` is between 0 and `SETTINGS.FRAME_MEMORY`.

Again the segmentation masks and cell datsets are updated using `Tracker().apply_lut()`.

Next: [Feature Extraction](../FeatureExtraction/)