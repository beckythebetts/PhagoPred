---
title: Cell Death Detection
layout: single 
sidebar: 
    nav: "navigation"
toc: true
---

In order to identify when a macrophage dies, the Mask R-CNN model can be trained on images containing both alive and dead cells. In general here, the moment of cell death is taken as when the cell looses it's structure and the membrane can no longer be clearly defined. (Cells often ball up and exhibit membrane blebbing before this occurs.)
![Images of cell death]({{ site.baseurl }}/images/CellDeathMontage.png)
KFold cross validation on the training images (taking one image out for validation per fold), suggested that this performed reasonably:
![Preciison recall curves for alive and dead macropahages]({{ site.baseurl }}/images/maskrcnn_performance.png)

These classifications could then be used to identify the frame of cell death. It was assumed there would be some noise in the classifications, so alive cells were asigned a score 1, and dead cells 0. These scores were then smoothed for each cell over time (by taking a 5 frame rolling average) and the cell death was taken as the latest frame for which the score dropped below 0.85. The number of frames in the rolling average, and the threshold score were chsoen through a hyperparamter grid search. 
![Predicted vs true cell death frame]({{ site.baseurl }}/images/celldeathpred.png)