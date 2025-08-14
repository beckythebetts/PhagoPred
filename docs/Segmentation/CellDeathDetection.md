---
title: Cell Death Detection
layout: single 
sidebar: 
    nav: "navigation"
toc: true
---

In order to identify when a macrophage dies, the Mask R-CNN model can be trained on images containing both alive and dead cells. In general here, the moment of cell death is taken as when the cell looses it's structure and the membrane can no longer be clearly defined. (Cells often ball up and exhibit membrane blebbing before this occurs.)
![Images of cell death]({{ site.baseurl }}/images/CellDeathMontage.png)
KFold cross validation on the training images (taking one image out for validation per fold), suggested that this performed reaosonably:
![Preciison recall curves for alive and dead macropahages]({{ site.baseurl }}/images/maskrcnn_performance.png)
These classifications coudl then be used to identify the fram eof cell death. It was assumed there would be some noise in the classifications, so alive cells were asigned a score 1, and dead cells 0. These scores were then smoothed for each cell over time (by taking a 5 frame rolling average) and the cell death was taken as the first frame for which the score dropped below 0.5. However, as shown below, this did not accurately identify cell death. Average error was 56.5 frames.
![Predicted vs true cell death frame]({{ site.baseurl }}/images/CellDeathPredictionAccuracy.png)
The training images contained both alive and dead cells, but to accurately identify the exact frame of death training data of cells just before and just after death would be required. For 50 cells, the exact frame of death was manually identified. Frames [1, 2, 3, 45, 50, 100, 150, 200] before and after this were used to create a new fine tuning dataset. Everything except the classification head of the Mask R-CNN model was frozen, and the previously trained model was fine tuned on this dataset.
![Confusion matrices before and after classification head fine tuning]({{ site.baseurl }}/images/death_confusion.png)