---
title: Fine Tuning
layout: single 
sidebar: 
    nav: "navigation"
toc: true
---

The Mask R-CNN model was initially trained on full FOV images (2048 x 2048 pixels). Having segmented and tracked full time series with this model, crops of cells just before and just after cell death could be taken and manually labelled, allowing the model to be further fine tuned to better identify the exact frame of cell death. Additionally, crops of clusters which were often incorrectly segmented could be manually segmented and used for fine tuning. From this additional data consisting of {N} images and {K} training and validation sets were created with an 80/20 splits for K-Fold cross validation.

|   | No Fine Tuning                              | Fine Tuning                                 |
| --- | ----------------------------------------- | ----------------------------------------- |
| Confusion Matrix (crops containing 0 or 1 cells only) - median (5th-95th percentile) | ![CM]({{ site.baseurl }}/images/cmnofinetune.png) |![CM]({{ site.baseurl }}/images/cmfinetune.png) |
| Classification Accurcay | 0.74 ± 0.02               | 0.90 ± 0.02            |
| Example Cluster Segmentation (Green - Manual, Red - Mask R-CNN, Yellow - Overlap) | ![Example]({{ site.baseurl }}/images/cluster0.png) ![Example]({{ site.baseurl }}/images/cluster1.png)  | ![Example]({{ site.baseurl }}/images/cluster0f.png) ![Example]({{ site.baseurl }}/images/cluster1f.png) |

Precision recall curves for crops containing clusters are also shown:
![Precicions Recall Curve]({{ site.baseurl }}/images/cluster_prec_recall.png)

Various methods of fine tuning were tested:
- Fine tuning ROI head (freezing weights in all other layers)
- Fine tuning classfication head only (freezing weights in all other layers)
- Fine tuning whole model

FIne tuning the whole model was the only method found to show substantial improvement in performance. {Get hyperparameters file}