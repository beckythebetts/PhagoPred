# Overview
Phagocytosis, the engulfment and elimination of particles by cells, is a fundamental component of the innate immune response. Bacteria have evolved various strategies to survive and escape phagocytosis, some of which are well understood, while others remain unidentified.

**Aim**: Identify patterns in phagocyte morphology and dynamics that indicate successful or failed phagocytosis.

Using time-lapse, two-channel microscopy, we image macrophages (professional phagocytes) and Staphylococcus aureus bacteria to capture phagocytosis events.
Segmentation via Mask R-CNN provides cell outlines.
Tracking is performed with a linear sum assignment problem approach 
We extract multivariate time series features describing the morphology, movement, and interactions of cells involved in each phagocytosis event. 
To classify phagocytosis events as successful or failed, we plan to evaluate several machine learning approaches, balancing accuracy and interpretability. 
![Phagocytosis](docs\images\phago_11_3.gif)