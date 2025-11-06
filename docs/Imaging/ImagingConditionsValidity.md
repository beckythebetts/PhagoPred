---
title: Imaging Conditions Validity
layout: single 
sidebar: 
    nav: "navigation"
toc: true
---
Fluorescence imaging may lead to photoxicity in macrophages, mostly due to the generation of ROS. Therefore, various fluorescence illumination exposures where tested in an attempt to balance maximising the signal intensity of the labelled *S. Aureus* bacteria against minimising the phototoxic affects on J774 macrophages.
# Fluorescent signal
![Comparison of signal for different exposures]({{ site.baseurl }}/images/exposure_signals.png)

$$SNR = \frac{\hat{S}}{\sigma_b}$$

$$SBR = \frac{\hat{S}}{\hat{b}}$$

$$CV = \frac{\sigma_S}{\hat{S}}$$

$\hat{S}$, $\sigma_S$ are mean and standard deviation of signal, and $\hat{b}$, $\sigma_b$ are mean and standard deviaton of background.
A background image for each exposure was taken and subtracted, then a Gaussian smoothed image ($\sigma=200$) was also subtracted.

# Macrophage Survival
For 0s, 2.5s and 5s exposures, unlabelled macrophges were imaged over a period of 67 hours. The resulting timelapses were [segmented]({{ site.baseurl }}/Segmentation), [tracked]({{ site.baseurl }}/Tracking) and the [time of death for cells identified]({{ site.baseurl }}/Segmentation/CellDeathDetection). From this a Kaplan-Meier estimator was used to estimate survival functions for each cell population. It is important to note that the KM estimator assumes independence between each sample (cell) which is not the case here due to signalling and interactions between macrophages. However, it still describes the differences in survival trends between each exposure group.
![KM curves]({{ site.baseurl }}/images/km_curves.png)
Shaded regions show 95% confidence intervals. Only cells present and alive during the first frame of imaging were included here.

# Macrophage Population Growth
The total number of alive macrophages at each frame was also calculated and fitted with a Gompertz curve. Population $N$ at time $t$ is paramaterised by the maximum poulation $K$, the time of maximum growth $t_0$ and an intial growth rate $r$. 

$$N(t) = K \exp({-\exp({r(t-t_0)}}))$$

![Gompertx fits]({{ site.baseurl }}/images/gompertz.png)
The growth rate $r$ found for each fit was compared.
![Gompertx r]({{ site.baseurl }}/images/gompertz_r.png)

# Macrophage Morphology
Finally a number of morphological and movement features were calcuated for each macorpahge at each time step. The histograms for each of these feature were compared between each exposure poulation using the Kolomogorov-Smirnov statistic, $D$, which is the maximum difference between the cumulative density functions ($F_1$, $F_2$) of each distribution. 
 
$$D = sup_x \left| F_1(x) - F_2(x)\right|$$

The median and 5th and 95th percentile values were also plotted.

![feature Histograms]({{ site.baseurl }}/images/standard_hists.png)
![feature Histograms]({{ site.baseurl }}/images/skeleton_hists.png)


