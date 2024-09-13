# ECG Segmentation

This repository has implementation of ECG signal segmentation with adapted for 1D data TransUNet architecutre. The primary goal was to segment and classify heartbeats into five distinct classes using the MIT-BIH dataset. I implemented two key backbones, MobileNet and EfficientNet, tailored for 1D data. Both of them are lightweigth which is a need in real life scenarion when latency and overall complexity have to be minimized.

### TransUNet

That architecutre proposed by Chen et al. in 2021 ([Click for paper!]([Chen](https://arxiv.org/pdf/2102.04306)) is simply but effective modification of well-known U-Net arehictecutre



To-do List:
- Implement other popular CNNs in the 1D version
- Implement other architecures for segmentation
- Test on other medical time series data (e.g. EEG)
