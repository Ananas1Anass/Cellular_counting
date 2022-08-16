
## Goal
In this repository, we implement a fully convolutional regression  networks (FCRNs) approach for regression of a density map in order to get density map of cell pictures for biomedical goals.
## Generation of Database 

Generating squares randomly positionned in a 100x100 image.
First step is to randomly position the squares, then get the coordinates of the center of squares, and apply Gaussian filter over them for ground truth.
Example from dataset : 

![alt-text-1](https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/image_in_7_22.png "Input example") ![alt-text-2](https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/image_gth_4_17.png "Ground truth example")



## Network description 
![alt text](https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/FCRN.png)

## Test on example : 

![alt-text-1](https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/image_4_35.png "Input example") ![alt-text-2]((https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/4_35(1).png "Density map")


![alt-text-1](https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/image_8_9.png "Input example") ![alt-text-2](https://github.com/Ananas1Anass/Cellular_counting/blob/main/Images/8_9(1).png "Density map")


## 🔗 Links
- [ Paper : Microscopy cell counting and detection with fully convolutional regression networks)](https://www.tandfonline.com/doi/abs/10.1080/21681163.2016.1149104?journalCode=tciv20 "Microscopy cell counting and detection with fully convolutional regression networks")

