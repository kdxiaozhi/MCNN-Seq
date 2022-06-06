# Code for Deeply synergistic optical and SAR time series for crop dynamic monitoring
This is the prototype code for MCNN-seq that publish in the RSE journal (Deeply synergistic optical and SAR time series for crop dynamic monitoring). Please cite the following paper if you find to be useful.

```
@article{ZHAO2020111952,
title = {Deeply synergistic optical and SAR time series for crop dynamic monitoring},
journal = {Remote Sensing of Environment},
volume = {247},
pages = {111952},
year = {2020},
}
```

How to use it?
## Step 1 read the configs
check the configs.py for inputs, depth, units, outputs, etc. Also, be aware of the length of the S1 and S2 input time-series.

## Step 2 run train.py
The train.py starts with dataset parsing which includes training and testing datasets.
Then, the 1D-CNN was followed to filter the time-series for more robust feature generation.
After that, the LSTM was also included for contextual information extraction in the temporal domain.
Finally, the model can be saved as offline files.

## Step 3 run predict.py
Suppose you have the well-trained model to transfer "SAR time-series" into "optical time-series", then you can do it by running this file.

## TIPS: Use your own dataset?
Attend to the dataset.py file, where you can construct your own dataset for training and testing. 
