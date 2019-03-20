**Just realized detection part. Updating...**

## Requirements

- python2.7
- pytorch>=0.4
- numpy

## Quickstart

> You can use the optional argument `-h` to see more arguments you can set.
- Generate the dataset: `$ python readData.py`
- Train the model:  `$ python train.py` . Set some parameters as you want.
- Test:  `$ python activities_prediction_srnn.py` . The outputs are the detection/prediction accuracy.

## Neural network

Based on the original codes and the paper, the structure of this neural network is following:
![](https://i.imgur.com/WfI2YM7.png)
