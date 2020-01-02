# SpeedChallenge-Solution


This is the repository for the solution to car speed estimation challenge from Commai found here:
https://github.com/commaai/speedchallenge

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python libraries to be installed 

```
pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision
pip install Pillow==5.3.0
```
The video files are stored using GitLab LFS

### Model Training and Evaluation
Steps:
1: Split the data into 80% training set and 20% validation set to measure the performance after each epoch.
2. Mean Squared Error (MSE) as a loss function to measure how close the model predicts to the car's speed to the ground truth given in the training video for each input frame.
3. Adaptive Moment Estimation (Adam) Algorithm minimize to the loss function.

### Model Training:

