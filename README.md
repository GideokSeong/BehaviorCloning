# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./left.jpg "Image taken by left camera"
[image2]: ./right.jpg "Image taken by right camera"
[image3]: ./center.jpg "Image taken by center camera"
[image4]: ./flipped.jpg "flipped Image"
[image5]: ./cropped_image.jpg "Cropped Image"
[image6]: ./Model_architecture.jpg "Model Architecture Image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Reducing overfitting in the model

The model contains normalizaiton, flipping, and cropping data in order to reduce overfitting.
![alt text][image4]
![alt text][image5]

The model was trained with different testing and validation datasets which are quite lots of images taken by left, right and center camera.
![alt text][image1]
![alt text][image2]
![alt text][image3]

#### Model architecture
![alt text][image6]

My model uses 5 convolutional layers with 2*2 filter in the first three layers and 5 fully connected layers  ( line: 50-63 ).


#### Tuning model parameter

Adam optimizer was used for tuning the learning rate, so it automatically tunes learning rate.

### Conclusion

The whole model was based on Nvidia model architecture which works enough to get the car trained properly. In order to train the car, I needed to arrange the model such activities like normalization, cropping, and flipping. By doing each tuning, the car ran better. Through the step such as training data in training mode, and validation which was used by the model I created in autonomous mode, it was a good chance to understand how deep neural network works.
