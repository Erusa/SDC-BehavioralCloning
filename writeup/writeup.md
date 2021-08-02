# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set, adjusting the learning parameters
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* loss.png with training and validation learning curve
* run1.mp4 a video recording of your vehicle driving autonomously around the track for one lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of:
- Preprocessing of the Images: normalization through Keras lambda layer and cropping
- Convolutional Neural Network:
based on Nvidia CNN (+ Droput after second convolutional layer: 
[image1]: cnn-architecture.png "cnn"
![alt_text][image1]

https://developer.nvidia.com/blog/deep-learning-self-driving-cars/


```python
model = Sequential()
#preprocessing
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1))
#model.add(Flatten(input_shape = (160,320,3)))
model.add(Conv2D(24,(5,5), subsample =(2,2), activation="relu")) #(filters, kernel size,...)
model.add(Conv2D(36,(5,5), subsample =(2,2), activation="relu"))
#model.add(Dropout(0.5))
model.add(Conv2D(48,(5,5), subsample =(2,2), activation="relu"))
model.add(Dropout(0.5))
#model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))

model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 111). Also Epochs and Batch size was chossen so that not overfitting occurs.

[image2]: loss.png "loss"
![alt_text][image2]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer and mse, so the learning rate was not tuned manually.


```python
#creating a Regresssion Model (not classification model) --> mse instead cross entropy       
model.compile(loss = 'mse', optimizer = 'adam')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, data augmentation flipping the images, and also saving a little extra data from simulator. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

To get a good model, first I used the provided dataset from Udacity and preprocessed the images. After that, I use a simple CNN and see the performance. Then, I augmented the data using left, right and flipped images. After that, I tried with Model used from NVIDIA. Because, I notice that best performance is reached by 2-3 Epochs, I tried to avoid underfitting by adding 1-2 Dropout layer and compare performance, deciding to use only one Dropout Layer. I notice that the car always go outside the way at the same position, and because of that, I save extra training information on that position. For final adjustments, I played with different batch sizes by training.

#### 1. Creation of the Training Set

To augment the data sat,:
- I flipped images and angles thinking that this would, because it would reduce left car bias.
- I used right and left pictures with a modification of the steering angle. This would give extra data to the car, how to drive when car is not driving on the center of the road. I added to correction factor to the steering angle, because I consider that the angle changes depending on the curvature of the road. Following code was used:


```python
#steering correction
k0 =0.15 #0.15
k1 = 0.1
        
steering_left = (1 - np.sign(steering)*k1)*steering + k0
steering_right = (1+ np.sign(steering)*k1)*steering - k0
```

I finally randomly shuffled the data set and put 20% of the data into a validation set. 


#### 1. Preprocessing
Data is normalized and cropped. Normalization allows NN to optimize faster and better. Cropping erase unnecessary Background information, which tend to mislead CNNs. 


#### 2. NN Model
My first step was to use a convolution neural network model similar to the CNN from NVIDIA. I thought this model might be appropriate because it was designed specuially for autonomous driving. It also has more CNN than LeNet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added a dropout layer and adjust the number of Epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 3. Tunning and Adding Extra Data

The final step was to run the simulator to see how well the car was driving around track one. There was one spots where the vehicle fell off the track... 

[image3]: Away.png "problem"
![alt_text][image3]

To improve the driving behavior in this case, I drive one on this spot and use the information for training. After that, car drive well this spot but fell off nearby. It also use generator, just in case I need to record more data, but it was at the end not necessary.

I considered to use an even powerfulll NN Model, however, I tunned again the learning parameters, specifically the Batch Size and the loss decreased. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Here 's a link to result.

[Watch the video](run1.mp4 "Video")
(see in Repository run1.mp4, if link is not working)




```python

```
