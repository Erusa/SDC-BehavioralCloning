import csv
import cv2
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import ceil
from random import shuffle

#saving the lines from csv in a list
lines = []
with open('data/driving_log.csv') as csvfile:
       reader = csv.reader(csvfile)
        #to avoid the head
       next(reader)
       for line in reader:
        lines.append(line)
        
train_samples, validation_samples = train_test_split(lines, test_size=0.2)       

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            steerings = []
            
            for line in batch_samples:            
                image_path = line[0]
                image_left_path = line[1]
                image_right_path = line[2]
                #saving the path name of each image on a list
                #actual path: IMG/center_...
                filename = image_path.split('/')[-1]
                filename_left = image_left_path.split('/')[-1]
                filename_right = image_right_path.split('/')[-1]
    
                #print(filename)
                current_path =  'data/IMG/' + filename
                current_path_left =  'data/IMG/' + filename_left
                current_path_right =  'data/IMG/' + filename_right
    
                #image = cv2.imread(current_path)
                image = ndimage.imread(current_path)
                image_left = ndimage.imread(current_path_left)
                image_right = ndimage.imread(current_path_right)
    
                #images.append(image)
                images.extend([image,image_left, image_right])
    
                #saving the throttle of each line
                #steering goes from -1 to 1, 
                steering = float(line[3])
                #steerings.append(steering)
    
                #steering correction
                k0 =0.15 #0.15
                k1 = 0.1
        
                steering_left = (1 - np.sign(steering)*k1)*steering + k0
                steering_right = (1+ np.sign(steering)*k1)*steering - k0
    
                #steering_left = steering + k0
                #steering_right = steering - k0
    
                #saving steerings
                steerings.extend([steering, steering_left, steering_right])
    
            #augmeting images
            augmented_images, augmented_steerings = [], []

            for image, steering in zip(images, steerings):
                augmented_images.append(image)
                augmented_steerings.append(steering)
                augmented_images.append(cv2.flip(image,1))
                augmented_steerings.append(steering*-1.0)
        
    
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_steerings)
            yield sklearn.utils.shuffle(X_train, y_train)
            #print(X_train.shape)
            
#Set our batch size #default=32
batch_size=16#32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
#Convolution2D = Conv2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

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

#creating a Regresssion Model (not classification model) --> mse instead cross entropy       
model.compile(loss = 'mse', optimizer = 'adam')
#this sentence train the model, also split the model, and if not indicated, it uses 10 epocs
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)
history_object =model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=20, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
plt.savefig('loss.png')





    