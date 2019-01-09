import csv
import cv2
from scipy import ndimage
import numpy as np
import matplotlib.image as mpimg

lines = []
# reading all the info from csv file 
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# images -> Images taken by left, right, and center camera
images = []
# measurements -> angle
measurements = []

for line in lines[1:]:
    # This roop is for the sake to take left, right, and center images.
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    #Fliping images and steering measurements
    #A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement. 
    images.append(cv2.flip(image,1))
    measurements.append(measurement*-1.0)
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()

#Nomalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))  

#Cropping Image
model.add(Cropping2D(cropping=((70,25), (0,0))))

#Use of model provided by nvidia 5 convolutional and 3 fully connected nerual network.
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
