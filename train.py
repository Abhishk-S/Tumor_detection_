import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
#from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
#from keras.utils import to_categorical


image_directory = 'datasets/'

no_tumor = os.listdir(image_directory + 'no/')
yes_tumor = os.listdir(image_directory + 'yes/')

dataset = []
label = []

for i, image_name in enumerate(no_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)


for i, image_name in enumerate(yes_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, 
          epochs=10, validation_data = (x_test, y_test), shuffle=False)

#model.load_weights(os.path.join('..','models','checkpoint'))

model.save('Tumor10Epochs.h5')



#print (x_train.shape, x_test.shape, y_train.shape, y_test.shape)
