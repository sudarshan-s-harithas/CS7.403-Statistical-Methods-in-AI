## This code is a part of the Ablation study section of the SMAI Project at IIIT Hyderabad 
## It used to Train a network on the CIFAR100 dataset 


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D , ZeroPadding2D
import numpy as np
import pickle
import keras 
from keras.layers.core import Lambda
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
tf.compat.v1.disable_eager_execution()

from skimage import io
from skimage.transform import resize
# from scipy.misc import imread, imresize
from matplotlib import pyplot as plt
# import cv2


tf.keras.datasets.cifar100.load_data(label_mode="fine")
tf.debugging.set_log_device_placement(True)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
# y_train = to_categorical(y_train)
x_train = x_train/255.0
x_test = x_test/255.0

with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):

	
	model = Sequential()

	## Block 1

	model.add(Conv2D(8, (3, 3), input_shape=x_train.shape[1:] , padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	## Block 2


	model.add(Conv2D(32, (3, 3 )  , padding ='same' )   )
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	## Block 3

	model.add(Conv2D(64, (3, 3 )  , padding ='same' )   )
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	## Block 4

	model.add(Conv2D(128, (2, 2 )  , padding ='same' )   )
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	## Block 5

	model.add(Conv2D(256, (2, 2 )  , padding ='same' )   )
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2) , padding = 'same' ) )


	model.add(Conv2D(256, (2, 2 )  , padding ='same' )   )
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2) , padding = 'same' ))
	model.add(Dropout(0.2))


	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(512))
	model.add(Dropout(0.2)) 
	model.add(Dense(100))
	model.add(Activation('softmax'))

	
	model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	model.fit(x_train, y_train , batch_size=20, epochs=50,validation_split=0.3)

	# model.save("CIFAR100_model")
