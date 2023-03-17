# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:04:07 2022

@author: Sanju
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import RMSprop

(x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.imshow(x_train[0])
plt.show()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype('float64')
x_test=x_test.astype('float64')
x_train/=255
x_test/=255
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0,2))
model.add(Dense(512,activation='relu',))
model.add(Dropout(0,2))
model.add(Dense(512,activation='relu',))
model.add(Dropout(0,2))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1,validation_data=(x_test,y_test))