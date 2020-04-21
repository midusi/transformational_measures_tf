#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:44:31 2020

@author: oem
"""


import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt


class Modelo:
    
    def __init__(self,datos):
        self.modelo = ks.models.Sequential()
        self.datos=datos        
        self.epochs=10
        self.batch_size=16
        self.lr=0.002
        
        
    def resumen(self):
        self.modelo.summary()
    
    def compilar(self):
        self.modelo.compile(optimizer=ks.optimizers.SGD(lr=self.lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'],)
    
    def entrenar(self):
        self.historia = self.modelo.fit(self.datos.x_train,self.datos.y_train,epochs=self.epochs,batch_size=self.batch_size,validation_data=(self.datos.x_test,self.datos.y_test))
        
    def plot_training_curves(self):
        plt.figure()
        plt.plot(self.historia.history['accuracy'])
        plt.plot(self.historia.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show(), plt.grid()
   
    def guardar(self,ruta_nombre):
        self.modelo.save(ruta_nombre)
    
    def predecir(self,x):
        return self.modelo.predict(x)
    
    def cargar(self,ruta_nombre):
        self.modelo = tf.keras.models.load_model(ruta_nombre)

    def ejecutar_guardar(self,ruta_nombre):
        self.compilar()
        self.entrenar()
        self.guardar(ruta_nombre)
        


class ModeloSimple(Modelo):
    
    def __init__(self,datos):
        Modelo.__init__(self,datos)
        self.modelo.add(ks.layers.Conv2D(50, (3, 3), input_shape=(28, 28, 1), name='Convolutional_layer'))
        self.modelo.add(ks.layers.Activation('relu'))
        self.modelo.add(ks.layers.MaxPooling2D((2, 2), name='Maxpooling_2D'))
        self.modelo.add(ks.layers.Flatten(name='Flatten'))
        self.modelo.add(ks.layers.Dense(50, activation='relu', name='Hidden_layer'))
        self.modelo.add(ks.layers.Dense(10, activation='softmax', name='Output_layer'))
        self.datos=datos        
        self.epochs=10
        self.batch_size=16
        self.lr=0.002
        
        


class ModeloSimpleConv(Modelo):
    
    def __init__(self,datos,conv_filters=32,kernel_size=3,activation="relu",pool_size=2):
        Modelo.__init__(self,datos)
        
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = datos.input_shape
        self.pool_size = pool_size

        self.modelo.add(ks.layers.Conv2D(self.conv_filters,kernel_size=self.kernel_size, input_shape=self.input_shape, name='Conv1'))
        self.modelo.add(ks.layers.Activation(self.activation))
        
        self.modelo.add(ks.layers.Conv2D(self.conv_filters,kernel_size=self.kernel_size, input_shape=self.input_shape, name='Conv2'))
        self.modelo.add(ks.layers.Activation(self.activation))
        
        self.modelo.add(ks.layers.MaxPooling2D(self.pool_size, name='MaxPool1'))
       
        self.modelo.add(ks.layers.Conv2D(self.conv_filters*2,kernel_size=self.kernel_size, input_shape=self.input_shape, name='Conv3'))
        self.modelo.add(ks.layers.Activation(self.activation))
      
        self.modelo.add(ks.layers.Conv2D(self.conv_filters*2,kernel_size=self.kernel_size, input_shape=self.input_shape, name='Conv4'))
        self.modelo.add(ks.layers.Activation(self.activation))
      
        self.modelo.add(ks.layers.MaxPooling2D(self.pool_size, name='MaxPool2'))

        self.modelo.add(ks.layers.Conv2D(self.conv_filters*4,kernel_size=self.kernel_size, input_shape=self.input_shape, name='Conv5'))
        self.modelo.add(ks.layers.Activation(self.activation))
      
        self.modelo.add(ks.layers.Flatten(name='Flatten'))
        
        ndense1 = datos.input_shape[0]*datos.input_shape[1] * self.conv_filters * 4
        ndense2 = self.conv_filters
        self.modelo.add(ks.layers.Dense(ndense1, activation=self.activation, name='Dense1'))
        self.modelo.add(ks.layers.Dense(ndense2, activation=self.activation, name='Dense2'))
        self.modelo.add(ks.layers.Dense(10, activation='softmax', name='Softmax'))
        
        self.datos=datos        
        self.epochs=1
        self.batch_size=16
        self.lr=0.002














































