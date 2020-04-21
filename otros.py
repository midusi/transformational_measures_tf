#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:01:29 2020

@author: oem
"""


import tensorflow as tf
from tensorflow import keras as ks
import numpy as np



class Datos:
    
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = ks.datasets.mnist.load_data()
        
        d_in,x,y = self.x_train.shape
        
        self.media=np.mean(self.x_train)
        self.var=np.std(self.x_train)
        # Normalizod_in las variables de entrada
        for i in range(d_in):
            self.x_train[i,:,:]=(self.x_train[i,:,:]-self.media)/self.var
            
                
        d_in,x,y = self.x_test.shape
        
        # Normalizod_in las variables de entrada
        for i in range(d_in):
            self.x_test[i,:,:]=(self.x_test[i,:,:]-self.media)/self.var
               
        self.x_train=self.x_train.reshape(-1,28,28,1)
        self.x_test=self.x_test.reshape(-1,28,28,1)
        self.x_test_original = self.x_test
        
        self.input_shape = self.x_train.shape[1:4]
        
        
        
        

    def set_longitud_x_test(self,n):
        self.x_test=self.x_test_original[0:n,:,:,:]
        
        
        
class Transformaciones:

  def __init__(self):
      pass
  
  """
  def generar(self,imagenes,rotacion):
      for imagen in imagenes:
          yield tf.keras.preprocessing.image.apply_affine_transform(imagen,theta=rotacion)
  """  
    
  def generar(self,imagenes,rotacion):
      y = [ tf.keras.preprocessing.image.apply_affine_transform(imagenes[i],theta=rotacion) for i in range(len(imagenes))]
      y=tf.stack(y)
      return y

