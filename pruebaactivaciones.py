#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:03:29 2020

@author: oem
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import AAPutils 
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = ks.datasets.mnist.load_data()

d_in,x,y = X_train.shape


media=np.mean(X_train)
var=np.std(X_train)
# Normalizod_in las variables de entrada
for i in range(d_in):
    X_train[i,:,:]=(X_train[i,:,:]-media)/var
  
print(media)

d_in,x,y = X_test.shape

# Normalizod_in las variables de entrada
for i in range(d_in):
    X_test[i,:,:]=(X_test[i,:,:]-media)/var
    


X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)




modelo = ks.models.Sequential()
modelo.add(ks.layers.Conv2D(100, (3, 3), input_shape=(28, 28, 1), name='Convolutional_layer'))
modelo.add(ks.layers.Activation('relu'))

modelo.add(ks.layers.MaxPooling2D((2, 2), name='Maxpooling_2D'))
modelo.add(ks.layers.Flatten(name='Flatten'))
modelo.add(ks.layers.Dense(50, activation='relu', name='Hidden_layer'))
modelo.add(ks.layers.Dense(10, activation='softmax', name='Output_layer'))



modelo.summary()



modelo.compile(optimizer=ks.optimizers.SGD(lr=0.002),loss='sparse_categorical_crossentropy',metrics=['accuracy'],)

# Entrenamiento del modelo
history=modelo.fit(X_train,Y_train,epochs=1,batch_size=16,validation_data=(X_test,Y_test))





# --- Clase Métrica Invarianza (modelo , muestras , transformaciones) 
# --- set...()
# --- calcular()
# --- graficar() 

    
layer_outputs = [layer.output for layer in modelo.layers]
layer_names = [ layer.name for layer in layer_outputs] 
layer_names = [ name[:name.index('/')] for name in layer_names] 


activation_model = ks.models.Model(inputs=modelo.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input


k=0
for layer in modelo.layers:
    a=1
    for i in layer.output_shape:
        if(i!=None):
            a=a*i
    print(layer.output_shape)
    k=k+a
    
layers_k = []
for layer in modelo.layers:
    a=1
    for i in layer.output_shape:
        if(i!=None):
            a=a*i
    layers_k.append(a)
    

capas = len(activation_model.output)





muestras = X_test.reshape(-1,28,28,1)
n=muestras.shape[0]
rotaciones = np.array([20*x for x in range(18)])
m=rotaciones.shape[0]



def get_rotation(x,r):
    return tf.keras.preprocessing.image.apply_affine_transform(x,theta=r)

    



varianza_layer = [tf.zeros([layers_k[r]],dtype=tf.dtypes.float32) for r in range(capas)]

media_muestral_capas = [[ tf.zeros([layers_k[r]],dtype=tf.dtypes.float32) for j in range(m)] for r in range(capas)]
momento_muestral_capas = [[ tf.zeros([layers_k[r]],dtype=tf.dtypes.float32) for j in range(m)] for r in range(capas)]


n=200
for i in range(n):
    media = [tf.zeros([layers_k[r]],dtype=tf.dtypes.float32) for r in range(capas)]
    momento = [tf.zeros([layers_k[r]],dtype=tf.dtypes.float32) for r in range(capas)]
    for j in range(m):
        x = get_rotation(muestras[i,:,:,:], rotaciones[j])
        x = x.reshape(1,28,28,1)
        layers = activation_model.predict(x)
        l=0
        for layer in layers:
            layer = tf.reshape(layer,[-1])
            media[l] = tf.add(media[l],layer/m)
            momento[l] = tf.add(momento[l],tf.pow(layer,2)/(m-1))
            
            #calculos para muestral
            media_muestral_capas[l][j] = tf.add(media_muestral_capas[l][j],layer/n)
            momento_muestral_capas[l][j] = tf.add(momento_muestral_capas[l][j],tf.pow(layer,2)/(n-1))

            l=l+1

    media_cuadrada = [ tf.pow(layer_media,2)*(m/(m-1)) for layer_media in media]

    #
    print(str((i+1)*100/n) + " %")
    
    varianza_layer = [ tf.add(varianza_layer[r],tf.math.subtract(momento[r],media_cuadrada[r])/n) for r in range(capas)]
    

varianza_layer = [ tf.reduce_mean(varianza_layer[r]) for r in range(capas)]
       

mediacuadrada_muestral_capas = [[ tf.pow(media_muestral_capas[l][j],2)*(n/(n-1)) for j in range(m)] for l in range(capas)]

varianza_muestral_capas_activaciones = [[ tf.math.subtract(momento_muestral_capas[l][j],mediacuadrada_muestral_capas[l][j]) for j in range(m)] for l in range(capas)]

varianza_muestral_capas_activaciones = [tf.reduce_mean(varianza_muestral_capas_activaciones[l][:]) for l in range(capas)]

varianza_muestral_capas = [tf.reduce_mean(varianza_muestral_capas_activaciones[l]) for l in range(capas)]


varianza_normalizada = [tf.math.divide(varianza_layer[l],varianza_muestral_capas[l]) for l in range(capas)]


    

fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title('Transformacional')
axs[0, 0].plot(layer_names, varianza_layer)
axs[0, 1].set_title('Muestral')
axs[0, 1].plot(layer_names, varianza_muestral_capas)
axs[1, 0].set_title('Normalizada')
axs[1, 0].plot(layer_names, varianza_normalizada)





"""


from os import system, name 
  
# define our clear function 
def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 


print("aa")


clear()



plt.figure()
plt.title("normalizada")
plt.plot(layer_names,varianza_normalizada)
plt.figure()
plt.title("transformacional")
plt.plot(layer_names,varianza_layer[:])
plt.figure()
plt.title("muestral")
plt.plot(layer_names,varianza_muestral_capas[:])







for i in range(len(rotaciones)):
    transformation = tf.keras.preprocessing.image.apply_affine_transform(
    muestras[0,:,:,:],
    theta=rotaciones[i]
    )
    plt.matshow(transformation.reshape(28,28))



# Returns a list of five Numpy arrays: one array per layer activation


#print(activations[1].shape)


#plt.matshow(activations[0][0, :, :, 4], cmap='viridis')







var_transformacional = tf.zeros([k],dtype=tf.dtypes.float32)

var_aux = tf.ones([10],dtype=tf.dtypes.float32)
var_aux = var_aux[:]/2
var_aux = tf.pow(var_aux[:],2)[:]/4
var_aux = tf.pow(var_aux,2)/4
print(var_aux)





x= get_rotation(muestras[0,:,:,:], rotaciones[0])
x=x.reshape(1,28,28,1)

gen = gen_activaciones(x,activation_model)

for activacion in gen:
    print(activacion)  
    input()    


media = [tf.zeros([layer.shape],dtype=tf.dtypes.float32) for layer in activation_model.output]




def gen_activaciones(x,model):
    layers = model.predict(x)
    for layer in layers:
        layer_activations = tf.reshape(activations[0],[-1])
        for activacion in layer_activations:
            yield activacion



def get_activaciones(x,model):
    layers = model.predict(x)
    for layer in layers:
        layer_activations = tf.reshape(activations[0],[-1])
        for activacion in layer_activations:
            yield activacion

 

X=X_test[0,:,:].reshape(1,28,28,1)
activations = activation_model.predict(X) 
#activations = tf.stack(activations)
activationLayer = activations[0].reshape(26*26*50)
activationLayer = tf.reshape(activations[0],[-1])

print(len(activation_model.output))
print(activation_model.output[0].shape)
print(activations[0].shape)
print(activationLayer.an)
print(activationLayer.name)
print(activationLayer.shape)
print(layer_outputs[3].name)
print(muestras.shape)
print(varianza_layer)
print(varianza_layer[0])







AAPutils.plot_training_curves(history)


var_aux[0] =  2
print(var_aux[0])
#var_muestral = tf.zeros([k],dtype=tf.dtypes.float32)



"""


