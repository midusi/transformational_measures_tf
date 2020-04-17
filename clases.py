#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:07:03 2020

@author: oem
"""
import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
import AAPutils 
import matplotlib.pyplot as plt
from time import time
 



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







class Modelo:
    
    def __init__(self,datos):
        self.modelo = ks.models.Sequential()
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
        
        
    def resumen(self):
        self.modelo.summary()
    
    def compilar(self):
        self.modelo.compile(optimizer=ks.optimizers.SGD(lr=self.lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'],)
    
    def entrenar(self):
        self.historia = self.modelo.fit(self.datos.x_train,self.datos.y_train,epochs=self.epochs,batch_size=self.batch_size,validation_data=(self.datos.x_test,self.datos.y_test))
        
    def plot_training_curves(self):
        AAPutils.plot_training_curves(self.historia)

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
        


class MetricaInvarianza:
    
    def __init__(self,ruta_nombre,datos,transformaciones):
        
        self.datos = datos
        self.transformaciones = transformaciones
        
        self.modelo_original = tf.keras.models.load_model(ruta_nombre)
        
 
        # se crea un nuevo modelo en capas para poder acceder a las activaciones    
 
        self.capas_salidas = [ capa.output for capa in self.modelo_original.layers ]
        self.capas_nombres = [ capa.name for capa in self.capas_salidas ] 
        self.capas_nombres = [ nombre[:nombre.index('/')] for nombre in self.capas_nombres ] 
               
        self.modelo_capas = ks.models.Model(inputs=self.modelo_original.input, outputs=self.capas_salidas)
            
        self.modelo_capas.summary()
        
        self.nactivaciones=0  #variable que almacena el numero de activaciones totales
        self.capas_nactivaciones = [] #lista que almacena el numero de activaciones de cada capa
        
        for capa in self.capas_salidas:
            a=1
            for i in capa.shape:
                if(i!=None):
                    a=a*i
                    
            self.nactivaciones=self.nactivaciones+a
            self.capas_nactivaciones.append(a)
            
            
        self.ncapas = len(self.capas_salidas) #variable que almacena el numero de capas del modelo
        
        
    def graficar(self):
        
        
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(self.calculo +" " + str(int(self.tiempo)))
        axs[0, 0].set_title('Transformacional')
        axs[0, 0].plot(self.capas_nombres, self.varianza_transformacional_capas)
        axs[0, 1].set_title('Muestral')
        axs[0, 1].plot(self.capas_nombres, self.varianza_muestral_capas)
        axs[1, 0].set_title('Normalizada')
        axs[1, 0].plot(self.capas_nombres, self.varianza_normalizada_capas)
        
        
    def heat_map(self):
        capas_activaciones = self.varianza_normalizada_capas_activaciones

        capas_nombres = self.capas_nombres
        ncapas = len(capas_nombres)
        y=[ tf.reduce_max(capas_activaciones[i]) for i in range(ncapas)]
        vmax=tf.reduce_max(tf.stack(y))
        y=[ tf.reduce_min(capas_activaciones[i]) for i in range(ncapas)]
        vmin=tf.reduce_min(tf.stack(y))
        if(vmin<0):
            vmin=0
        
        f, axes = plt.subplots(1,ncapas)
        for i in range(ncapas):
            ax = axes[i]
            ax.axis("off")
            activationes = capas_activaciones[i][:,np.newaxis]
            
            mappable = ax.imshow(activationes,vmax=vmax,vmin=vmin,cmap='inferno',aspect="auto")
            name = capas_nombres[i]
            if len(name)>7:
                name=name[:10]+"."
            ax.set_title(name, fontsize=8,rotation = 45)
            
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
        cbar.cmap.set_over('green')
        cbar.cmap.set_bad(color='blue')




        
        
        
    def calcular_1for(self):
        
                                       
        tiempo_inicial = time()
        
        def get_rotation(x,r):
            return tf.keras.preprocessing.image.apply_affine_transform(x,theta=r)
        
        
        muestras = self.datos.x_test.reshape(-1,28,28,1)
        n=muestras.shape[0]
        n=200
        m=len(self.transformaciones)
        k=self.ncapas
        capas_k = self.capas_nactivaciones
        modelo = self.modelo_capas
        
        self.varianza_transformacional_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        
        media_muestral_capas = [[ tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for j in range(m)] for r in range(k)]
        momento_muestral_capas = [[ tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for j in range(m)] for r in range(k)]
        
        
        for i in range(n):
            media = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            momento = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            for j in range(m):
                x = get_rotation(muestras[i,:,:,:], rotaciones[j])
                x = x.reshape(1,28,28,1)
                capas_salidas = modelo.predict(x)
                l=0
                for capa in capas_salidas:
                    capa = tf.reshape(capa,[-1])
                    media[l] = tf.add(media[l],capa/m)
                    momento[l] = tf.add(momento[l],tf.pow(capa,2)/(m-1))
                    
                    #calculos para muestral
                    media_muestral_capas[l][j] = tf.add(media_muestral_capas[l][j],capa/n)
                    momento_muestral_capas[l][j] = tf.add(momento_muestral_capas[l][j],tf.pow(capa,2)/(n-1))
        
                    l=l+1
        
            media_cuadrada = [ tf.pow(capa_media,2)*(m/(m-1)) for capa_media in media]
        
            print("\r1 solo for: "+str((i+1)*100/n) + " %",end="")
            
            self.varianza_transformacional_capas = [ tf.add(self.varianza_transformacional_capas[r],tf.math.subtract(momento[r],media_cuadrada[r])/n) for r in range(k)]
            
        
        self.varianza_transformacional_capas = [ tf.reduce_mean(self.varianza_transformacional_capas[r]) for r in range(k)]
               
        
        mediacuadrada_muestral_capas = [[ tf.pow(media_muestral_capas[l][j],2)*(n/(n-1)) for j in range(m)] for l in range(k)]
        varianza_muestral_capas_activaciones = [[ tf.math.subtract(momento_muestral_capas[l][j],mediacuadrada_muestral_capas[l][j]) for j in range(m)] for l in range(k)]
        varianza_muestral_capas_activaciones = [tf.reduce_mean(varianza_muestral_capas_activaciones[l][:]) for l in range(k)]
        
        self.varianza_muestral_capas = [tf.reduce_mean(varianza_muestral_capas_activaciones[l]) for l in range(k)]
        
        
        self.varianza_normalizada_capas = [tf.math.divide(self.varianza_transformacional_capas[l],self.varianza_muestral_capas[l]) for l in range(k)]
        
        self.tiempo = time() - tiempo_inicial
        self.calculo = "1 for sin batchs" 
 
        print(self.tiempo)
        
        
    def calcular_2for(self):
        
                                       
        tiempo_inicial = time()
        
        def get_rotation(x,r):
            return tf.keras.preprocessing.image.apply_affine_transform(x,theta=r)
        
        
        muestras = self.datos.x_test.reshape(-1,28,28,1)
        n=muestras.shape[0]
        n=200
        m=len(self.transformaciones)
        k=self.ncapas
        capas_k = self.capas_nactivaciones
        modelo = self.modelo_capas
        
        
        #Calculo Varianza Transformacional
        
        self.varianza_transformacional_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        
        for i in range(n):
            media = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            momento = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            for j in range(m):
                x = get_rotation(muestras[i,:,:,:], rotaciones[j])
                x = x.reshape(1,28,28,1)
                capas_salidas = modelo.predict(x)
                l=0
                for capa in capas_salidas:
                    capa = tf.reshape(capa,[-1])
                    media[l] = tf.add(media[l],capa/m)
                    momento[l] = tf.add(momento[l],tf.pow(capa,2)/(m-1))
                    l=l+1
        
            media_cuadrada = [ tf.pow(capa_media,2)*(m/(m-1)) for capa_media in media]
            print("\r1 er for: " + str((i+1)*100/n) + " %",end="")
            self.varianza_transformacional_capas = [ tf.add(self.varianza_transformacional_capas[r],tf.math.subtract(momento[r],media_cuadrada[r])/n) for r in range(k)]
            
        
        self.varianza_transformacional_capas = [ tf.reduce_mean(self.varianza_transformacional_capas[r]) for r in range(k)]
               
        
             
        #Calculo Varianza Muestral
        
        self.varianza_muestral_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        
        for j in range(m):
            media = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            momento = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            for i in range(n):
                x = get_rotation(muestras[i,:,:,:], rotaciones[j])
                x = x.reshape(1,28,28,1)
                capas_salidas = modelo.predict(x)
                l=0
                for capa in capas_salidas:
                    capa = tf.reshape(capa,[-1])
                    media[l] = tf.add(media[l],capa/n)
                    momento[l] = tf.add(momento[l],tf.pow(capa,2)/(n-1))
                    l=l+1
        
            media_cuadrada = [ tf.pow(capa_media,2)*(n/(n-1)) for capa_media in media]
            print("")
            print("\r2 do for : " + str((j+1)*100/m) + " %",end="")
            self.varianza_muestral_capas = [ tf.add(self.varianza_muestral_capas[r],tf.math.subtract(momento[r],media_cuadrada[r])/m) for r in range(k)]
            
        
        self.varianza_muestral_capas = [ tf.reduce_mean(self.varianza_muestral_capas[r]) for r in range(k)]
               
       
        
        
        self.varianza_normalizada_capas = [tf.math.divide(self.varianza_transformacional_capas[l],self.varianza_muestral_capas[l]) for l in range(k)]
        
        self.tiempo = time() - tiempo_inicial
          
        self.calculo = "2 for sin batchs" 
        print("")
        print("Tiempo 2 for: "+ str(self.tiempo))
        
        
        
        
      
    def calcular_usando_batchs(self,batchs_tamaño):
        
                                       
        tiempo_inicial = time()
        
        
        muestras = self.datos.x_test.reshape(-1,28,28,1)
        n=muestras.shape[0]
        n=200
        m=len(self.transformaciones)
        k=self.ncapas
        capas_k = self.capas_nactivaciones
        modelo = self.modelo_capas
        
               
        b=batchs_tamaño
        h=int(n/b)
                
        batchs = [muestras[j*b:(j+1)*b,:,:,:] for j in range(h)]
        
        if(n%b>0):
            batchs.append(muestras[h*b:n,:,:,:])
            h=h+1
        
        transformaciones = Transformaciones()
                
        
        
        #Calculo Varianza Transformacional
        
        self.varianza_transformacional_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        w=0
        for batch in batchs:
            
            
            nbatch,x1,x2,x3 = batch.shape
            media = [[tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)] for q in range(nbatch)]
            momento = [[tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)] for q in range(nbatch)]
            for j in range(m):
                
                
                x = transformaciones.generar(batch,rotaciones[j])
                
                capas_salidas_batch = modelo.predict(x)  # es una lista bidimensional [l,i] donde l representa la capa e i la muestra 
                                                         #, el elemento [l,i] es un tensor que contiene todas las salidas de la capa l para la muestra i 
                
                for i in range(nbatch):
                    
                    for l in range(k):
                        capa = capas_salidas_batch[l][i]
                        capa = tf.reshape(capa,[-1])
                        media[i][l] = tf.add(media[i][l],capa/m)
                        momento[i][l] = tf.add(momento[i][l],tf.pow(capa,2)/(m-1))
                        
                
                        
            
            
            media_cuadrada = [ [ tf.pow(capa_media,2)*(m/(m-1)) for capa_media in media[q][:]] for q in range(nbatch) ]
            
            
            for i in range(nbatch):
                self.varianza_transformacional_capas = [ tf.add(self.varianza_transformacional_capas[r],tf.math.subtract(momento[i][r],media_cuadrada[i][r])/n) for r in range(k)]
            
            w=w+1
            print("\rTransformacional: " + str(w*100/h) + " %",end="")
            
        self.varianza_transformacional_capas_activaciones = self.varianza_transformacional_capas
        self.varianza_transformacional_capas = [ tf.reduce_mean(self.varianza_transformacional_capas[r]) for r in range(k)]
               
        
        
        
        #Calculo Varianza Muestral
        
        self.varianza_muestral_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        
        for j in range(m):
            media = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            momento = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            
            
            for batch in batchs:

               x = transformaciones.generar(batch,rotaciones[j])
               capas_salidas_batch = modelo.predict(x)  # es una lista bidimensional [l,i] donde l representa la capa e i la muestra 
                                                         #, el elemento [l,i] es un tensor que contiene todas las salidas de la capa l para la muestra i 
               nbatch = batch.shape[0]
               for i in range(nbatch):
                    
                    for l in range(k):
                        capa = capas_salidas_batch[l][i]
                        capa = tf.reshape(capa,[-1])
                        media[l] = tf.add(media[l],capa/n)
                        momento[l] = tf.add(momento[l],tf.pow(capa,2)/(n-1))
                


        
            media_cuadrada = [ tf.pow(capa_media,2)*(n/(n-1)) for capa_media in media]
            self.varianza_muestral_capas = [ tf.add(self.varianza_muestral_capas[r],tf.math.subtract(momento[r],media_cuadrada[r])/m) for r in range(k)]

            print("")
            print("\rMuestral : " + str((j+1)*100/m) + " %",end="")

        
        self.varianza_muestral_capas_activaciones = self.varianza_muestral_capas

        self.varianza_muestral_capas = [ tf.reduce_mean(self.varianza_muestral_capas[r]) for r in range(k)]
        
        
        
        self.varianza_normalizada_capas = [tf.math.divide(self.varianza_transformacional_capas[l],self.varianza_muestral_capas[l]) for l in range(k)]
        self.varianza_normalizada_capas_activaciones = [tf.math.divide(self.varianza_transformacional_capas_activaciones[l],self.varianza_muestral_capas_activaciones[l]) for l in range(k)]

        
                   
        self.tiempo = time() - tiempo_inicial
        self.calculo = "2 for con batchs de tamaño " + str(batchs_tamaño) 

        print("")
        print("Tiempo con batchs: "+ str(self.tiempo))
        
        
        
    def calcular_usando_batchs_welford(self,batchs_tamaño):
        
                                       
        tiempo_inicial = time()
        
        
        muestras = self.datos.x_test.reshape(-1,28,28,1)
        n=muestras.shape[0]
        n=200
        m=len(self.transformaciones)
        k=self.ncapas
        capas_k = self.capas_nactivaciones
        modelo = self.modelo_capas
        
               
        b=batchs_tamaño
        h=int(n/b)
                
        batchs = [muestras[j*b:(j+1)*b,:,:,:] for j in range(h)]
        
        if(n%b>0):
            batchs.append(muestras[h*b:n,:,:,:])
            h=h+1
        
        transformaciones = Transformaciones()
                
        
        
        #Calculo Varianza Transformacional
        
        self.varianza_transformacional_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        w=0
        for batch in batchs:
            
            
            nbatch,x1,x2,x3 = batch.shape
            media = [[tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)] for q in range(nbatch)]
            momento = [[tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)] for q in range(nbatch)]
            
            count=0
            for j in range(m):
                
                
                x = transformaciones.generar(batch,rotaciones[j])
                
                capas_salidas_batch = modelo.predict(x)  # es una lista bidimensional [l,i] donde l representa la capa e i la muestra 
                                                         #, el elemento [l,i] es un tensor que contiene todas las salidas de la capa l para la muestra i 
                count+=1
                for i in range(nbatch):
                    
                    for l in range(k):
                        capa = capas_salidas_batch[l][i]
                        capa = tf.reshape(capa,[-1])
                        media_anterior = media[i][l]
                        media[i][l] = tf.add(media_anterior,tf.subtract(capa,media_anterior)/count)
                        momento[i][l] = tf.add(momento[i][l],tf.multiply(tf.subtract(capa,media_anterior),tf.subtract(capa,media[i][l])))
                        
                
                        
            
            
            
            
            for i in range(nbatch):
                self.varianza_transformacional_capas = [ tf.add(self.varianza_transformacional_capas[r],(momento[i][r]/(count-1))/n) for r in range(k)]
            
            w=w+1
            print("\rTransformacional: " + str(w*100/h) + " %",end="")
            
        self.varianza_transformacional_capas_activaciones = self.varianza_transformacional_capas
        self.varianza_transformacional_capas = [ tf.reduce_mean(self.varianza_transformacional_capas[r]) for r in range(k)]
               
        
        
        
        #Calculo Varianza Muestral
        
        self.varianza_muestral_capas = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
        
        for j in range(m):
            media = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            momento = [tf.zeros([capas_k[r]],dtype=tf.dtypes.float32) for r in range(k)]
            
            count=0
            for batch in batchs:

               x = transformaciones.generar(batch,rotaciones[j])
               capas_salidas_batch = modelo.predict(x)  # es una lista bidimensional [l,i] donde l representa la capa e i la muestra 
                                                         #, el elemento [l,i] es un tensor que contiene todas las salidas de la capa l para la muestra i 
               nbatch = batch.shape[0]
               for i in range(nbatch):
                    count+=1
                    for l in range(k):
                        capa = capas_salidas_batch[l][i]
                        capa = tf.reshape(capa,[-1])
                        media_anterior = media[l]
                        media[l] = tf.add(media_anterior,tf.subtract(capa,media_anterior)/count)
                        momento[l] = tf.add(momento[l],tf.multiply(tf.subtract(capa,media_anterior),tf.subtract(capa,media[l])))
                       
                


        
            self.varianza_muestral_capas = [ tf.add(self.varianza_muestral_capas[r],(momento[r]/(count-1))/m) for r in range(k)]

            print("")
            print("\rMuestral : " + str((j+1)*100/m) + " %",end="")

        
        self.varianza_muestral_capas_activaciones = self.varianza_muestral_capas

        self.varianza_muestral_capas = [ tf.reduce_mean(self.varianza_muestral_capas[r]) for r in range(k)]
        
        
        
        self.varianza_normalizada_capas = [tf.math.divide(self.varianza_transformacional_capas[l],self.varianza_muestral_capas[l]) for l in range(k)]
        self.varianza_normalizada_capas_activaciones = [tf.math.divide(self.varianza_transformacional_capas_activaciones[l],self.varianza_muestral_capas_activaciones[l]) for l in range(k)]

        
                   
        self.tiempo = time() - tiempo_inicial
        self.calculo = "2 for con batchs de tamaño " + str(batchs_tamaño) 

        print("")
        print("Tiempo con batchs: "+ str(self.tiempo))
     
        
    
            

datos = Datos()
modelo = Modelo(datos)
rotaciones = [20*x for x in range(18)]




metrica = MetricaInvarianza("modelosimple_entrenado.h5",datos,rotaciones)
metrica.calcular_usando_batchs(30)
metrica.heat_map()

metrica.calcular_2for()


metrica.calcular_usando_batchs_welford(30)

metrica.graficar()
























harvest = [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]]

print(harvest[2][:])
capas_activaciones = metrica.varianza_normalizada_capas_activaciones

plt.figure()
plt.plot(capas_activaciones[0])

capas_nombres = metrica.capas_nombres
ncapas = len(capas_nombres)
y=[ tf.reduce_max(capas_activaciones[i]) for i in range(ncapas)]
vmax=tf.reduce_max(tf.stack(y))

y=[ tf.reduce_min(capas_activaciones[i]) for i in range(ncapas)]
vmin=tf.reduce_min(tf.stack(y))

if(vmin<0):
    vmin=0
#print(vmin)

#len(capas_activaciones)
#print(capas_activaciones[0].shape)
#x = capas_activaciones[0][:,np.newaxis]

f, axes = plt.subplots(1,ncapas)
#for i, (activation, name) in enumerate(zip(m.layers, m.layer_names)):
for i in range(ncapas):
    ax = axes[i]
    ax.axis("off")
    
    #activation = harvest[i][:]
    #aux = np.array(activation)
    #activation = aux[:,np.newaxis]
    
    activation = capas_activaciones[i][:,np.newaxis]
    
    mappable = ax.imshow(activation,vmax=vmax,vmin=vmin,cmap='inferno',aspect="auto")
    name = capas_nombres[i]
    if len(name)>7:
        name=name[:10]+"."
    ax.set_title(name, fontsize=8,rotation = 45)
    
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
cbar.cmap.set_over('green')
cbar.cmap.set_bad(color='blue')

















"""
metrica.calcular_2for()
metrica.graficar()

metrica.calcular_1for()
metrica.graficar()



b=160
n=200 
h=int(n/b)
        
batchs = [datos.x_test[j*b:(j+1)*b,:,:,:] for j in range(h)]

if(n%b>0):
    batchs.append(datos.x_test[h*b:n,:,:,:])


print(len(batchs))

print(batchs[0].shape)

y = [ tf.keras.preprocessing.image.apply_affine_transform(batchs[0][i,:,:,:],theta=20) for i in range(160)]

y=tf.stack(y)


print(y.shape)
       
print(y[0].shape)



x2=  metrica.modelo_capas.predict(y)



print(x2[5][0].shape)


x3 = x2[:][0]


print(x3[2].shape)


print("\rshape",end="")
print("\nshape",end="")
print("\rshape",end="")
print("\rshape",end="")



"""
"""



transformaciones = Transformaciones()

x=transformaciones.generar(batchs[0],20)
print(x.shape)

b=160
n=200 
h=int(n/b)
        
batchs = [datos.x_test[j*b:(j+1)*b,:,:,:] for j in range(h)]

if(n%b>0):
    batchs.append(datos.x_test[h*b:n,:,:,:])


print(len(batchs))

print(batchs[0].shape)

y = [ tf.keras.preprocessing.image.apply_affine_transform(batchs[0][i,:,:,:],theta=20) for i in range(len(batchs))]

y=tf.stack(y)


print(y.shape)
       
print(y[0].shape)



x2=  metrica.modelo_capas.predict(x)


print(x2[0][1].shape)
print(x[4][1].shape)


w = [ tf.stack(x[l][:]) for l in range(6)]

print(w[0].shape)



q = tf.math.reduce_sum(w[0], axis=0)

print(q.shape)





metrica = MetricaInvarianza("modelosimple_entrenado.h5",datos,rotaciones)
    


"""






"""
datos = Datos()
modelo = Modelo(datos)
modelo.ejecutar_guardar("modelosimple_entrenado.h5")

datos.set_longitud_x_test(100)
rotaciones = [20*x for x in range(18)]

metrica = MetricaInvarianza("modelosimple_entrenado.h5",datos,rotaciones)
    
metrica.calcular_1for()
metrica.graficar()
metrica.calcular_2for()
metrica.graficar()


"""
























"""
print(media)

(x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()
        
d_in,x,y = x_train.shape

print(x_train.shape)

x_train = x_train[0:2000,:,:]
x2 = tf.reshape(x_train,[-1])
print(x2.shape)






x3 = tf.reduce_mean(x2)
print(x3)


x=tf.zeros([d_in],dtype=tf.dtypes.float32)

x3 = [tf.reduce_mean(x_train[i,:,:]) for i in range(d_in)]

x3 = tf.reduce_mean(x)



datos = Datos()
modelo = Modelo(datos)

modelo.resumen()
modelo.compilar()
modelo.entrenar()
modelo.plot_training_curves()

modelo.modelo.save("modelobasico.hs")

new_model = tf.keras.models.load_model('modelosimple.h5')
new_model.save("modelosimple.h5")
# Check its architecture
new_model.summary()

x=new_model.predict(modelo.datos.x_train[0:1,:,:,:])

print(x)

plt.matshow(modelo.datos.x_train[0,:,:,0], cmap='viridis')


print(new_model.input.shape)






"""








