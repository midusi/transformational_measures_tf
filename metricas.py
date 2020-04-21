#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:51:29 2020

@author: oem
"""


import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np
from time import time
from otros import Transformaciones



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
                x = get_rotation(muestras[i,:,:,:], self.transformaciones[j])
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
                x = get_rotation(muestras[i,:,:,:], self.transformaciones[j])
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
                x = get_rotation(muestras[i,:,:,:], self.transformaciones[j])
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
                
                
                x = transformaciones.generar(batch,self.transformaciones[j])
                
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

               x = transformaciones.generar(batch,self.transformaciones[j])
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
                
                
                x = transformaciones.generar(batch,self.transformaciones[j])
                
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

               x = transformaciones.generar(batch,self.transformaciones[j])
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
     
        