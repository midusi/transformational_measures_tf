import tensorflow as tf
from tensorflow import keras as ks
import matplotlib.pyplot as plt
import numpy as np
from modelo import Modelo
from dataset import Dataset
from iterador import Iterador
from varianzas import VarianzaNormalizada

ruta = ""


modelo = Modelo(ruta+"/modelosimple.h5")
dataset = Dataset()
iterador = Iterador(modelo, dataset)
#varianza_transformacional = VarianzaTransformacional(iterador,modelo.numero_capas(),modelo.numero_activaciones_por_capa())
# varianza_transformacional.calcular(10,2)
#varianza_muestral = VarianzaMuestral(iterador,modelo.numero_capas(),modelo.numero_activaciones_por_capa())
# varianza_muestral.calcular(10,2)

varianza_normalizada = VarianzaNormalizada(
    iterador, modelo.numero_capas(), modelo.numero_activaciones_por_capa())
varianza_normalizada.calcular(10, 2)

# plt.figure()
# plt.title("transformacional")
# plt.plot(modelo.capas_nombres,varianza_transformacional.varianza_capas)
# print(varianza_transformacional.varianza_capas)

plt.figure()
plt.title("transformacional")
plt.plot(modelo.capas_nombres, varianza_normalizada.varianza_capas)
print(varianza_normalizada.varianza_capas)
