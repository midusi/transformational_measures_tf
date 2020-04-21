#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:05:29 2020

@author: oem
"""


from modelos import ModeloSimple,ModeloSimpleConv
from metricas import MetricaInvarianza
from otros import Datos



datos = Datos()
modelo = ModeloSimpleConv(datos)

#modelo.ejecutar_guardar("pr.h5")
