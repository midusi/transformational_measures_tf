
from calculo_varianza import Calculo_Varianza
from iterador import Iterador


class VarianzaTransformacional:

    def __init__(self, iterador: Iterador, numero_capas, numero_activaciones_por_capa):
        self.iterador = iterador
        self.numero_capas = numero_capas
        self.numero_activaciones_por_capa = numero_activaciones_por_capa

    def calcular(self, alto, ancho):

        tiempo_inicial = time()

        calculo = Calculo_Varianza(
            self.numero_capas, self.numero_activaciones_por_capa, "horizontal")

        for porcentaje, memoria, bloques_muestra in self.iterador.por_bloques_muestras_primero(alto, ancho):
            print("\r Realizado: " +
                  str((int(porcentaje*1000)/1000)*100)+" %", end="")
            calculo.renovar(memoria)
            for long_mov, bloque in bloques_muestra:
                calculo.adicionar(bloque, long_mov)
            calculo.actualizar()

        calculo.finalizar()
        print("\r Transformacional Realizado: " + "100 %", end="")

        self.varianza_capas = calculo.varianza_capas

        self.tiempo = time() - tiempo_inicial
        print("\rTiempo Transformacional: " + str(self.tiempo), end="")


class VarianzaMuestral:

    def __init__(self, iterador: Iterador, numero_capas, numero_activaciones_por_capa):
        self.iterador = iterador
        self.numero_capas = numero_capas
        self.numero_activaciones_por_capa = numero_activaciones_por_capa

    def calcular(self, alto, ancho):

        tiempo_inicial = time()

        calculo = Calculo_Varianza(
            self.numero_capas, self.numero_activaciones_por_capa, "vertical")

        for porcentaje, memoria, bloques_transformacion in self.iterador.por_bloques_transformaciones_primero(alto, ancho):
            print("\r Realizado: " +
                  str((int(porcentaje*1000)/1000)*100)+" %", end="")
            calculo.renovar(memoria)
            for long_mov, bloque in bloques_transformacion:
                calculo.adicionar(bloque, long_mov)
            calculo.actualizar()

        calculo.finalizar()
        print("\r Muestral Realizado: " + "100 %", end="")

        self.varianza_capas = calculo.varianza_capas

        self.tiempo = time() - tiempo_inicial
        print("\rTiempo Muestral: " + str(self.tiempo), end="")


class VarianzaNormalizada:
    def __init__(self, iterador: Iterador, numero_capas, numero_activaciones_por_capa):
        self.iterador = iterador
        self.numero_capas = numero_capas
        self.numero_activaciones_por_capa = numero_activaciones_por_capa
        self.varianza_muestral = VarianzaMuestral(
            iterador, numero_capas, numero_activaciones_por_capa)
        self.varianza_transformacional = VarianzaTransformacional(
            iterador, numero_capas, numero_activaciones_por_capa)

    def calcular(self, alto, ancho):
        tiempo_inicial = time()
        self.varianza_transformacional.calcular(alto, ancho)
        self.varianza_muestral.calcular(alto, ancho)
        self.varianza_capas = [tf.math.divide(self.varianza_transformacional.varianza_capas[l],
                                              self.varianza_muestral.varianza_capas[l]) for l in range(self.numero_capas)]

        self.tiempo = time() - tiempo_inicial
        print("\rTiempo: " + str(self.tiempo), end="")
