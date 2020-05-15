import tensorflow as tf


class Calculo_Varianza:

    def __init__(self, numero_capas, numero_activaciones_por_capa, mov):
        self.mov = mov
        self.numero_activaciones_por_capa = numero_activaciones_por_capa
        self.numero_capas = numero_capas
        self.numero_divisor = 0  # numero divisor de la varianza de la activación
        self.varianza_capas_activaciones = [tf.zeros(
            [self.numero_activaciones_por_capa[r]], dtype=tf.dtypes.float32) for r in range(self.numero_capas)]

    def renovar(self, longitud_memoria):
        self.count = 0
        self.longitud_memoria = longitud_memoria
        self.media = [[tf.zeros([self.numero_activaciones_por_capa[r]], dtype=tf.dtypes.float32)
                       for r in range(self.numero_capas)] for q in range(longitud_memoria)]
        self.momento = [[tf.zeros([self.numero_activaciones_por_capa[r]], dtype=tf.dtypes.float32)
                         for r in range(self.numero_capas)] for q in range(longitud_memoria)]
        self.numero_divisor += longitud_memoria

    def adicionar(self, bloque, longitud_mov):
        self.longitud_mov = longitud_mov
        for q in range(self.longitud_mov):
            self.count += 1
            for r in range(self.longitud_memoria):

                for l in range(self.numero_capas):

                    if(self.mov == "horizontal"):
                        capa = bloque[(r, q, l)]
                    else:
                        capa = bloque[(q, r, l)]

                    media_anterior = self.media[r][l]
                    self.media[r][l] = tf.add(media_anterior, tf.subtract(
                        capa, media_anterior)/self.count)
                    self.momento[r][l] = tf.add(self.momento[r][l], tf.multiply(
                        tf.subtract(capa, media_anterior), tf.subtract(capa, self.media[r][l])))


    def actualizar(self):
        for r in range(self.longitud_memoria):
            self.varianza_capas_activaciones = [tf.add(self.varianza_capas_activaciones[l], (
                self.momento[r][l]/(self.count-1))) for l in range(self.numero_capas)]

    def finalizar(self):
        self.varianza_capas_activaciones = [
            self.varianza_capas_activaciones[l]/self.numero_divisor for l in range(self.numero_capas)]
        self.varianza_capas = [tf.reduce_mean(
            self.varianza_capas_activaciones[l]) for l in range(self.numero_capas)]

