import numpy as np


class PerceptronHCC():
    def __init__(self, entradas, pesos):
        """Constructor de la clase."""
        self.n = len(entradas)
        self.entradas = np.array(entradas)
        self.pesos = np.array(pesos)

    def voy_no_voy(self, umbral):
        """Calcula el output deseado."""
        si_no = (self.entradas @ self.pesos) >= umbral
        if si_no:
            return "Sí voy."
        else:
            return "No voy."


if __name__ == '__main__':
    entradas = [0, 1, 0, 1]
    pesos = [-4, 3, 11, 2]

    meetup = PerceptronHCC(entradas, pesos)
    print(meetup.voy_no_voy(1))
