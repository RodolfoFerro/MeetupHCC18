# Introducción a Deep Learning con Keras

> **Author:** Rodolfo Ferro Pérez <br/>
> **Email:** [ferro@cimat.mx](mailto:ferro@cimat.mx) <br/>
> **Twitter:** [@FerroRodolfo](http://twitter.com/FerroRodolfo) <br/>
> **GitHub:** [RodolfoFerro](https://github.com/RodolfoFerro) <br/>

## Acerca de 🕹

El [Centro de Investigación en Matemáticas (CIMAT) A.C.](https://www.cimat.mx/), [Unidad Zacatecas](http://www.ingsoft.mx/), a través del *Laboratorio de Computación Centrada en el Humano* crea el [**Meetup HCC-CIMAT 2018**](https://hcc-cimat.com/meetup2018/), encuentro de Computación Centrada en el Humano donde se presentan diversas charlas y talleres dirigidos a estudiantes universitarios, investigadores, profesionales del área y líderes funcionarios.

El presente repo contiene una charla sobre reconocimiento de emociones en conjuntos de personas, así como un taller de introducción a *Deep Learning* con Python 🐍, para el [**Meetup HCC-CIMAT 2018**](https://hcc-cimat.com/meetup2018/) a realizarse en *CIMAT, Unidad Zacatecas* el próximo 16 de agosto de 2018.

Este último es básicamente un taller introductorio sobre [*Redes Neuronales Artificiales*](https://en.wikipedia.org/wiki/Artificial_neural_network) utilizando [Keras](https://keras.io/). Respecto a la charla, se describe la motivación, el diseño experimental y desarrollo del mismo (hasta este punto).

Para conocer el programa completo del evento, revisa el siguiente enlace: [https://hcc-cimat.com/meetup2018/programa/](https://hcc-cimat.com/meetup2018/programa/)

## Configuración ⚙️

Se trabajará en la nube, utilizando el [**Google CoLaboratory**](https://colab.research.google.com) para crear modelos de redes neuronales de manera gratuita y utilizando GPUs para el entrenamiento en línea.

Para ello, bastará crear un nuevo Notebook para Python 3. Todo el contenido será copiado directamente del Notebook contenido en este repositorio: [https://github.com/RodolfoFerro/MeetupHCC18/](https://github.com/RodolfoFerro/MeetupHCC18/)


## Contenido 👾

Todo el contenido del taller y la charla se encuentran en este repositorio. Asimismo, puede notarse la existencia de la carpeta [`more`](https://github.com/RodolfoFerro/MeetupHCC18/tree/master/more), que contiene un conjunto de archivos que complementan el contenido principal del taller así como un modelo pre-entrenado y un conjunto de imágenes utilizadas en los notebooks y el repo.

Scripts utilizados en el taller:
- [`PerceptronHCC.py`](https://github.com/RodolfoFerro/MeetupHCC18/blob/master/PerceptronHCC.py)
- [`SigmoidHCC.py`](https://github.com/RodolfoFerro/MeetupHCC18/blob/master/SigmoidHCC.py)


#### `PerceptronHCC.py`
```python
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
    entradas = [1, 1, 1, 1]
    pesos = [-4, 3, 1, 2]

    meetup = PerceptronHCC(entradas, pesos)
    print(meetup.voy_no_voy(3))
```

#### `SigmoidHCC.py`
```python
import numpy as np


class SigmoidNeuron():
    def __init__(self, n):
        np.random.seed(123)
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_output, iterations):
        for iteration in range(iterations):
            output = self.predict(training_inputs)
            error = training_output.reshape((len(training_inputs), 1)) - output
            adjustment = np.dot(training_inputs.T, error *
                                self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    # Initialize Sigmoid Neuron:
    sigmoid = SigmoidNeuron(2)
    print("Inicialización de pesos aleatorios:")
    print(sigmoid.synaptic_weights)

    # Datos de entrenamiento:
    training_inputs = np.array([[1, 0], [0, 0], [0, 1]])
    training_output = np.array([1, 0, 1]).T.reshape((3, 1))

    # Entrenamos la neurona (100,000 iteraciones):
    sigmoid.train(training_inputs, training_output, 100000)
    print("Nuevos pesos sinápticos luego del entrenamiento: ")
    print(sigmoid.synaptic_weights)

    # Predecimos para probar la red:
    print("Predicción para [1, 1]: ")
    print(sigmoid.predict(np.array([1, 1])))
```

***

### ABOUT COPYING OR USING PARTIAL INFORMATION: 🔐
* These documents were originally created by the author.
* Any usage of these documents or their contents is granted according to the provided license and its conditions.
* The datasets, models and experimental procedures are part of a research group @ CIMAT, and hereby they shall not be considered part of the workshop or main content of this repo. To use them for personal purposes, please contact the owner of this repo.
* For any question, you can contact the author via email or Twitter.

**Copyright (c) 2018 Rodolfo Ferro**
