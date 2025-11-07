# Implementacion de modelo de Deep Learning

Ulises Orlando Carrizalez Lerín - A01027715

## Dataset

Este proyecto hace uso del conjunto de datos de clasificación de emociones "Emotions Dataset" de Kaggle, este está diseñado para facilitar la investigación y la experimentación en el campo del procesamiento del lenguaje natural y el análisis de emociones. Contiene una colección diversa de muestras de texto, cada una etiquetada con la emoción correspondiente que transmite. Las emociones pueden variar desde felicidad y entusiasmo hasta enojo, tristeza y más.

Las emociones del dataset se clasifican en seis categorías:

* tristeza (0)

* alegría (1)

* amor (2)

* enojo (3)

* miedo (4)

* sorpresa (5)

## Modelo
Para resolver el problema de análisis de sentimientos entre las diferentes clases, se implementó una red neuronal recurrente (RNN) del tipo GRU utilizando el framework PyTorch.
La elección de la GRU se debe a que es más sencilla de implementar que una LSTM y ofrece un rendimiento similar, además de evitar los problemas de vanishing gradient que suelen presentarse en las RNN tradicionales cuando se trabajan con secuencias de texto largas.

En la fase de preprocesamiento, se generó un embedding de 100 dimensiones. Posteriormente, se realizó la generación del vocabulario, la tokenización y la creación de secuencias necesarias para la entrada de la red neuronal. Con base en estos datos, se construyeron los objetos Dataset y, posteriormente, los DataLoader, utilizados para entrenar el modelo.

Como se mencionó anteriormente, el modelo está basado en una arquitectura GRU, cuya arquitectura consiste en las siguientes etapas:

1. Capa de embedding: transforma las palabras en vectores densos de 100 dimensiones.

2. Capa GRU: recibe los embeddings como entrada y aprende los patrones contextuales presentes en los textos.

3. Capa densa: compuesta por 128 neuronas y una capa de salida con 6 neuronas, una por cada emocion.

Para el entrenamiento se utilizó la función de pérdida Cross Entropy, ya que se trata de un problema de clasificación multiclase. La función de activación empleada fue ReLU por su eficiencia computacional, y se estableció una tasa de aprendizaje (learning rate) de 0.001.

El entrenamiento del modelo se llevó a cabo durante 10 épocas. Durante este proceso, se implementó una función que guarda automáticamente el modelo con la menor pérdida en su prueba de validacion. Este modelo óptimo se evalúa con test y se generan las gráficas correspondientes a la evolución de la pérdida durante el entrenamiento.

Finalmente, se guardaron los pesos del modelo en un archivo .pt, y tanto el vocabulario como el encoder en archivos .pkl, con el propósito de reutilizar el modelo sin necesidad de volver a entrenarlo.

## Documentos

Gen_Model.py     : Archivo de Python para generar el modelo
Model_Componetnes: contiene las funciones utiles y las arquitecturas del mmodelo y del dataset
Exe_Model        : Archivo para ejecutrar el modelo guardado en el .pt y los .pkl
