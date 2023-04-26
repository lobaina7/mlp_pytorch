# mlp_pytorch
Perceptron Multicapa para problemas de regression usando Pytorch

Este perceptron fue creado con tres capas (una de entrada, una oculta y otra de salida)
La capa oculta utiliza como funcion de activacion la tangente sigmoidea, aunque uede utilizarse tambien ReLU
Cuenta con una clase para el entrenamiento y la validacion, incorporando parada temprana como un simple metodo para evitar sobre entrenamiento de la red.
Ademas, cuenta con una clase para la normalizacion previa de los datos.
