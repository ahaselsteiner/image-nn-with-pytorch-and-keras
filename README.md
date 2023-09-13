# Neural network image classifier in pytorch and keras

The same neural network archictecture is implemented in Keras using TensorFlow and in 
PyTorch to read hand-written digits. The MNIST dataset is used.

Neural network archictecture:

_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0

 conv2d (Conv2D)             (None, 26, 26, 32)        320

 max_pooling2d (MaxPooling2  (None, 13, 13, 32)        0
 D)

 conv2d_1 (Conv2D)           (None, 11, 11, 32)        9248

 max_pooling2d_1 (MaxPoolin  (None, 5, 5, 32)          0
 g2D)

 conv2d_2 (Conv2D)           (None, 3, 3, 64)          18496

 max_pooling2d_2 (MaxPoolin  (None, 1, 1, 64)          0
 g2D)

 flatten (Flatten)           (None, 64)                0

 dense (Dense)               (None, 10)                650

 dense_1 (Dense)             (None, 10)                110
