# Neural network image classifier in PyTorch and Keras using TensorFlow

The same neural network archictecture is implemented in Keras using TensorFlow and in 
PyTorch to read hand-written digits. The MNIST dataset wich contains hand-written digits
is used.

Neural network archictecture (28714 parameters in PyTorch, 28824 in Keras/TensorFlow):
 - (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
 - (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
 - (2): ReLU()
 - (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
 - (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
 - (5): ReLU()
 - (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
 - (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
 - (8): ReLU()
 - (9): Flatten(start_dim=1, end_dim=-1)
 - (10): Linear(in_features=64, out_features=10, bias=True)
  