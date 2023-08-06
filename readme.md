# This is not a good ML framework but it is one
Implemented a ML autogead framework for pedagogical purposes. Goal was to train a Convnet on MNIST to decent accuracy (see mnist.ipynb).

## How it works
``Tensors`` (tensor.py) carry ``data`` and ``grad``. Numpy is the numerical backend. ``Op's`` (ops.py) implement ML operations (forward and backward pass) and serve as the operation nodes in the computation graph. ``Module's`` (nn.py) are objects with a forward() method that carry parameters, which can be retrieved using .parameters(). ``graph.py`` provides functionality to plot the computation graph. ``Optim.py`` contains an SGD optimizer used for model training.



