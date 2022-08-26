# diy-nn

Why bother with Tensorflow/Keras/Pytorch when just numpy and some math can do the job ?

This repo implements a simple Neural Network architecture for classification purpose built using only `numpy`.

## Setup

Setup the project with the following command:

```bash
python setup.py install
```

## Usage

Once setup is done, you can use the module as follow:

```python
from diynn.diy_nn import DIYNN

INPUT_LAYER_SIZE = 10
HIDDEN_LAYER_SIZE = 64
OUTPUT_LAYER_SIZE = 2

nn = DIYNN(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
nn.train(X_train, y_train)
```

## Requirements

Only `numpy` is required as specified in [requirements.txt](requirements.txt). 

However if you want to run the examples from [`docs/examples/`](docs/examples/) folder, you will need to install other packages specified in [doc/examples/requirements.txt](docs/examples/requirements.txt).
