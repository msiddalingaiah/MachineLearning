# MachineLearning
Machine learning models and related examples.

## Anaconda Python

* Install Anaconda version 3.x

Tensorflow 1.13 does not support Python 3.7, so Anaconda must be downgraded to 3.6 to support Tensorflow 1.13.

On Windows, it is necessary to run Anaconda Prompt as Administrator for the following step only:

```
conda install python=3.6
```

For the remaining steps, run Anaconda Prompt normally (Administrator is not required):

```
pip install --upgrade tensorflow
pip install Keras
```
