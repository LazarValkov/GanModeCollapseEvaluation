# Mode Collapse Evaluation in GANs
Code for training Generative Adversarial Networks (GANs), and evaluating the models' mode collapse.
The code was taken and adapted from https://github.com/carpedm20/DCGAN-tensorflow .

##### The following models are supported:
  - DCGAN (https://arxiv.org/pdf/1511.06434.pdf)
  - ALI (Adversarially Learned Inference, https://arxiv.org/pdf/1606.00704.pdf)
  - Unrolled GAN (https://arxiv.org/pdf/1611.02163.pdf)


##### Evaluating mode collapse using the following 2 methods, introduced in https://arxiv.org/pdf/1611.02163.pdf :
  - Number of modes covered with the Stacked Mnist dataset (section 3.3.1, Discrete Mode Collapse)
  - Inference via Optimisation (section 3.4.1)

## Requirements:

- Tensorflow
- SciPy
- Keras  (only used to import keras.optimizers.Adam )


## Datasets
- Stacked MNIST - can be created by running StackedMNIST_data_setup.py
- CIFAR10 - download the python version from  http://www.cs.toronto.edu/~kriz/cifar.html and place the content in /data/cifar10

## Running

The Run_*.py files contain preset configurations for each model/dataset.
