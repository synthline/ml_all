# Neural networks

```{figure} ./img/logo.png
---
width: 20%
---
```

````{margin}
```{tip}
Please read the whole exercise carefully before starting to
solve it.
```
````

In this exercise, you will experiment with  neural networks for a multiclass-classification using the [PyTorch](https://pytorch.org) library. The goal is to familiarize yourself with the multilayer-perceptrons (MLP) and convolutional neural netoworks (CNN) architectures and in particular how network topology and/or optimisation strategies may impact performance.

We will provide a basic framework that allows you to quickly test different networks and settings, relieving you of the agony of setting up everything yourself. You may, of course, use your own implementation if you like agonoy.

## Overview

This assignment focuses on image classification of clothes. The purpose is for you to experiment with different classification approaches and evaluate the results.

### Dataset

This exercise uses the FashionMNIST dataset. FashionMNIST is a database of clothes article images (from Zalando), consisting of $28\times 28$ pixel grayscale images associated with one of ten classes of clothing articles.
A total of $60,000$ training samples and $10,000$ test samples are provided. 
 A small sample of the images sorted by class is shown in
{numref}`examples`.

```{figure} ./img/examples.jpg
---
name: examples
width: 100%
---
Sample pictures from the FashionMNIST dataset, sorted by class.
```

FashionMNIST is an excellent starting point to experiment with neural architectures since it is not too easy (as
you will see) but is small enough to allow you to train
classifiers within a reasonable timeframe. 

```{note}
The dataset should be downloaded automatically when you run the training script `train_pytorch.py` if you are connected to the internet.
```

### Framework
Most of the code is contained in traditional python scripts. A helper class `PyTorchTrainer` (in `trainers.py`) to perform the training and evaluation duties is provided. Use the following suggestions to get familiar with the overall system (you can skip some detail when you have understood the overall structure). We will go through the exercise in class covering the follwoing steps. 

<!-- Note: This is not actual tasks but simply helpfull suggestions for reading the code. This is not meant to test their understanding. -->

-  {{ taskread }}: Refer to exercise 8.4.3 for an introduction to how PyTorch works and how the `nn.Module` class is used. The networks in `networks.py` are all subclasses of  `nn.Module`.

- {{ taskread }}: The file `networks.py` contains a selection of neural architectures with different topologies.  Inspect the predefined networks in  the file.

- {{ taskread }}: The file  `train_pytorch.py` contains the functionality to train the networks.  Focus on how the network model `BasicNetwork` is used for `PyTorchTrainer` and  the optimizer `optim.SGD`
- {{ taskread }}: Inspect the source code of `PyTorchTrainer` in `Trainer.py` and refer to the docstrings whenever you are in doubt. 

- {{ taskread }}: Inspect the code for `MetricLogger` in `metrics.py`. Read the docstrings to get aquainted with the structure of the class.

```{tip}
Focus on the **networks** and **training setups** because this is what is needed in this exercise.
Currenly you do not have to worry too muvh about the methods in `PyTorchTrainer` and even `MetricLogger`. You can  return and understand these later. 
```


## Metrics for multiclass classification
This exercise will use metrics for multiclass classification problems. The metrics for binary classification are derived from the _confusion matrix_ $C$. The presented problem contains 10 classes and hence the the confusion matrix is $10\times 10$. Figure
{numref}`confusion` shows an example where the true
class is on the x-axis and the predicted class on the y-axis.  $C_{i,j}$  contains 
the number of samples predicted to be class $i$ and with true
label $j$.

```{figure} ../img/confusion.jpg
---
name: confusion
width: 70%
---
Confusion matrix of sample support vector machine. The true class is
on the x-axis and the predicted class on the
y-axis.
```

 Below is a description of the evaluation metrics and how to calculate them:

**Accuracy:** The ratio of correct predictions to the total number of
test samples.

$$accuracy = \frac{\sum_{i=1}^{10} C_{i,i}}{\sum_{i=1}^{10}\sum_{j=1}^{10} C_{i, j}}$$

**Precision:** The ratio of correct predictions for a certain class to
the number of predictions for that
class.

$$precision_i = \frac{C_{i, i}}{\sum_{j=1}^{10} C_{i, j}}$$

**Recall:** The ratio of correct predictions for a certain class to the
number of samples belonging to that
class.

$$recall_i = \frac{C_{i, i}}{\sum_{j=1}^{10} C_{j, i}}$$

Note that precision and recall for multiclass classification are vectors and hence describe the metric per class.

We provide a partial implementation of the class `MetricLogger` in `metrics.py`. The constructor initializes the confusion matrix and `reset()` resets the confusion matrix. `log(predicted, target)` adds the provided results to the matrix. The `one_hot` argument in the constructor is needed since Scikit-learn provides numerical predictions while PyTorch provides one-hot encoded predictions.

