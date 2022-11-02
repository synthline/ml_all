#  {{ exercise }} Tuning

In this exercise you will investigate tehchiques that can help to improve
your models and in particular you will be experimenting with techniques to avoid overfitting such as

  * Dropout layer
  * Dropout layer
  * Early stopping
  * Data augmentation
Like in the first exercise the `PyTorchTrainer` method must be used to set up training the different architechtures. 
  
As mentioned in the [youtube video](://youtu.be/njKP3FqW3Sk?thttps=2807)
used for lecture 10, dropout layers and early stopping are useful methods constrain the
optimisation to counteract overfitting.

Just as in the first exercise the `PyTorchTrainer` method must be used to set up training the different architechtures. 
  
## <i class="fas fa-exclamation-triangle important"></i> Dropout layer

In this task you will implement dropout in the  the `TopCNN` and `CNN4layer` models. 

```{tip}
PyTorch provides the layer [nn.Dropout2d](https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d) for convolutional layers and [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout) for fully connected layers.
```

- **Code:** The model (class) `TopCNN`  in `networks.py`  already have dropout layers implemented. Identify the dropout layers  in the architechtcure test the model performance for the following:
- Uncomment the line with the dropout layers in `TopCNN`.
- Experiment with  the probability parameter, $p$,  of the dropout layers and determine which setting gives the best results
- Insert a drop out layer in the `CNN4Layer` model in `networks.py`. 
- Compare the model performance with and without drop out.
- **Preparation for the exam :** In the notebook, commenent on  how the performance made above compares to the model in  `TopCNN`.
 
## <i class="fas fa-exclamation-triangle important"></i> Early stopping
- Identify the location in the `train` method of the `PyTorchTrainer` class in ´trainer.py´
- Uncomment the lines for early stopping.
- use the loss of the validation set to determine when to stop
- change the `patience` parameter and see how it change the performance of using early stopping. 
- **Preparation for the exam :** Produce figures for the model results with
  the models above and add text to describe your observations e.g. how do dropout and early stopping
  influence the results and try to explain the results? 

## <i class="fas fa-exclamation-triangle important"></i> Data augmentation

In this task you will implement data augmentation using
PyTorch's `torchvision` library. Data augmentation involves subjecting the current training batch data to random transformation / alterations. This effectively
creates new training samples for free, which is commonly used to balance uneven datasets. 


`train_pytorch.py` already contains the methods to transform the input images
(in PIL format) to PyTorch tensors.
Your task in to add new augmentations to the transformation. 

Use the function [torchvision.transforms.RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine) to randomly subject the images to affine transformation. 

The conversion of the image data to torch tensors in addition to the affine transformation can be combined with the [torchvision.transforms.Compose](https://pytorch.org/vision/stable/transforms.html?highlight=compose#torchvision.transforms.Compose) method.

```{tip}
An example of how multiple transformations for data augmentation can be composed is shown in the `train_pytorch_MLPBasic` method in `train_pytorch.py` but commented out right now.
```

 **Code:** Implement the training method of the `TopCNN` and `CNN4layer` models with different types of data augmentation, starting with random affine transformations. [Examples are show here.](https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py)

That includes the following steps: 
  - Train the `TopCNN` and `CNN4layer` networks . 
  - Compile a transformation object in `train_pytorch.py` using the `RandomAffine`. **Remember** to end each transformation object with `transforms.ToTensors`.
  - Retrain the network using data augmentation
  - Experiment with adding other types of data augmentation as well.
  - Discuss (write text) how the data augmentation impact the results and the training. Support your arguments with figures.
