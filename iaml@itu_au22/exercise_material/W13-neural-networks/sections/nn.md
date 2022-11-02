# {{ exercise }} Neural network architectures
In the first part of the exercise you will modify a couple of neural network architectures and evaluate their performance. 


## TensorBoard
[TensorBoard](https://www.tensorflow.org/tensorboard) is a tool for monitoring neural network training progress. Although TensorBoard was created for TensorFlow (Google's framework for neural networks), it can also be used for Pytorch. You will use it in this exercise to evaluate model training.

 In a terminal window (after activating the `iaml` environment) execute:

```
pip install --user tensorboard
```

TensorBoard is started from the terminal by typing the following command:

```
tensorboard --logdir=runs
```

```{info}
 Start `tensorboard` inside the exercise source folder for  the logging to work with the relative path  (./runs).  
```
Click or copy the address specified in the terminal to open tensorboard in a browser. The interface is shown in {numref}`tensorboard`.

```{figure} ../img/tensorboard.PNG
---
name: tensorboard
width: 70%
---
Screenshot of the TensorBoard interface showing the validation
accuracies for a number of models.
```


* Set up base MLP and base CNN
* Change number of MLP layers
* Change number of CNN layers
* Eval


## The basic setup
In this exercise you will train and modify a number of neural networks.  TensorBoard allows you to evaluate model loss live. The notebook `evaluation.ipynb`  will  allow you  to evaluate different models using various evaluaton metrics. The script `train_pytorch.py` should be used for network training.

- {{ task-impl }} The model architechtures are defined in  `networks.py`. The training of the models is setup in  `train_pytorch.py`.
  - Run the script `train_pytorch.py`.
  - Test that your TensorBoard configuration is working.
  - Make sure that the model completes training.
- {{ task-impl }} Copy the `train_pytorch_MLPBasic` function and swap the `MLPBasic` model with the `CNNBasic` model.
  - Train the CNN by running the script.
- {{ task-impl }} Compare models in evaluation script:
  - Add the two models to the `models` dictionary in cell 2 in `evaluation.py`. The notebook contains examples to help you with the syntax. Look in the `./models` folder to find your own model filenames.
  - {{ task-q }} Run the notebook code and look at the plots for accuracy and precision.
    - Which model has the best accuracy? Why is this the case?
    - The MLP has trouble on certain classes. Look at the sample images at the bottom of the notebook. Why are these classess more difficult for the MLP?


## Modifications
<!-- Here should be some tasks about modifying the networks and observing the changes.  -->

**MLP:**
- {{ task-impl }} In `networks.py`, copy the `MLPBasic` class and rename it to something else.
- {{ task-impl }} Add a new hidden layer to your new class. You have to both create the layer object in the `__init__` method and call it in the `forward` method. Make sure that the number of input and output features match between the layers.
- {{ task-q }} Train the new network and add it to the evaluation notebook.
  - Is the new model better than the original? 

**CNN:**


- {{ task-impl }} Now copy the `CNNBasic` class and rename it.
- {{ task-impl }} Add a new `Conv2d` layer to the model. This is a bit more tricky than a linear layer because there are more parameters that need to fit between the layers (see tips below).
  - The easiest solution is to decrease the kernel size of the existing first layer and add your new layer as the second layer.
  - Make sure that everything fits together. You will likely need to debug the model a few times to get it right. Just use `tensor.size` to get the size of any tensor.
- {{ task-q }} Train the new network and add it to the evaluation notebook.
  - Is the new CNN better than the original? 


```{tip}
- The kernel size changes the size of the output image. Specifically, for kernel size $k$, the output has dimensions $w-(k-1) \times h-(k-1)$. 
- Each `max_pool2d` operation in `forward` halves the image resolution.
- You can have multiple convolutional layers in sequence without max pooling layers in between. Just remember to use an activation function.
```
## {{ optional }} Additional improvements
{{ task-impl }} Create as many other configurations as you would like. The setup we have provided makes rapid testing of new concepts easy and should help you get a feel for how you should go about designing neural networks.