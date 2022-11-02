---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# <i class="fas fa-circle" style="color: #d9534f;"></i> PCA for shape generation
In this mandatory exercise you will use PCA to generate face shapes. The goal is to have you explore the relation between points in latent space and real space (which in this instance is faces). 

```{warning}
Please read the introduction to PCA in exercise 10.1 carefully and solve the exercises before attempting to solve this assignment.
```


```{note}
<i class="fas fa-keyboard" style="color: #d9534f;"></i> markers indicate where you should write your answers to each question. All enumerated tasks (tasks with preceeding letters) has an answer field if it requires answers in prose.
```

The exercise is subdivided into the following two sections:

- <i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> {ref}`pca:gen` (questions A-J)
- <i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> {ref}`pca:reflect` (questions A-B)

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2 as cv

import os
from glob import glob

from utils import *
```

## PCA implementation
Copy your PCA implementation into the following cell:

```{code-cell} ipython3
# Copy your PCA implementation to here.
```

+++ {"tags": []}

(pca:gen)=
## <i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> Using PCA for shape generation

As shown in the lecture, PCA can be used as a generative process, where
new samples are generated. Generating a new face $x$ can be done by using $x = \Phi_{:k} y + \mu$, where $\Phi$ is the matrix of eigenvectors and $y$ is a vector in latent space.

We start at $y=\mathbf{0}$. The variance in each direction can be used to define reasonable boundary values for each principal component. Since $y$ is in latent space, we need to calculate the variance of our transformed data. For each feature space dimension, a range of $[-3\sigma_i, 3\sigma_i]$ is used where $\sigma_i = \sqrt{Var(y_i)}$. Assuming the faces are normally distributed, $\pm 3\sigma$ covers $99.7\%$ of all points. This is shown in {numref}`fig:normal`. 

```{figure} ./img/normal.png
---
name: fig:normal
width: 400px
---
Illustration of a normal distribution. The shaded regions indicate the
area of the distribution covered by 1, 2, 3, and 4, standard deviations
($\sigma$) respectively.
```



A): <i class="fas fa-code"></i> **Get principal components:** Repeat what you did in the last exercise and compute the principal components $\Phi$ for the face dataset.

```{code-cell} ipython3
# Implementation for A)
# Get the principal components for the dataset and transform the faces to feature space.
```

B): 
- <i class="fas fa-code"></i> **Generate vectors:** Create a zero vector with $5$ dimensions, i.e. $y = [0, 0, \dots, 0]$. This will be your starting point. 
- <i class="fas fa-code"></i> **Shape generation:** Use $x = \Phi_{:5} y + \mu$ to generate a vector in real space from the zero vector (you may reuse the implementation from the exercise). Use the provided utility function `plot_face` to visualise the generated face as a plot.

```{code-cell} ipython3
# Implementation for B)
# std = transformed.std(axis=1)
```

C): <i class="fas fa-pen"></i> **Base face:** Describe what the face generated using the zero vector represents and why this is the case. You should be able to deduce why from your knowledge about PCA.

**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...

+++

D): 
- <i class="fas fa-code"></i> **Varying components:** Start experimenting with changing the vector component values and watch the effect it has on the output.
- <i class="fas fa-code"></i> **Component controls:** Interpolate 5 values in the range $[-3\sigma_1, 3\sigma_1]$ for the first component $y_1$, generate the faces and visualise them.

```{code-cell} ipython3
# Implementation for C)
vectors[:, :5]

# Write your solution here
```

E): {{ task-writing }} **More components:** Do the same for each of the next four components, i.e. $b_2$-$b_5$. Write down what each component changes in the image (for example, the first component may change the height of the face). 

**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...

F): {{ task-writing }} **Extreme values:** What happens if you select values outside the suggested range. Explain why this happens.

**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...

+++

G): <i class="fas fa-code"></i> **Transform data:** For this task, use your $\Phi_{:k}^\top$ to transform the database faces into feature-space (using $x' = \Phi_{:k}^\top(x-\mu)$).

```{code-cell} ipython3
# Implementation for G)
```

H): <i class="fas fa-code"></i> Plot the two first components in latent space using a scatterplot.
- {{ task-writing }} What does this plot tell you about the distribution of faces?
- {{ task-writing }} Experiment with showing different components. Describe how the plot changes.

**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...

```{code-cell} ipython3
# Implementation for H)
```

I): <i class="fas fa-code"></i> **Shape Interpolation:** Create two different faces by selecting two points in latent space. It doesn't matter how many components you use or set to non-zero values. Now you should `np.linspace` to interpolate between the two point. Use `num` to control the number of output values. Plot the resulting faces. Below is an example of how to interpolate two vectors.

```{code-cell} ipython3
d = np.linspace(np.array([1, 2]), np.array([3, 5]), num=5)
plt.scatter(d[:, 0], d[:, 1])
```

J): {{ task-writing }} **Behaviour:** Assume a set of points on a line in latent space. Argue for why the same set of points, when mapped to real space using $x = \Phi_{:k}y + \mu$ also lie on a line.

**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...

+++

(pca:reflect)=
## <i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> Reflection
Describe and reflect on the results gathered from various parts of the exercises for week 10 (including the non-mandatory ones). Answer the following questions (and back up your arguments with results or theory)

A): {{ task-writing }} What did you think of the results from the generative model? Explain why so few principal components are needed to reproduce the samples quite accurately.
  
**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...

 
B): {{ task-writing }} How many components do you need to have $50\%$, $60\%$, $80\%$, $95\%$ of the total variation?

**<i class="fas fa-keyboard" style="color: #d9534f;"></i> Answer**

...