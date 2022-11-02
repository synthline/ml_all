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

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from utils import *
```

+++ {"tags": []}

# <i class="fas fa-circle" style="color: #f0ad4e;"></i> Introduction to PCA

In this exercise you will create a basic PCA implementation that you will use for the next exercise.

+++

## Implementing PCA

Your first task is to implement the PCA method as well as functions for
transforming to and from the space defined by the principal components.
But first, a quick recap of the terminology to minimize confusion.

Principal component analysis is about finding a linear transformation
that reduces the number of dimensions used to represent samples while
destroying as little of the variation as possible. PCA is defined by
$\Phi_{:k}$, an $M\times k$ matrix representing a linear transformation from
vectors in $M$-dimensional real space to $k$-dimensional latent space. We have the
following transformations

$$ 
b = \Phi_{:k}^\top x, 
$$

$$
x + \epsilon = \Phi_{:k} b,
$$

where
$x\in\mathbb{R}^M$ is the input vector and $y\in\mathbb{R}^K$ is the
embedded vector. As shown in the second equation, it is possible to reconstruct $x$ with some amount of error $\epsilon$. To find $\Phi$, we use the _eigenvectors_ of the covariance matrix of our data matrix $W$ where each row $i$ is a sample $x_i \in \mathbb{R}^M$. The eigenvectors are sorted by their associated eigenvalues which represent the variance of each dimension in latent space. Selecting the $k$ first columns (we use the notation $\Phi_{:k}$) results in a transformation that reduces the dimensionality of the latent space to $k$ dimensions. 

1.  <i class="fas fa-code"></i> **Setup:** Create a script file or notebook for this exercise. In it, start by loading the face shapes and images using `utils.face_shape_data`.

2.  <i class="fas fa-code"></i> **Implement PCA:** Create a function that calculates and returns the
    principle components of the shapes dataset. Use the method described
    above where the eigenvectors of the covariance matrix is used.
    **Make sure to center the samples (subtract the mean before
    calculating the covariance matrix)**.

3.  <i class="fas fa-code"></i> **Implement transformations:** Create two functions, one for
    transforming from feature space to principal component space
    (eqaution {eq}`trans`) and one for transforming from principal
    component space to feature space
    (equation {eq}`inv`). You have to subtract the $\mu$ vector when
    transforming to the principal component space and add it again when
    transforming back to feature space. You may use the following
    modified equations for reference:

$$
b = \Phi_{:k}^\top(x-\mu)
$$  

$$
x = \Phi_{:k} b + \mu
$$

```{tip}
The reading material for the PCA lecture contains an excellent [tutorial](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html) on how this can be done, but remember that copying is not allowed!!
```

```{tip}
Some of the later tasks will be easier if you return all 146 principle components. You can then create another function for extracting $n$ components to generate $\Phi$.
```

```{code-cell} ipython3
# Write your implementation here.
```

```{code-cell} ipython3
# Load data
path = './db'
shapes, images = face_shape_data(path)

# Write your implementation here.
```

+++ {"tags": []}

## Evaluating precision

As described above, using PCA to transform a sample $x$ to a principal
component space and back again likely results in an error $\epsilon$, called the _reconstruction error_. In this task you will implement a
method for calculating this error and use it to test the effect of
increasing or decreasing the number of principal components used.

When solving regression problems, the error is typically measured as the
average distance error, otherwise known as root mean square error
(RMSE). This is also used when calculating the construction error. For
reference, the RMSE is

$$RMSE(x, \widetilde{x}) = \sqrt{\frac{1}{N}\sum_i (x_i-\widetilde{x}_i)^2},$$

where $x$, $\widetilde{x}$ are the original and transformed samples
respectively and $N$ is the total number of samples $x_i$.

Another method for evaluating PCA models is to look at the eigenvalues,
where eigenvalue $i$ is denoted $\lambda^{(i)}$. The eigenvalues explain
the variance of each dimension when that data has been transformed by
PCA. The sum of all eigenvalues $\lambda^{(1)}+\dots+\lambda^{(n)}$ is
equal to the total variance of the data. By comparing all the
eigenvalues we can calculate:

(1) **Proportional variance:** What proportion of the total variance is
explained by a single component. The following formula can be used

$$\frac{\lambda^{(i)}}{\lambda^{(1)} + \dots + \lambda^{(n)}}$$

(2) **Cumulative proportional variance:** What cumulative proportion of
the total variance is explained by the first $k$ components.

$$\frac{\lambda^{(1)} + \dots + \lambda^{(k)}}{\lambda^{(1)} + \dots + \lambda^{(n)}}$$


1.  **<i class="fas fa-code"></i> Calculate reconstruction error:** Implement a function in your
    script that calculates the reconstruction error given a dataset $X$,
    principle components $\Phi$, and a mean vector $\mu$.

2.  **<i class="fas fa-code"></i> Plot reconstruction error:** When constructing $\Phi$ you may use a single principal component or all of them. Plot the reconstruction error of $\Phi$ for all possible numbers of principle components. An example is shown in {numref}`fig:plot_var`.

3.  **<i class="fas fa-code"></i> Calculate variance:** Create functions that calculate the
    _proportional_ and _cumulative proportional_ variance.

4.  **<i class="fas fa-code"></i> Plot variance metrics:** Plot both the proportional and cumulative
    proportional variance in a single plot. An example is shown in
    {numref}`fig:plot_var`.


```{figure} ./img/plot_var.png
---
name: fig:plot_var
width: 400px
---
The expected result of the cumulative and individual variance
proportion.
```

```{code-cell} ipython3
# Write your implementation here.
```

```{code-cell} ipython3
# Write your implementation here.
```

```{code-cell} ipython3

```