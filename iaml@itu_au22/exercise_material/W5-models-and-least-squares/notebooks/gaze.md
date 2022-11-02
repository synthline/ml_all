---
jupytext:
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

+++ {"tags": []}

# <i class="fas fa-circle" style="color: #d9534f;"></i> Gaze estimation
<span style="background-color: #d9534f; color: white; border-radius: 10px; padding-top: 2px; padding-bottom: 2px; padding-left: 6px;padding-right: 6px;">mandatory</span> <span style="background-color: #343A40; color: white; border-radius: 10px; padding-top: 2px; padding-bottom: 2px; padding-left: 6px;padding-right: 6px;">notebook</span>
This is the first mandatory exercise which means you will have to hand in this Jupyter Notebook with your implementation and notes. This exercise is split into multiple parts which have to be submitted together. The submission deadline is available on LearnIT.

## Tasks
The following list is a summary of the tasks you need to complete to pass the exercise. Find the tasks in the exercise text with further instructions on what to do. 

<i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> {ref}`gaze:viz` (**A-B**)

<i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> {ref}`gaze:implement` (**A-D**)

<i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> {ref}`gaze:evaluation` (**A-D**)

<i class="fas fa-exclamation-circle" style="color: #d9534f;"></i> {ref}`gaze:improve` (**A-C**)



## Overview

In this exercise you will implement a regression model to estimate where a person is looking (this is known as *gaze*). Gaze estimation is performed by capturing images of a users eye as shown in FIG and mapping them to screen positions. Humans look at things by orienting their eyes so that the point of focus is directly in line with a point on their retinas known as the *fovea* (shown in {numref}`fig-model`). Therefore, the pupil position can be used to infer gaze. 

```{note}
The *fovea* is located slightly differently from person to person ($\pm$ 5 degrees) and as a consequence, a gaze model has to be retrained for every person using it. This difference is shown in {numref}`fig-kappa`.
```


```{figure} ../img/model.png
---
name: fig-model
width: 350px
---
Diagram of a gaze estimation system. The eye, which is directed
at a specific point on the screen is captured by the camera. The two red lines represent an unknown transformation from image to eye and eye to screen. We learn this transformation directly which is shown as $f_{\theta}(x, y)$ in the diagram.
```

In this exercise, $f_{\theta}(x, y)$ is the model mapping pupil positions in images (the $x$ and $y$ parameters) onto screen coordinates (the output of $f_{\theta}$). The model is trained using a set of paired pupil and screen positions. This ground-truth dataset has been collected in advance by asking the participant to look at a specific point on a screen while capturing an eye image. We have detected the pupils for each image using ellipse approximation. 

The next section will introduce you to the dataset.

```{figure} ../img/kappa.jpg
---
name: fig-kappa
figclass: margin
---
Shows the distinction between the visual and optical axes. The optical axis is defined as an axis perpendicular to the lens behind the pupil. The visual axis is personally dependent and is determined by the placement of the *fovea*.
```


### About data

The goal of this exercise is to estimate the gaze of image sequences using a regression model. Each image sequence contains 9 images for calibration and a varying number of images for inference. The calibration samples always represent the same 9 screen positions which form a simple 3 by 3 grid. An example of calibration images are shown in
{numref}`fig-calibration`. For each sequence, you will use the 9
calibration samples to train a regression model and then use the model
to predict gaze positions for the rest of the images.

```{figure} ../img/calibration.jpg
---
name: fig-calibration
width: 60%
---
Calibration images. All image sequences contain 9 calibration images
which all have equivalent gaze positions.
```

`positions.json` contains the ground-truth gaze positions for each image as an array
(stored as $y, x$ for each point). The included image sequences (found
in `inputs/images`) are divided into two groups:

- **No head movement:** `pattern0`, `pattern1`, `pattern2`, `pattern3`

- **Head movement and rotation:** `movement_medium`, `movement_hard`,

You may want to focus on the ones without head movement for now.

```{code-cell} ipython3
:tags: [remove-cell]

import os
import sys
import json
import cv2 as cv
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, NamedTuple, List

import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact
```

```{note}
The real notebook (the one in the materials repository or the one you can download from this page) contains some extra utility code that has been hidden here for brevity. The code is fully commented and we recommend you read it whenever you are in doubt about what is happening.
```

```{code-cell} ipython3
:tags: [remove-cell]

def load_json(folder, filename):
    """Load json file from subdirectory in "inputs/images" with the given filename
    - without .json extension!

    Returns: The json data as a dictionary or array (depending on the file).
    """
    with open(os.path.join(os.path.abspath('../inputs/images/' + folder), f'{filename}.json')) as file:
        data = json.load(file)
        return data

def create_pupil_dataframe(data):
    df = pd.DataFrame(data)
    df.index.name = 'idx'
    return df

def create_glint_dataframe(data):
    rows = [[{'idx': idx, 'num': idx2, 'x': x, 'y': y} for idx2, (x, y) in enumerate(row)] for idx, row in enumerate(data)]
    rows = np.concatenate(rows)
    df = pd.DataFrame.from_records(rows)
    return df

def create_pos_dataframe(data):
    rows = [{'idx': idx, 'x': x, 'y': y} for idx, (x, y) in enumerate(data)]
    df = pd.DataFrame.from_records(rows, index='idx')
    return df

def dist(a, b):
    return np.linalg.norm(a - b)

def center_crop(img, size):
    width, height = size
    i_height, i_width = img.shape[:2]

    dy = (i_height-height)//2
    dx = (i_width-width)//2

    return img[dy: i_height-dy, dx: i_width-dx]
```

```{code-cell} ipython3
:tags: [remove-cell]

def open_img(path, idx):
    """Open a single image from the provided path. The index specifies the image name."""
    img = cv.imread(path + f'/{idx}.jpg')
    if img is None:
        raise IOError("Could not read image")
    return img

def draw_features(img, feature):
    """Helper for drawing pupil and glints onto an image."""
    frame = img.copy()
    feature.pupil.draw(frame, color=(0, 0, 255), thickness=5)
    feature.glints.draw(frame, color=(0, 255, 0), thickness=3, size=20)
    return frame

def load_dataset(folder):
    """Load all images and screen positions for a valid data folder (any folder in "inputs/images")."""
    path = os.path.abspath('../inputs/images/' + folder)

    positions = np.array(load_json(path, 'positions'))
    images = [open_img(path, i) for i in range(len(positions)-1)]

    return list(map(lambda x: Sample(*x), zip(images, positions)))

def predict_features(dataset):
    """Predict eye image features for all images in `dataset`. The dataset is a list of Sample instances."""
    res = []
    for sample in dataset:
        pupil = find_pupil(sample.image)
        glints = find_glints(sample.image, pupil.center)
        res.append(FeatureDescriptor(pupil, glints))
    return res

def show_example_features(images, pupils):
    """Draw a grid of images with the predicted pupils drawn on top."""
    n = len(images)
    cols = 8
    rows = n//8+1

    fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    for i, d in enumerate(images):
        img = d.copy()
        row = pupils.iloc[i]
        img = cv.ellipse(img, (int(row['cx']), int(row['cy'])), (int(row['ax']/2), int(row['ay']/2)), row['angle'], 0, 360, (255, 0, 0), 5)
        ax[i//cols, i%cols].imshow(center_crop(img, (250, 250)))
    for row in ax:
        for a in row:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
    plt.show()
```

(gaze:viz)=
## <i class="fas fa-exclamation-circle mandatory"></i> Data and visualization
First you will familiarise yourself with the problem domain and data sources by doing a number of visualisations. 

The code below loads all the datasets specified in `inputs` and predicts features for each image. The `show_example_features` function then plots a grid of all the images from one of the datasets. 

**A): <i class="fas fa-code"></i>** Test the visualisation for different datasets (by changing the array index when accessing images, positions, or pupils).

```{code-cell} ipython3
# Enter dataset folder name here (any subfolder in inputs/images will do)
dataset_folders = ['pattern0', 'pattern1', 'pattern2', 'moving_medium', 'moving_hard']

# Load detected pupil features
pupils = [create_pupil_dataframe(load_json(f, 'pupils')) for f in dataset_folders]
# Load screen gaze positions
positions = [create_pos_dataframe(load_json(f, 'positions')) for f in dataset_folders]
# Load eye images
images = [[open_img(os.path.abspath('../inputs/images/' + f), i) for i in range(len(p)-1)] for p, f in zip(positions, dataset_folders)]
```

```{code-cell} ipython3
# Create train and test splits
pupils_train = [df.iloc[:9] for df in pupils]
pupils_test = [df.iloc[9:] for df in pupils]
positions_train = [df.iloc[:9] for df in positions]
positions_test = [df.iloc[9:] for df in positions]
images_train = [li[:9] for li in images]
images_test = [li[9:] for li in images]
```

```{admonition} Details about the data format
:class: tip, dropdown
The dataframes each contain a number of columns. Here's a description of what data each column contains.

**Pupil dataframes:**
| Name | Meaning |
| ---- | ------- |
| `cx` | Center $x$-coordinate |
| `cy` | Center $y$-coordinate |
| `ax` | Radius 1 |
| `ay` | Radius 2 |
| `angle` | Angle |

**Position dataframes**
| Name | Meaning |
| ---- | ------- |
| `x` | $x$-coordinate |
| `y` | $y$-coordinate |
```

```{code-cell} ipython3
show_example_features(images_train[0], pupils_train[0])
```

Lets visualize the recorded screen gaze positions. The subject was instructed to look at a red dot on a specific point on a screen while the camera captured each image. We provide the visualisation code for this part:

```{code-cell} ipython3
sns.scatterplot(x='x', y='y', data=positions_test[0]);
```

**B): <i class="fas fa-code"></i>** Test the visualisation for different datasets (by changing the array index when accessing images, positions, or pupils). Do the same but for the detected pupil positions. We provide the `DataFrame` object for each so you should be able to simply modify the example above.

```{code-cell} ipython3
pupils[0].head()
# Write the plotting code here (using Seaborn is easier, but feel free to use Matplotlib as well)
```

+++ {"tags": []}

(gaze:implement)=
## <i class="fas fa-exclamation-circle mandatory"></i> Implement a gaze estimation model

```{figure} ../img/gaze2.jpg
---
name: fig-gaze
figclass: margin
---
Calibration images. All image sequences contain 9 calibration images
which all have equivalent gaze
positions.
```

The mapping function $f_\theta(x, y)$ as shown in
{doc}`../main` is unknown. Because the pupil moves in a spherical curve (this is only true when the head is fixed), the relationship between pupil position in the image and gaze is non-linear. In this exercise, however, you will approximate the gaze mapping by a linear function. Because the function has two outputs, it is easier to train one model for each coordinate. You do this as in the exercises but with one model for the $x$ coordinate and one for the $y$
coordinate. To get the screen coordinates $x'$, $y'$ we have

$$
\begin{aligned}
x' &= ax + by + c\\
y' &= dx + ey + f
 \end{aligned}
$$

**A): <i class="fas fa-pen"></i>** **Construct the design matrix:** Write design matrices for both equations above. Use the previous exercises as a guideline. Answer the following:
- What are the knowns and unknowns in the equations?
- How many parameters does the model have?
- How many points (pupil to gaze point correspondances) do we need to solve the equations?
- What effect does the number of points used have on the solution?

The principle is demonstrated in {numref}`fig-gaze` to the right. Here, the $x$ coordinate of the pupil maps to the $x$ coordinate on the screen. In the real model, we use both $x$ and $y$ as inputs to both the model estimating the $x$ position on the screen and the model estimating the $y$ position.

**B): <i class="fas fa-code"></i>** **Implement the design matrix:** Implement a function for generating a design matrix from pupil positions.

**C): <i class="fas fa-code"></i>** **Calibration:** Learn the parameters $\theta$ for the linear regression using the `pupils_train` and `positions_train` lists (remember to select one of the datasets in the lists). Create a design matrix from the pupil positions. Use *two* linear models, as described above, one to learn the X-coordinates and one to learn the Y-coordinates. 

```{note}
This is possibly the most difficult part of the exercise. Try to use what you learned in the two non-mandatory exercises and apply it here. Remember that you need to fit two separate models, one for each screen coordinate.
```

**D): <i class="fas fa-code"></i>** **Estimation:** Implement a function which predicts the gaze point given a pupil position using the learned models. For reference, the linear model has the form $f(x; \theta)=\theta_0 x_0 + \theta_1 x_1 + \theta_2$. You may calculate the point for each coordinate seperately. Then calculate and return the estimated screen coordinates using the models created during calibration.

```{code-cell} ipython3
:tags: [remove-cell]

# You may use this cell for your implementation.
```

(gaze:evaluation)=
## <i class="fas fa-exclamation-circle mandatory"></i> Evaluation of the regression model

```{code-cell} ipython3
```

```{code-cell} ipython3

```

+++ {"tags": []}

**A): <i class="fas fa-code"></i>** **Calculate errors:** For each dataset, predict gaze positions and calculate the *mean squarred error* between the true values in `positions_test` and the predictions (one MSE for each coordinate). Additionally:
- Calculate the square root of the *mse* for each dataset.

**B): <i class="fas fa-code"></i>** **Calculate distance errors:**
- Calculate the euclidean distance between each predicted point and ground truth screen position. 
- Calculate the mean and variance of the distance error for each dataset. 
- Calculate the distance error for $x$ and $y$ seperately (this is just the absolute value of the error). Then calculate the correlation for the $x$ and $y$ errors for each dataset.
- <i class="fas fa-pen"></i> What does the correlation tell you of the error for each coordinate?
- Visualise the results using a suitable choice of plots (only include plots you think show something valuable about the results).
- <i class="fas fa-pen"></i> Explain why the distance metric is useful for this particular model.

**C): <i class="fas fa-pen"></i>** **Evaluate:**
- How does the model perform? Use both metrics and your visualisations to evaluate the performance.
- Explain your results in the notebook. Don't just save them in variables. 
- Explain any significant differences between the results for each dataset? 
- What would happen if you used the same corresponding points (i.e. dataset) for both training and testing?

**D): <i class="fas fa-code"></i>** **Create visualizations:** Create scatterplots similar to the ones shown earlier in the exercise, but with both ground truth `positions` and predictions. Answer the following:
- <i class="fas fa-pen"></i> Is the linear model a suitable model for this problem? Why/why not? 
- <i class="fas fa-pen"></i> What is the quality of the pupil input points? Are they accurate? What effect does their accuracy have on the final error?


(gaze:improve)=
## <i class="fas fa-exclamation-circle mandatory"></i> Improve the model
Hopefully, you have observed by now that the linear model is not entirely adequate to capture the movement of the pupil. You should understand why this is the case.

This final part of the exercise requires you to modify your linear model into a quadratic model. You have tried this before in one dimension, but here we have two. As before, you will still create one model for each output dimension.

```{note}
It is perfectly possible to create a single model that captures all inputs and outputs. However, we leave it as an optional extra exercise for you to figure out how to do this. Hint: You have to combine the design and parameter matrices in some way for this to work.
```

Since the model is two-dimensional, the quadratic polynomial has a few more factors than for one dimension. The equation for each axis is:

$$
f(x, y) = a\cdot x^2 + b\cdot y^2 + c\cdot xy + d\cdot x + e\cdot y + f.
$$

The design matrices then have the following form:

$$
D_x = D_y = \begin{bmatrix}
 		x_1^2 & y_1^2 & x_1y_1 & x_1 & y_1 & 1\\
  		x_2^2 & y_2^2 & x_2y_2 & x_2 & y_2 & 1\\
  		\vdots &&&&& \\
   		x_2^2 & y_2^2 & x_ny_n & x_n & y_n & 1\\
 	\end{bmatrix}.
$$(dmat)


**A): <i class="fas fa-code"></i>** **Implement model:** Create a new calibration and prediction method that uses quadratic models.

**B): <i class="fas fa-code"></i>** **Evaluate:** Calculate the *rmse* and distance errors as before and compare the two. Visualise errors like you did for the linear model.

**C):** **Compare with linear results:**
- {{ task-impl }} Repeat the evaluation steps for the linear model, i.e. calculate the same metrics and plots. Try to combine the plots for both models to make comparisons easier.
- <i class="fas fa-pen"></i> Use distance means and variance to compare the performance of the linear and quadratic models.
- <i class="fas fa-pen"></i> Which model is best in certain situations and why? Relate this to your knowledge of the problem domain (gaze estimation) and the general problem of choosing model complexity.

```{code-cell} ipython3

```

```{code-cell} ipython3
:tags: [remove-cell]

```

```{code-cell} ipython3

```