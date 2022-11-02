import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import required libraries
import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt


# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, 
                    noise=0.03, 
                    random_state=42)

circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
circles.head()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);