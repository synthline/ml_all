# {{ exercise }} Neural network architectures

In this exercise you will experiment with and evaluate  neural architectures. 
`evaluation.ipynb` contains a workflow to easily evaluate binary classification models through accuracy, precision, and recall matrix. Notice that the values to calculate the metrics are based on the confusion matrix.

## Metrics for multiclass classification

 In this exercise the problem  will contain $10$ classes and the confusion matrix is consequently
$10\times 10$. An example is shown in
{numref}`confusion` with the  true
class on the x-axis and predictions in the y-axis.

```{figure} ../img/confusion.jpg
---
name: confusion
width: 70%
---
Confusion matrix of sample support vector machine. The true class is
on the x-axis and the predicted class on the
y-axis.
```

The metrics can be derived from confusion matrix $C$, where $C_{i,j}$ is
the number of samples predicted to be class $i$ and with ground-truth
label $j$. The multiclass versions are generalisations of the metrics used for binary classification. Below is a description of each metric along with a formula for its calculation:

**Accuracy:** The ratio of the correct predictions and the total number of
test samples.

$$accuracy = \frac{\sum_{i=1}^{10} C_{i,i}}{\sum_{i=1}^{10}\sum_{j=1}^{10} C_{i, j}}$$

**Precision:** The ratio of the correct predictions for a given class and
the number of predictions for the class.

$$precision_i = \frac{C_{i, i}}{\sum_{j=1}^{10} C_{i, j}}$$

**Recall:** The ratio of correct predictions for a the class and the
total number of samples in the
class.

$$recall_i = \frac{C_{i, i}}{\sum_{j=1}^{10} C_{j, i}}$$

Note that in multi-class classification that the metrics (precision and recall) are described for each class and hence are vectors when evaluating multiclass classifiers.

The constructor in class `MetricLogger` (`metrics.py`) initializes the confusion matrix and the method  `reset()` resets the confusion matrix (sets it to 0). `log(predicted, target)` adds the provided results to the matrix. The `one_hot` argument in the constructor is added because Scikit-learn provides numerical predictions while PyTorch provides one-hot encoded predictions. One Hot Encoding is a common way of encoding categorical values (e.g. 1,2,3...N) into a binary vector which is one at the corresponding index. For example for a 4 class problem, class 1 will be represented as $[1,0,0,0]$, class 2 is encoded as $[0,1,0,0]$ and class 4 is encoded as $[0,0,0,4]$

## Evaluating results

All should now be setup for you to train and evaluate different classification models. When
running the training-script, the `save()` method will automatically creates a file
containing the complete model state in the `models` directory. You have to
evaluate the models and produce visualisations for the
`evaluation.ipynb` notebook.

1.  <i class="fas fa-exclamation-triangle required"></i> **Train different models:** Train one model for each of the
    following algorithms:

    - <i class="fas fa-exclamation-triangle required"></i> **Logistic regression:** Use `sklearn.linear_models.LogisticRegression`.

    - <i class="fas fa-exclamation-triangle required"></i> **Support vector machine (linear kernel):** Use `sklearn.svm.LinearSVC`.

    - <i class="fas fa-exclamation-triangle required"></i> **Support vector machine (polynomial kernel):** Use `sklearn.svm.SVC(kernel='poly')`

    - <i class="fas fa-exclamation-triangle required"></i> **K nearest neighbors:** Use `sklearn.neighbors.KNeighborsClassifier`.

    - <i class="fas fa-exclamation-triangle optional"></i> Try variations of the above or other models you might have heard
      of. Use [Scikit-learn's user guide for inspiration](https://scikit-learn.org/stable/supervised_learning.html).

2.  <i class="fas fa-exclamation-triangle required"></i> **Produce figures:** The provided Jupyter notebook `evaluation.ipynb` contains
    functionality to produce  visualisations. Add your models
    to the dictionary as specified and produce initial results.

```{figure} ../img/result.jpg
---
name: resulut
width: 100%
---
Sample of wrong results. The titles are formatted as `<prediction>/<ground-truth>`.
```

## <i class="fas fa-exclamation-triangle required"></i> Report

- <i class="fas fa-exclamation-triangle required"></i> Explain the most important characteristics of each model used.
  Remember to be precise and add references when relevant.

- <i class="fas fa-exclamation-triangle required"></i> Present and compare results of the different models.

- <i class="fas fa-exclamation-triangle required"></i> Create bar-graphs for the recorded metrics (accuracy, precision,
  recall) and include an example confusion matrix and image grid ANTON ???????????? with
  predictions and labels.

[^duck-label]: This is actually an example of Python's duck-typing system. The
    classes don't implement a common interface but just provide the
    method.
