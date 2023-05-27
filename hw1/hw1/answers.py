r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1) **False.**<br>
In-sample error (which is also known as training error), is the error on the training set,<br>
and it is not a good measure of the model's performance on unseen data.<br>
The test set is used to evaluate the model's performance on unseen data.<br>
Hence, the test set allows us to estimate the out of sample error and not the in-sample error.<br>


2) **False.**<br>
The following are all examples of bad data splits:<br>
Imbalanced dataset may require stratification.<br>
Splitting without shuffling is a bad practice.<br>
With time series data, the train set time is earlier then the test set time.<br>
Also, very small train set or very small test set, may result in bad performance or bad out of sample evaluation.<br>

3) **True.**<br>
The test set should not be used for cross validation,<br>
because it is used to evaluate the model's performance on unseen data.<br>

4) **True.**<br>
The validation set can be used as a proxy for the model generalization error,<br>
Because it is used to evaluate the model's performance on unseen data during the training process,<br>
with different train, validation splits.<br>

"""

part1_q2 = r"""
**Your answer:**

**False.**<br>
This is not a justified approach.<br>
Although it is common to use the validation set to select hyper parameters,
by using the test set for the model selection, the approach actively leaks information
from the test set into the training process.<br>
This method leads to an overly optimistic evaluation of the model's performance.<br>

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing the number of neighbors (k), can lead to better generalisation for unseen data up to a certain point.<br>
However, if we increase k too much, we will start to see a decrease
in performance (too many neighbors, underfitting).<br>
On the other hand, if we decrease k too much, we will start to see a decrease in performance due to overfitting.<br>

"""

part2_q2 = r"""
**Your answer:**

1) Using k-fold cross validation is better than using the training score of a specific split,
Evaluating the model based on the training score of a specific split may lead to overfitting.<br>
Especially in the case of selecting k neighbors of 1 (100% accuracy).<br>

2) Using k-fold cross validation is better than using the test score of a specific split,
both because it gives a better proxy for the unseen data and the model selection is not based on a specific split.<br>
Furthermore, evaluating the model based on the test score of a specific split may lead to overfitting,
as we select the hyper parameters based on a specific test score,
which can lead to a very specific generalisation based on that split.<br>
However, cross validation is computationally more expensive than using a single split.<br>

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The selection of the regularization parameter $\Delta$ in the SVM loss function is arbitrary because it 
is a hyperparameter that needs to be set before training the model.<br>
It controls the trade-off between classification error and weight magnitude,
and different values can lead to different levels of performance.<br>
The appropriate value of $\Delta$ can be determined through cross-validation or other model selection techniques.<br>
Changing the delta will not change the hyperplane, but it will change the margin.<br>

"""

part3_q2 = r"""
**Your answer:**

1) In the weights_as_images() function, the model identifies the pixels that are most likely to be associated
with each label.<br>
This is done by assigning higher weights to the pixels that appear frequently
in the training images for a particular label.<br>
In the tensors_as_images() function, we can better understand why the model makes certain mistakes,
such as when there are white pixels in locations where the model expects black pixels based on the label.

2) The key difference between KNN and SVM is that KNN considers only the K nearest neighbors when making a prediction,
while SVM takes into account all of the data points.<br>
However, both algorithms use distance based decisions to make their predictions.<br>

"""

part3_q3 = r"""
**Your answer:**

1) Good.<br>
- When the learning rate is too low, the model will take a long time to converge,
and loss will decrease slowly in the loss graph.<br>
- When the learning rate is too high, we will see spikes in the loss graph,


2) The model is slightly overfitting.<br>
We can see that the training loss is lower than the validation loss.<br>
And the accuracy on the training set is higher than the accuracy on the validation set.<br> 

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


The ideal residual plot is one with values that are closely distributed around the horizontal axis (y).<br>
The values should be symmetric around zero, if they are not symmetric,
it means that the model is biased (underestimating or overestimating).<br>

We can see that the residual graph of the final model has improved and behaves according to an ideal residual plot.<br>
While the top 5 features residual plot is not ideal,
the values are not symmetric around the horizontal axis,
hence, the model is biased (due to the non-linear relationship.<br>  

"""

part4_q2 = r"""
**Your answer:**

1) The model is still linear because it is still built based on linear combination of input features.<br>
Although some of the features are not linear, it is possible to model the polynomial features in a linear fashion
(due to the degree of the features and the feature interaction).<br>

2) By adding non-linear features (polynomial features),
we are able to fit any non-linear function of the original features.<br>
In practice, the success of this approach depends on the data and the complexity of the non-linear function.<br>

3) By adding non-linear features it will be possible to create a
linear separation with an hyperplane in higher dimension.<br>
However, the decision boundary (in lower dimension) 
can take different shapes and forms and is not necessarily linear.<br>  

"""

part4_q3 = r"""
**Your answer:**

1) By using a log space instead of linear space,
Lambda is the hyper-parameter that controls the power of the regularization term.<br>
By using a log space instead of linear space, to define a range of values for lambda,
we get numbers in log scale that distribute more low values and less high values.<br>
Since regularization terms are usually small positive numbers, we need to explore a range of low lambda values,
but also check some high values to ensure we don't miss the best one.<br>

2)<br>
- parameter grid: 20 (lambda_range), 3 (degree_range)<br>
- k-folds: 3<br>
- total fits: $20 * 3 * 3 = 180$<br>

"""

# ==============
