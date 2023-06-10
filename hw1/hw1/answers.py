r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1) **False**.
In-sample error, also known as training error, measures the error of the model on the training set.
It represents how well the model fits the training data.
However, it does not provide an accurate estimate of the model's performance on unseen data. 
The goal of machine learning is to create a model that can generalize well to new, unseen data. 
To evaluate the model's performance on such data, we use a separate set called the test set.
The test set allows us to estimate the out-of-sample error, which provides a more reliable measure of the model's 
performance in real-world scenarios.

2) **False**.
Not all splits of the data into two disjoint subsets would constitute equally useful train-test splits.
There are several considerations to keep in mind when creating a train-test split. for example:
* Imbalanced dataset: If the dataset is imbalanced, meaning one class is significantly more prevalent than the others, 
a random split may result in an unequal distribution of classes in the train and test sets.
In such cases, it is important to use techniques like stratification to ensure that the class distribution is preserved in both subsets.
* Shuffling: It is crucial to shuffle the data before splitting it into train and test sets. 
Failure to do so may lead to ordered patterns in the data, causing the model to learn based on that order and perform poorly on new, unseen data.
* Time series data: If the dataset contains time series data, it is essential to consider the temporal aspect. 
The train set should consist of data from earlier time periods, while the test set should contain data from later time periods. 
This ensures that the model is evaluated on its ability to generalize to future time points.
* Size of train and test sets: Extremely small train or test sets may result in poor performance or unreliable evaluation. 
Having an adequate amount of data in both subsets is important for the model to learn meaningful patterns and for the evaluation to be statistically robust.
Taking these factors into account, it becomes clear that not all splits are equally useful, and careful consideration should be given to creating an appropriate train-test split.

3) **True**.
The test set should not be used during cross-validation.
Cross-validation is a technique used to assess the performance of a model by splitting the data into multiple subsets, or folds,
and iteratively training and evaluating the model on different combinations of these folds. 
The purpose of cross-validation is to obtain an unbiased estimate of the model's performance on unseen data.
If the test set were used during cross-validation, it would result in data leakage. 
Data leakage occurs when information from the test set is unintentionally used during model training or tuning, leading to overoptimistic performance estimates.
To prevent this, the test set should be kept completely separate and used only once, after the model has been finalized, to evaluate its performance.

4) **True.**
After performing cross-validation, the validation-set performance of each fold can be used as a proxy for the model's generalization error.
Cross-validation provides a robust estimate of the model's performance by evaluating it on multiple subsets of the data.
By averaging the performance across different folds, we obtain a more reliable indication of how well the model is likely to generalize to new, unseen data.

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

Increasing the number of neighbors (k) in K-nearest neighbors (KNN), has the potential to improve the model's generalization capability 
for unseen data up to a certain threshold. 
By considering more neighbors, the model can incorporate a broader range of information from the training data and make predictions based on a larger pool of similar instances. 
However, there is a point where increasing k too much can have a detrimental effect on performance. 
When the number of neighbors becomes excessively large, the model may start to suffer from underfitting, where it fails to capture the underlying patterns in the data and becomes overly simplistic. 
This is because an excessive number of neighbors can reduce the influence of relevant training instances, leading to less discriminative decision boundaries and decreased accuracy. 

<br> Conversely, decreasing k too much can lead to overfitting, where the model becomes overly sensitive to the training data and fails to generalize well to unseen examples.
In this case, the model might become too specialized to the training set, capturing noise and irrelevant patterns that do not hold true in the broader context. 

<br> Therefore, finding the optimal value for k is crucial in striking a balance between bias and variance, 
and it typically involves experimentation and validation to determine the most suitable number of neighbors for a given dataset and problem.

"""

part2_q2 = r"""
**Your answer:**

Using k-fold cross-validation offers several advantages over the alternative approaches of training on the entire 
train-set and selecting the best model based on either the train-set or test-set accuracy.

1) When training on the entire train-set and selecting the best model based on the train-set accuracy, there is a risk of overfitting.
This approach can inadvertently favor complex models that fit the training data extremely well but may not generalize effectively to unseen data. 
By evaluating models solely on their performance on the training set, we may end up selecting a model that is overly specialized to the idiosyncrasies of the training data, 
resulting in poor performance on new, unseen instances. In contrast, k-fold cross-validation helps mitigate this risk by providing a more robust estimate of the model's performance on unseen data.

2) Similarly, using the test-set accuracy for model selection can also lead to overfitting. 
When we evaluate multiple models on the same test set and choose the one with the best performance, there is a danger of "data leakage". 
The models may inadvertently learn patterns specific to that particular test set, which may not hold true for new, unseen data
Consequently, the selected model may not perform as well when deployed in a real-world setting.
In contrast, k-fold cross-validation provides a better proxy for unseen data because it evaluates the models on multiple partitions of the data,
ensuring that the model's performance is assessed on a more diverse range of instances.

Moreover, k-fold cross-validation reduces the dependence on a single split for model selection. 
By partitioning the data into multiple folds and systematically rotating which fold is used as the validation set, 
we obtain a more comprehensive evaluation of the models across different subsets of the data. 
This reduces the influence of any peculiarities in a particular split, yielding a more reliable estimate of the model's performance.

It is worth noting that k-fold cross-validation is computationally more expensive than using a single split.
Since the model is trained and evaluated multiple times on different folds, it requires more computational resources. 
However, the additional computational cost is justified by the improved reliability and generalization ability of the model.

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
