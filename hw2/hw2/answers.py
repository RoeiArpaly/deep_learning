r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

###### CHECK CHECK CHECK CHECK CHECK CHECK CHECK THIS OUT

1) The shape of the Jacobian tensor, representing the derivative of the layer's output with respect to the input tensor,
would be (128, 2048, 1024). This indicates that for each input sample in the batch,
the Jacobian tensor has dimensions corresponding to the output features and input features of the linear layer.

2) Storing the Jacobian tensor, assuming single-precision floating-point representation (32 bits) for the elements,
would require approximately 128 megabytes (MB) of RAM or GPU memory.
This calculation takes into account the total number of elements in the tensor,
which is determined by the batch size (128), output features (2048), and input features (1024) of the linear layer.

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr_vanilla = 0.025
    lr_momentum = 0.0025
    lr_rmsprop = 0.00016
    reg = 0.01
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1) In the comparison between no dropout and dropout configurations, the graphs confirm our expectations.
With no dropout (dropout rate of 0), we observe overfitting, where the training accuracy is significantly higher
than the validation accuracy.
This is evident as the training accuracy curve steadily increases while the validation accuracy plateaus or even
declines.
In contrast, the dropout configurations (dropout rates of 0.4 and 0.8) reduced the overfitting while dropout of 0.8
even showed just a slight of overfitting.
This aligns with our anticipation,
as dropout helps in reducing overfitting by randomly deactivating neurons during training.

2) Comparing the low-dropout setting (dropout rate of 0.4) with the high-dropout setting (dropout rate of 0.8),
we can observe that both configurations exhibit similar characteristics of slight overfitting.
However, the high-dropout setting shows a more pronounced effect, as the training accuracy remains lower and closer to
the validation accuracy. This indicates that a higher dropout rate leads to a stronger regularization effect,
resulting in a better generalization performance and reduced overfitting.

"""

part2_q2 = r"""
**Your answer:**

Yes, When training a model with the cross-entropy loss function,
it is typically expected for the test loss to decrease while the test accuracy increases.
However, in cases of overfitting, the test loss may start to increase even as the test accuracy improves
(but it sacrifices the generalisation).
This can indicate of a decline in the model's performance on unseen data.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
1.

Number of parameters after each convolution is:

$K \cdot (C_in \cdot F^2 + 1) = (ChannelsOut \cdot ((ChannelsIn \cdot width \cdot height) + 1)$

Example 1 - Regular  block
    First Conv layer:
   $  parameters = 256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $ <br>
   Second Conv layer:
       $ parameters = 256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $ <br>
   Total parameters = $590,080 \cdot 2 = 1,180,160$

Compared to a Bottleneck Block:<br>
First Conv layer:
   $ parameters = 64 \cdot ((256 \cdot 1 \cdot 1) + 1) = 16,448 $ <br>
Second Conv layer:
   $ parameters = 64 \cdot ((64 \cdot 3 \cdot 3) + 1) = 36,928 $ <br>
Third Conv layer:
   $ parameters = 256 \cdot ((64 \cdot 1 \cdot 1) + 1) = 16,640 $ <br>
Total parameters = $ 16,448 + 36,928 + 16,640 = 70,016 $

Hence, the number of parameters in a regular block is much larger than in a bottleneck block.<br><br>

2.
FLOPs (floating point operations) in a convolution layer and the ReLU activation function can be calculated as follows:
To compute a convolution layer between input dimensions (C_in, H_in, W_in) and output dimensions (C_out, H_out, W_out), 
we perform 2 * C_in * k^2 * C_out * W_out * H_out floating point operations.<br>
The factor of 2 accounts for both the multiplication and addition operations involved in the convolution.<br>
Additionally, the ReLU activation function in each layer requires H_out * W_out FLOPs.
Comparing a regular block to a bottleneck block, we observe that the regular block involves significantly more FLOPs. 
Consequently, the bottleneck block is much more computationally efficient.

3.
Spatially, the regular block consists of two convolution layers with a filter size of 3x3, 
resulting in a receptive field of 5x5. In contrast, the bottleneck block comprises two convolution layers with a filter 
size of 1x1 and one convolution layer with a filter size of 3x3, resulting in a receptive field of 3x3.<br>
Therefore, the regular block combines the input in a more comprehensive manner in terms of spatial information.<br><br>

Regarding the influence across the feature map, the bottleneck block projects the input to a smaller dimension in the first layer. 
As a result, not all inputs have the same influence across the feature map. In contrast, the regular block does not perform such projection,
allowing the input to have equal influence across the feature map.<br><br>
"""

part3_q2 = r"""
**Your answer:**
1. The graph indicates that increasing the depth (L) does not necessarily lead to improvement for both values of K (at least as we can see from 10 epochs).
When L=4, the best result is obtained, implying that increasing the depth does not contribute to enhancing the outcomes.
On the other hand, for L=8 and L=16, the model becomes untrainable in K=64, which will be further explained in the following section.<br><br>

2. it is true that for L=8 (in K=64), and L=16 (for both K's), the model becomes untrainable. <br>
"vanishing gradient problem" might be the underlying cause for this issue. 
This phenomenon occurs in deep networks when the gradient approaches zero, 
resulting in the failure of gradient information to propagate back to the initial layers during the backpropagation process.<br><br>

Here are two suggestions for the solution to this problem:
1. Incorporating batch normalization can assist in maintaining the derivative within an appropriate range.
2. Employing a residual block can help overcome this issue by allowing the network to skip layers.

"""

part3_q3 = r"""
**Your answer:**
Observing the graphs, it becomes evident that increasing the value of K leads to improved outcomes but the models are more likely to overfit.
Furthermore, when increasing L we had to change the hyperparameter to pool every 4 layers in order to handle the small images.<br>
Due to the higher complexity of the model we achieved a slightly better result compared to the first experiment.
"""

part3_q4 = r"""
**Your answer:**
Upon examining the graph, it is more likely that L=1 will yield the best result.
The explanation for this phenomenon aligns with what we observed in the first experiment.
Subsequent to L=1, there is a decline in performance, and when L=4 is employed, the model becomes overly complex due to the large amount of features. 
Consequently, the model becomes untrainable, likely due to the reemergence of the vanishing gradient issue. 
These outcomes are consistent with the results obtained from the previous experiments.
"""

part3_q5 = r"""
**Your answer:**
We will begin by mentioning that we do not encounter the same problem as previous experiments.
We attribute this to the implementation ResNet, which allows the network to skip layers.
Consequently, we can leverage more complex and deeper networks, leading to improved performance compared to the previous experiments.<br>
Furthermore, the results obtained in this experiment surpass those of previous iterations, likely due to the above reasons.<br>
Based on the graph, it is easy to see that L=16 and L=4 are the optimal choices.
These parameters strike a balance between model complexity and simplicity, resulting in optimal performance.
"""

part3_q6 = r"""
**Your answer:**
1. In the implementation of YourCnn, we made the following decisions:

    a. Incorporating dropout: This technique was employed to mitigate overfitting on the training set. 
    By adjusting the value of dropout, we fine-tuned its impact to achieve an optimal performance.
    
    b. Applying batch normalization: This step was crucial in improving the speed and stability of the learning process.
    
    c. Utilizing a residual block: The inclusion of a residual block addressed issues encountered in previous experiments,
    specifically the problem of an untrainable model.

2. Compared to the first experiment (YourCnn), by observing the test loss we can see that the performance of the model can increase by adding more epochs,
due to the regularization of the dropout parameter. 
The best performance was achieved with L=6, attaining an accuracy of approximately ~77% with just 10 epochs.

Based on our analysis, we believe that L=6 is the optimal parameter choice.
This selection also strikes a balance between model complexity and simplicity as mentioned in the previous questions
"""
# ==============
