r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1) The Jacobian matrix represents the partial derivatives of the output with respect to the input. 
In this case, the input tensor has a shape of $(128, 1024)$, indicating a batch size of $128$ and input dimension of $1024$.
The output tensor has a shape of $(128, 2048)$ with 128 samples and an output dimension of $2048$.
The Jacobian tensor is calculated by computing the partial derivatives for each instance separately,
resulting in a shape of $(128, 2048, 128, 1024)$. 
While intuitively it may seem more logical to have a shape of $(128, 2048, 1024)$ to keep derivatives between instances separate,
in practice, the framework like PyTorch considers the Jacobian in the given shape.

2) Each element in the Jacobian tensor is represented by a single-precision floating point, which occupies $4$ bytes ($32$ bits).
To calculate the memory required, we multiply the number of elements in the tensor by the size of each element and convert it to gigabytes.
The total number of elements is given by $(4 \cdot 128 \cdot 2048 \cdot 128 \cdot 1024)$. Dividing this by $ 1024^3 $ converts the result to gigabytes:

$$\begin{align}
\frac{(4 \cdot 128\cdot 2048 \cdot 128 \cdot 1024)} {(1024^3)} = 128 \quad gigabytes
\end{align}$$

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
With no dropout (dropout rate of 0), we observe overfitting, where the training accuracy is significantly higher than the validation accuracy. 
This indicates that the model has memorized the training data and is not generalizing well to unseen data. 
The training accuracy curve steadily increases as the model becomes more complex and fits the training data more closely, 
while the validation accuracy plateaus or even declines as the model fails to generalize.

On the other hand, the dropout configurations (dropout rates of 0.4 and 0.8) reduce the overfitting phenomenon. 
Dropout randomly deactivates a fraction of the neurons during training, forcing the model to learn more robust and less dependent representations. 
This helps in reducing overfitting by preventing the model from relying too heavily on specific features or neurons.

When we observe the graphs for the dropout configurations, we can see that the training accuracy is slightly lower than the validation accuracy. 
This indicates that the model is not overfitting as much as in the case without dropout. The training accuracy may not reach its maximum potential, 
but the validation accuracy is higher, indicating better generalization performance.

2) Comparing the low-dropout setting (dropout rate of 0.4) with the high-dropout setting (dropout rate of 0.8), 
we can observe that both configurations exhibit similar characteristics of slight overfitting. However, the high-dropout setting shows a more pronounced effect. 
The training accuracy remains lower and closer to the validation accuracy, indicating a stronger regularization effect.

A higher dropout rate means a larger fraction of neurons are deactivated during training. 
This results in a more severe form of regularization, as the model is forced to rely on a smaller subset of active neurons for each training instance. 
Consequently, the model becomes more robust and less sensitive to specific features or patterns in the training data. 
As a result, the high-dropout setting leads to better generalization performance and reduced overfitting compared to the low-dropout setting.

In summary, dropout regularization helps in reducing overfitting by randomly deactivating neurons during training. 
The dropout configurations exhibit lower overfitting compared to the no-dropout case, 
with the high-dropout setting demonstrating a stronger regularization effect and improved generalization performance.

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases when training a model with the cross-entropy loss function. The cross-entropy loss function is defined as:

$[-\sum_{i}y_i \log(\hat{y_i})]$

where $(y_i)$ represents the true label and $(\hat{y_i})$ represents the predicted probability distribution for each class.

Unlike accuracy, which only considers whether the maximum probability label matches the correct label, 
the cross-entropy loss takes into account the confidence or certainty of the classifier's predictions.

In some cases, even if the model predicts more instances correctly in the test set, 
the output distribution from the softmax function may become more spread out and closer to a uniform distribution. 
This can lead to an increase in the cross-entropy loss, as the model becomes less certain or less confident about its predictions.

However, the test accuracy may still increase because the model is making correct predictions for a larger number of instances. 
The model's overall performance on the test set improves, but the increase in test loss indicates that the model's confidence or certainty in its predictions has decreased.

To address this issue, regularization techniques or model adjustments can be applied to improve the model's ability to generalize and make more confident predictions.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Your answer:
1) Number of parameters after each convolution is:
$K \cdot (C_in \cdot F^2 + 1) = (ChannelsOut \cdot ((ChannelsIn \cdot width \cdot height) + 1)$
Example 1 - Regular  block
    First Conv layer:
   $  parameters = 256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $ 
   Second Conv layer:
       $ parameters = 256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $ 
   Total parameters = $590,080 \cdot 2 = 1,180,160$
Compared to a Bottleneck Block:<br>
First Conv layer:
   $ parameters = 64 \cdot ((256 \cdot 1 \cdot 1) + 1) = 16,448 $ 
Second Conv layer:
   $ parameters = 64 \cdot ((64 \cdot 3 \cdot 3) + 1) = 36,928 $ 
Third Conv layer:
   $ parameters = 256 \cdot ((64 \cdot 1 \cdot 1) + 1) = 16,640 $ 
Total parameters = $ 16,448 + 36,928 + 16,640 = 70,016 $
Hence, the number of parameters in a regular block is much larger than in a bottleneck block.
<br><br>
2) The number of parameters in a convolutional layer can be calculated using the formula: $2HW\times(C_{in} k_w k_h +1)C_{out}$, 
where $H$ is the input height, $W$ is the input width, $C_{in}$ is the number of input channels, $k_w$ is the kernel width, $k_h$ is the kernel height, 
and $C_{out}$ is the number of output channels. The number of FLOPs (floating-point operations) in a convolutional layer is equal to the number of parameters multiplied by $2HW$.
For the regular ResBlock, which performs two 3x3 convolutions directly on the 256-channel input, the total number of FLOPs is $2HW\times73,856$. 
This means it has a larger number of parameters compared to the bottleneck ResBlock.
On the other hand, the bottleneck ResBlock has a total of $2HW\times70,016$ FLOPs, which is lower than the regular ResBlock.
The bottleneck ResBlock achieves this by first using a 1x1 convolution to reduce the number of channels to a lower value, typically 64, 
and then performing two 3x3 convolutions on the reduced channels. This approach reduces the computational complexity while still capturing important features.

In summary, the regular ResBlock has more parameters and performs approximately $2HW\times3000$ more FLOPs compared to the bottleneck ResBlock, 
which uses a 1x1 convolution to reduce the number of channels before applying the 3x3 convolutions.
<br><br>
3) 
    a. Spatial combination within feature maps: The regular ResBlock demonstrates a higher capacity to combine input within feature maps. 
In the regular ResBlock, each output feature within a single layer depends on nine input features, 
resulting in a total dependency of at least 25 input features for one output feature (assuming a stride of 1). 
Conversely, the bottleneck ResBlock relies on only one input feature for the first and last layers, and nine input features for the second layer, 
resulting in a total dependency of nine input features for one output feature.

    b. Combination across feature maps: The bottleneck ResBlock, due to its specific structure, facilitates the creation of a compact representation of the feature map. 
By projecting the input feature map to a smaller channel size and then back to the original size, it enables efficient combination across feature maps. 
In contrast, regular ResBlocks do not assume any particular structure and function as conventional convolutional layers when it comes to feature map projections.

In summary, the regular ResBlock excels in combining input within feature maps, with each output feature depending on a larger number of input features. 
On the other hand, the bottleneck ResBlock is effective in combining input across feature maps, 
thanks to its distinct structure that allows for a compact representation and projection.
"""

part3_q2 = r"""
**Your answer:**
1. Analyzing the effect of depth on accuracy, it can be observed that increasing the depth (L) does not consistently lead to improved results for both values of K, 
at least within the observed 10 epochs. best performance is achieved when L=4, 
suggesting that increasing the depth beyond a certain point does not contribute to enhancing the outcomes. 
The reasons for this observation could be attributed to the complexity and capacity of the model, as well as the nature of the dataset and the specific task at hand. 
It is possible that the increased depth introduces more parameters and complexity, making it difficult for the model to effectively learn and generalize.<br><br>

2. In the experiments, it was observed that for L=8 (in K=64) and L=16 (for both K values), the network became untrainable. 
This issue is often associated with the "vanishing gradient problem" where the gradients during backpropagation diminish to extremely small values, 
hindering the flow of gradient information to earlier layers of the network. As a result, the network fails to learn effectively and convergence becomes difficult.

To address this problem, two potential solutions can be considered:
    a. Incorporating batch normalization can help alleviate the vanishing gradient problem by normalizing the activations within each mini-batch. 
This normalization process helps in maintaining the derivative within an appropriate range, facilitating the gradient flow and improving training stability.
    b. Another approach to mitigate the vanishing gradient problem is to employ residual blocks.
Residual connections allow the network to bypass certain layers and directly propagate the input to subsequent layers. 
This mechanism enables the network to effectively learn residual information and gradients, ensuring better information flow through the network and mitigating the vanishing gradient problem.

By applying these techniques, the network may become more trainable, allowing for better optimization and potentially improved accuracy.

"""

part3_q3 = r"""
**Your answer:**
When analyzing the results from experiment 1.2 and comparing them to experiment 1.1, several observations can be made. 
Firstly, increasing the value of K (number of channels) in the models leads to improved outcomes. 
he higher capacity and representation power afforded by larger channel sizes allow the models to capture more complex features and patterns, 
resulting in better performance. However, it is worth noting that as K increases, the models become more prone to overfitting, as indicated by the larger gaps between training and validation accuracy.

Additionally, in experiment 1.2, when increasing the depth (L), it was necessary to adjust the hyperparameter and introduce pooling every 4 layers.
This change was made in order to accommodate the smaller image sizes. Despite the increased complexity of the model, this modification helped handle the smaller images effectively. 
Consequently, the slightly better results obtained in experiment 1.2 can be attributed to the increased depth and the adjustment made to address the image size issue.
"""

part3_q4 = r"""
**Your answer:**
When analyzing the results from experiment 1.3, it appears that L=1 is more likely to produce the best outcome. 
This observation aligns with the findings from the first experiment, where increasing the depth beyond a certain point did not lead to improved performance.
As the depth increases beyond L=1, there is a decline in the model's performance.<br>
Notably, when L=4 is utilized, the model becomes excessively complex due to the large number of features involved. 
This complexity likely contributes to the model becoming untrainable, possibly due to the reemergence of the vanishing gradient problem.
These results are consistent with the patterns observed in the previous experiments, reinforcing the notion that excessively deep models may not necessarily yield better outcomes.
"""

part3_q5 = r"""
**Your answer:**
When analyzing the results from experiment 1.4 and comparing them to experiments 1.1 and 1.3, several notable observations can be made. 
Firstly, we can observe that the problem encountered in the previous experiments is not present in experiment 1.4. 
This is attributed to the implementation of ResNet, which incorporates skip connections and allowing for the efficient flow of gradients. 
As a result, we can leverage more complex and deeper networks, leading to improved performance compared to the previous experiments.

Moreover, the results obtained in experiment 1.4 surpass those of the previous iterations, further emphasizing the benefits of the ResNet architecture. 
The combination of skip connections and deeper networks contributes to better model performance. By examining the graph, 
it becomes apparent that both L=16 and L=4 yield optimal results. These values strike a balance between model complexity and simplicity,
leading to a favorable trade-off and achieving optimal performance.

In summary, the utilization of ResNet in experiment 1.4 resolves the issues encountered in previous experiments and enables the use of more complex and deeper networks. 
As a result, the performance improves compared to the previous iterations. 
The optimal choices of L=16 and L=4 demonstrate a balance between model complexity and simplicity, leading to optimal performance.
"""

part3_q6 = r"""
**Your answer:**
1. In the implementation of the `YourCodeNet` class, several modifications were made to enhance the architecture:
    * Dropout regularization: To combat overfitting, dropout was incorporated into the network. 
    By randomly disabling a fraction of the neurons during training, dropout prevents the model from relying too heavily on specific features or patterns.
    The dropout rate was adjusted to find an optimal balance between regularization and preserving important information.
    * Batch normalization: By adding batch normalization layers, the learning process was improved in terms of speed and stability.
    Batch normalization normalizes the activations within each mini-batch, reducing the internal covariate shift and allowing for more efficient gradient flow during training.
    * Residual block implementation: The utilization of a residual block addressed the problem of an untrainable model observed in previous experiments.
    Residual connections enable the network to bypass certain layers and directly propagate the input to subsequent layers.
    This mechanism helps alleviate the vanishing gradient problem, enabling better information flow and improving the training process.<br>
2. Comparing the results of experiment 2 to experiment 1, we observe notable improvements. 
By increasing the number of epochs and incorporating regularization techniques like dropout, the performance of the model in terms of test loss increases. 
The best performance is achieved with L=6, where an accuracy of approximately 77% is attained even with just 10 epochs.

Based on our analysis, we consider L=6 as the optimal parameter choice for this experiment. This choice strikes a balance between model complexity and simplicity, 
enabling the network to capture relevant features and patterns while avoiding excessive complexity that may lead to overfitting. 
By implementing dropout regularization and utilizing a residual block, the `YourCodeNet` architecture achieves improved performance compared to the original `YourCnn` architecture.
"""
# ==============
