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

"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""

part3_q5 = r"""
**Your answer:**

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
