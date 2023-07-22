r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.3
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.45
    hypers['lr_sched_patience'] = 5

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. SCENE I."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

The main reason that we split the corpus into sequences instead of training on the whole text is that the corpus might be very large,
hence including all the data in the RNN will not fit in memory.
Additionally, training on the whole corpus will cause the RNN to be very deep,
which might cause vanishing gradients and will make the model un-trainable.
Furthermore, splitting the corpus into sequences will help the model to generalize better and prevent over fitting,
because each train iteration will be unique,
and the words in the same sequence will have a strong relation to each other since they are from the same context in the corpus.
Therefore, the RNN will be able to learn patterns and dependencies in smaller sequences instead of the whole text. 

"""

part1_q2 = r"""
**Your answer:**

The hidden state in an RNN can capture and retain information from earlier timesteps,
allowing the generated text to show memory longer than the sequence length.
This is due to the recurrent connections in the network that enable it to maintain in memory the contextual representation of past inputs.

"""

part1_q3 = r"""
**Your answer:**

The order of batches is not shuffled during training of RNNs, due to the fact that the hidden state in an RNN is updated sequentially,
building upon the information from previous timesteps.
Shuffling the order of batches would disrupt the continuity of the hidden state's evolution,
leading to a loss of sequential information and potentially harm the RNN's ability to capture the dependencies in the data.
By maintaining the order of batches,
the RNN can preserve the sequential nature of the hidden state updates and effectively learn from the patterns of the input sequences.

"""

part1_q4 = r"""
**Your answer:**

1) We lower the temperature for sampling because we want to reduce the randomness of the sampling.
lower temperature means that the model will be more confident in its predictions, and will be less random.
it also means it will be more deterministic, and will be more likely to repeat itself, which can be easier to reproduce.

2) When the temperature is very high, the model will be very random, and will be less likely to repeat itself.
The next character will be chosen almost uniformly from the possible characters.
When we increase T,
then $e^{y/T}\rightarrow 1$ and the $\text{hot_softmax}_T(y) = \frac{e^{y/T}}{\sum_k e^{y_k/T}}$ distribution will become more uniform.

3) When the temperature is very low, the model will be very deterministic, and will be more likely to repeat itself.
The next character will be chosen almost deterministically from the possible characters, sampling will be almost like taking the Argmax.
When we decrease T, ${1/T}$ become very large, which will increase the $\text{hot_softmax}_T(y) = \frac{e^{y/T}}{\sum_k e^{y_k/T}}$. 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    hypers['batch_size'] = 16
    hypers['h_dim'] = 64
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = (0.9, 0.999)

    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

$\sigma^2$ is the hyperparameter that control the variance of the likelihood distribution.
Low values of sigma will cause the likelihood distribution to be narrow, hence produce more diverse images with larger variety (in our case, different images of Bosh).
High values of sigma will cause the likelihood distribution to be wide, hence produce less diverse images with smaller variety,
which will be similar to the dataset.
Furthermore, $\sigma^2$ control the weight given to the reconstruction against the KL divergence.

"""

part2_q2 = r"""
**Your answer:**

1. The reconstruction term in the VAE loss measures the difference between the original input and its reconstruction.
By minimizing this term, the autoencoder learns to reconstruct the input.
The KL divergence term in the VAE loss is the KL divergence between the latent space distribution and the prior distribution.
The purpose of the KL divergence loss is used to control the diversity of the 
outputs that produce by the model by comparing latent output to desired distribution.

2. The KL loss term encourages the latent space distribution to be similar to the prior distribution,
and it serves as regularization term to affect the decoder output.
It controls how diverse are the samples from the latent space and how much they are close to normal distribution.

3. The benefit of this effect is that it simplify the model, hence allows it to generalise better (avoiding overfitting).
Therefore, reaching to better result in mapping an output point to its latent space coordinates.

"""

part2_q3 = r"""
**Your answer:**

Our mission is to create an output that is similar to the input after reconstruction.
Maximizing the evidence distribution $p(\bb{X})$, is equivalent to maximizing the log likelihood of the data.
We start by maximizing the log likelihood of the data because we want to maximize the probability of the data given the model.
This is equivalent to minimizing the negative log likelihood of the data given the model.
Therefore, by maximizing the evidence distribution we can get an output distribution that is similar to the input image distribution (from the dataset).

"""

part2_q4 = r"""
**Your answer:**

We model the log of the latent-space variance instead of directly modelling the variance,
because we want to ensure that the training process is numerically stable.
The log function is a monotonic function, which means that it preserves the order of the values, but decreases the magnitude of the values.
Additionally, the log function expands small values, which well help to reduce vanishing gradients.
The log function is also differentiable, which is important for the backpropagation algorithm.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
Based on our experimentation with the fine tuning methods, "Unfreezing all layers" (method 2) outperforms The "Unfreeze The Final 2 Linear Layers" (method 1) due to several reasons.
Firstly, by unfreezing all layers, we allowed the model to fine-tune the entire network, including the lower layers responsible for capturing general features and patterns. 
Unfreezing all layers allows the model to adapt and learn from the specific nuances and patterns the may be related to the movie reviews and their sentiment (IMDB dataset).
Secondly, unfreezing all layers provides the model with more parameters to update during fine-tuning. 
The additional flexibility allows the model to better fit the training data, potentially capturing more intricate details and improving overall performance. 

Lastly, the size of the fine-tuning dataset might have influenced the performance.
With only 25k training datapoints and 480 testing datapoints, the model's ability to generalize might be limited.
By unfreezing all layers, the model has more capacity to capture and memorize the training examples, potentially leading to higher accuracy on the limited testing dataset.

It is also worth mentioning that method 2 (unfreezing all layers) took a significantly longer time to train than "Fine-tuning".

"""

part3_q2 = r"""
**Your answer:**
Fine-tuning the internal layers of a model, such as the multi-headed attention blocks, instead of the last two linear layers,
can have different implications on the performance of the model for a specific task.

1. Improved Performance: Fine-tuning internal layers can potentially improve the performance of the model. 
The multi-headed attention blocks are responsible for capturing contextual relationships and dependencies within the input data. 
By fine-tuning these layers, the model can adapt its attention patterns specifically to the task at hand, 
potentially improving its ability to understand and represent relevant information.

2. Degraded Performance: On the other hand, fine-tuning internal layers other than the last two linear layers may negatively impact the performance of the model. 
The last two linear layers are typically responsible for mapping the distilled representation learned by the internal layers to the specific task labels or outputs. 
By not fine-tuning these layers, the model might struggle to effectively utilize the learned representations to make accurate predictions.

3. Task Dependency: The impact of fine-tuning internal layers can also depend on the specific task being performed. 
Some tasks may benefit more from adapting the internal layers, while others may rely more on the final linear layers for performance. 
It is difficult to make a general statement about all possible tasks, as the effectiveness of fine-tuning internal layers would vary based on the specific requirements and characteristics of the task.

In summary, fine-tuning internal layers other than the last two linear layers can have varied outcomes depending on the specific task and data at hand. 
It may lead to improved performance by allowing the model to adapt its attention patterns, 
or it could result in degraded performance by neglecting the mapping of distilled representations to the task labels. 
The ultimate effectiveness of fine-tuning internal layers should be determined empirically through experimentation and validation on the target task.

However, in the case of the IMDB review dataset, fine-tuning the internal layers of the transformer encoder instead of the last two linear layers will potentially harm
the performance because this task is to classify movie reviews as positive or negative (binary classification) and fine tuning the final linear layers will help the model to
learn the mapping between the distilled representation to the task labels. 
Additionally by only fine tuning the internal layers the model will not effectively utilize the learned representations to make accurate predictions.


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
