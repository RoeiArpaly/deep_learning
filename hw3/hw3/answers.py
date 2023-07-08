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
    hypers['batch_size'] = 64
    hypers['seq_len'] = 128
    hypers['h_dim'] = 256
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
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


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


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

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
