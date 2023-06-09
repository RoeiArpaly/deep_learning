import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        for i in range(len(self.channels)):
            layers.append(nn.Conv2d(in_channels, self.channels[i], **self.conv_params))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            in_channels = self.channels[i]
            if (i + 1) % self.pool_every == 0:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            model = self.feature_extractor(torch.zeros(1, *self.in_size))
            return torch.prod(torch.tensor(model.size()))
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        layers.append(nn.Flatten())
        for i in range(len(self.hidden_dims)):
            layers.append(nn.Linear(n_features, self.hidden_dims[i]))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            n_features = self.hidden_dims[i]
        layers.append(nn.Linear(n_features, self.out_classes))
        # ========================

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        padding = kernel_sizes[0] // 2
        activation = ACTIVATIONS[activation_type](**activation_params)

        main_layers = [nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=kernel_sizes[0], padding=padding)]

        for in_channel, out_channel, kernel_size in zip(channels[:-1], channels[1:], kernel_sizes[1:]):

            if dropout > 0:
                main_layers.append(nn.Dropout2d(dropout))
            if batchnorm:
                main_layers.append(nn.BatchNorm2d(in_channel))

            main_layers.append(activation)
            padding = kernel_size // 2

            main_layers.append(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=padding))

        if in_channels == channels[-1]:
            layers_shortcut = [nn.Identity()]
        else:
            layers_shortcut = [nn.Conv2d(in_channels=in_channels, out_channels=channels[-1], bias=False, kernel_size=1)]

        self.shortcut_path = nn.Sequential(*layers_shortcut)
        self.main_path = nn.Sequential(*main_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======
        new_channels = [inner_channels[0]] + inner_channels + [in_out_channels]
        new_kernel_sizes = [1] + inner_kernel_sizes + [1]
        super().__init__(in_channels=in_out_channels, channels=new_channels, kernel_sizes=new_kernel_sizes, **kwargs)
        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        # Loop over groups of P output channels and create a block from them.

        residuals_kernels = self.pool_every * [3]
        iterator = 0
        cur_in = in_channels
        cur_channels = self.channels[:self.pool_every]
        while iterator <= (len(self.channels) - self.pool_every):

            layers.append(ResidualBlock(in_channels=cur_in,
                                        channels=cur_channels,
                                        kernel_sizes=residuals_kernels,
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

            layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

            iterator = iterator + self.pool_every
            cur_in = cur_channels[-1]
            cur_channels = self.channels[iterator:iterator + self.pool_every]

        if iterator < len(self.channels):
            cur_channels = self.channels[iterator:]
            residuals_kernels = len(cur_channels) * [3]

            layers.append(ResidualBlock(in_channels=cur_in,
                                        channels=cur_channels,
                                        kernel_sizes=residuals_kernels,
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, *args, **kwargs):
        """
        See ConvClassifier.__init__
        """
        self.batchnorm = True
        self.dropout = 0.15

        super().__init__(*args, **kwargs)

        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        pass
        # ========================

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):

        self.pooling_params = {'kernel_size': 2}

        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        residuals_kernels = self.pool_every * [3]
        iterator = 0
        cur_in = in_channels
        cur_channels = self.channels[:self.pool_every]
        while iterator <= (len(self.channels) - self.pool_every):

            layers.append(ResidualBlock(in_channels=cur_in,
                                        channels=cur_channels,
                                        kernel_sizes=residuals_kernels,
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

            layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

            iterator = iterator + self.pool_every
            cur_in = cur_channels[-1]
            cur_channels = self.channels[iterator:iterator + self.pool_every]

        if iterator < len(self.channels):
            cur_channels = self.channels[iterator:]
            residuals_kernels = len(cur_channels) * [3]

            layers.append(ResidualBlock(in_channels=cur_in,
                                        channels=cur_channels,
                                        kernel_sizes=residuals_kernels,
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

        seq = nn.Sequential(*layers)
        return seq
    # ========================
