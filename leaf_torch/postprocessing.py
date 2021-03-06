from typing import Union

import torch
import torch.nn as nn

class ExponentialMovingAverage(nn.Module):
  """Computes of an exponential moving average of an sequential input."""

  def __init__(
      self,
      coeff_init: Union[float, torch.Tensor],
      per_channel: bool = False, trainable: bool = False):
    """Initializes the ExponentialMovingAverage.

    Args:
      coeff_init: the value of the initial coeff.
      per_channel: whether the smoothing should be different per channel.
      trainable: whether the smoothing should be trained or not.
    """
    super(ExponentialMovingAverage, self).__init__()
    self._coeff_init = coeff_init
    self._per_channel = per_channel
    self._trainable = trainable

  def build(self, input_shape):
    num_channels = input_shape[-1]

    #TODO: add const initializer
    self._weights = nn.Parameter(
      num_channels if self._per_channel else 1,
      requires_grad = self._trainable
    )

  def call(self, inputs: torch.Tensor, initial_state: torch.Tensor):
    """Inputs is of shape [batch, seq_length, num_filters]."""
    w = self._weights.clip(min=0.0, max=1.0)
    result = tf.scan(lambda a, x: w * x + (1.0 - w) * a,
                     tf.transpose(inputs, (1, 0, 2)),
                     initializer=initial_state)
    return tf.transpose(result, (1, 0, 2))

class PCEN(nn.Module):
  """Per-Channel Energy Normalization.

  This applies a fixed or learnable normalization by an exponential moving
  average smoother, and a compression.
  See https://arxiv.org/abs/1607.05666 for more details.
  """

  def __init__(self,
               alpha: float = 0.96,
               smooth_coef: float = 0.04,
               delta: float = 2.0,
               root: float = 2.0,
               floor: float = 1e-6,
               trainable: bool = False,
               learn_smooth_coef: bool = False,
               per_channel_smooth_coef: bool = False):
    """PCEN constructor.

    Args:
      alpha: float, exponent of EMA smoother
      smooth_coef: float, smoothing coefficient of EMA
      delta: float, bias added before compression
      root: float, one over exponent applied for compression (r in the paper)
      floor: float, offset added to EMA smoother
      trainable: bool, False means fixed_pcen, True is trainable_pcen
      learn_smooth_coef: bool, True means we also learn the smoothing
        coefficient
      per_channel_smooth_coef: bool, True means each channel has its own smooth
        coefficient
      name: str, name of the layer
    """
    super(PCEN, self).__init__()
    self._alpha_init = alpha
    self._delta_init = delta
    self._root_init = root
    self._smooth_coef = smooth_coef
    self._floor = floor
    self._trainable = trainable
    self._learn_smooth_coef = learn_smooth_coef
    self._per_channel_smooth_coef = per_channel_smooth_coef

  def build(self, input_shape):
    # TODO: Move to 1
    num_channels = input_shape[-1]

    # Add proper initializers to alpha, delta and root
    self.alpha = nn.Parameter(num_channels, self._trainable)
    self.delta = nn.Parameter(num_channels, self._trainable)
    self.root = nn.Parameter(num_channels, self._trainable)

    if self._learn_smooth_coef:
      self.ema = ExponentialMovingAverage(
          coeff_init=self._smooth_coef,
          per_channel=self._per_channel_smooth_coef,
          trainable=True)
    else:
      # input_size ??? The number of expected features in the input x
      # hidden_size ??? The number of features in the hidden state h
      # num_layers ??? Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
      # nonlinearity ??? The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
      # bias ??? If False, then the layer does not use bias weights b_ih and b_hh. Default: True
      # batch_first ??? If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
      # dropout ??? If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
      # bidirectional ??? If True, becomes a bidirection

      self.ema = nn.RNN(
          units=num_channels,
          nonlinearity=None,
          use_bias=False,
          kernel_initializer=tf.keras.initializers.Identity(
              gain=self._smooth_coef),
          recurrent_initializer=tf.keras.initializers.Identity(
              gain=1. - self._smooth_coef),
          return_sequences=True,
          trainable=False)

      for parameter in self.ema.parameters:
        parameter.requires_grad = False

  def forward(self, inputs):
    one = torch.ones
    alpha = torch.min(self.alpha, torch.ones(1))
    root = torch.max(self.root, torch.ones(1))
    ema_smoother = self.ema(inputs, initial_state=tf.gather(inputs, 0, axis=1))
    one_over_root = 1. / root
    output = ((inputs / (self._floor + ema_smoother)**alpha + self.delta)
              **one_over_root - self.delta**one_over_root)
    return output

# %%
