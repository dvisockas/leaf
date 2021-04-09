
"""Initializer classes for each layer of the learnable frontend."""

import torch
from leaf_torch import melfilters, initializers

class ConstInit:
  def __init__(self, const: torch.float32):
    self.const = const

  def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
    return tensor.fill_(self.const)

def PreempInit(tensor: torch.Tensor, alpha: float=0.97) -> torch.Tensor:
    """Pytorch initializer for the pre-emphasis.

    Returns a Tensor to initialize the pre-emphasis layer of a Leaf instance.

    Attributes:
        alpha: parameter that controls how much high frequencies are emphasized by
        the following formula output[n] = input[n] - alpha*input[n-1] with
        0 < alpha < 1 (higher alpha boosts high frequencies)
    """

    shape = tensor.shape
    assert shape == (1,1,2), f"Cannot initialize preemp layer of size {shape}"

    with torch.no_grad():
        tensor[0, 0, 0] = -alpha
        tensor[0, 0, 1] = 1

        return tensor

class GaborInit:
  """Pytorch initializer for the complex-valued convolution.
  Returns a Tensor to initialize the complex-valued convolution layer of a
  Leaf instance with Gabor filters designed to match the
  frequency response of standard mel-filterbanks.
  If the shape has rank 2, this is a complex convolution with filters only
  parametrized by center frequency and FWHM, so we initialize accordingly.
  In this case, we define the window len as 401 (default value), as it is not
  used for initialization.
  """
  def __init__(self, **kwargs):
    kwargs.pop('n_filters', None)
    self._kwargs = kwargs

  def __call__(self, tensor, dtype=None, **kwargs):
    shape = tensor.shape

    n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
    window_len = 401 if len(shape) == 2 else shape[0]
    gabor_filters = melfilters.Gabor(
        n_filters=n_filters, window_len=window_len, **kwargs)
    if len(shape) == 2:
        with torch.no_grad():
            return gabor_filters.gabor_params_from_mels
    else:
      raise NotImplementedError
      even_indices = tf.range(shape[2], delta=2)
      odd_indices = tf.range(start=1, limit=shape[2], delta=2)
      filters = gabor_filters.gabor_filters
      filters_real_and_imag = tf.dynamic_stitch(
          [even_indices, odd_indices],
          [tf.math.real(filters), tf.math.imag(filters)])
      return tf.transpose(filters_real_and_imag[:, tf.newaxis, :], [2, 1, 0])




