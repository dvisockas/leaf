import torch
import torch.nn as nn

class SquaredModulus(nn.Module):
  """Squared modulus layer.

  Returns a PyTorch module that implements a squared modulus operator.
  To implement the squared modulus of C complex-valued channels, the expected
  input dimension is N*1*W*(2*C) where channels role alternates between
  real and imaginary part.
  The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
  - squared operator on real and imag
  - average pooling to compute (real ** 2 + imag ** 2) / 2
  - multiply by 2

  Attributes:
    pool: average-pooling function over the channel dimensions
  """

  def __init__(self):
    super(SquaredModulus, self).__init__()
    self._pool = nn.AvgPool1d(2, stride=2)

  def forward(self, x):
    x = x.transpose(2, 1)
    output = 2 * self._pool(x**2)
    return output.transpose(2, 1)
