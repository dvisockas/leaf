import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from leaf_torch.impulse_responses import gabor_filters

class GaborConstraint:
  """Constraint mu and sigma, in radians.
  Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
  gaussian response is in [1,pi/2]. The full-width at half maximum of the
  Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
  https://arxiv.org/pdf/1711.01161.pdf for more details.
  """

  def __init__(self, kernel_size):
    """Initialize kernel size.
    Args:
      kernel_size: the length of the filter, in samples.
    """
    self._kernel_size = kernel_size

  def __call__(self, kernel):
    mu_lower = 0.
    mu_upper = math.pi
    sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
    sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi

    clipped_mu = torch.clamp(kernel[:, 0], mu_lower, mu_upper)
    clipped_sigma = torch.clamp(kernel[:, 1], sigma_lower, sigma_upper)
    return torch.stack([clipped_mu, clipped_sigma], axis=1)

class GaborConv1D(nn.Module):
  """Implements a convolution with filters defined as complex Gabor wavelets.
  These filters are parametrized only by their center frequency and
  the full-width at half maximum of their frequency response.
  Thus, for n filters, there are 2*n parameters to learn.
  """

  def __init__(self, filters, kernel_size, strides, padding, use_bias,
               kernel_initializer, kernel_regularizer,
               trainable, sort_filters=False):
    super(GaborConv1D, self).__init__()
    self._filters = filters // 2
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._use_bias = use_bias
    self._sort_filters = sort_filters

    # TODO: Add regularization
    initialized_kernel = kernel_initializer(torch.zeros(self._filters, 2))
    self._kernel = nn.Parameter(initialized_kernel)
    self._kernel.constraint = GaborConstraint(self._kernel_size)

    if self._use_bias:
      self._bias = nn.Parameter(torch.zeros(self._filters * 2), requires_grad=trainable)

  def forward(self, inputs):
    kernel = self._kernel.constraint(self._kernel)
    if self._sort_filters:
      filter_order = torch.argsort(kernel[:, 0])
      kernel = torch.gather(kernel, filter_order, dim=0, index=filter_order)

    filters = gabor_filters(kernel, self._kernel_size)

    stacked_filters = torch.stack([filters.real, filters.imag], dim=1)
    stacked_filters = stacked_filters.reshape(
        [2 * self._filters, self._kernel_size])

    stacked_filters = stacked_filters.transpose(1, 0).unsqueeze(1)

    outputs = F.conv1d(
        inputs, stacked_filters, stride=self._strides, padding=self._padding)

    if self._use_bias: outputs += self._bias

    return outputs
