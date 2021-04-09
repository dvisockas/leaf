import torch
from torch import nn
import torch.nn.functional as F
from leaf_torch import impulse_responses

class GaussianLowpass(nn.Module):
  """Depthwise pooling (each input filter has its own pooling filter).

  Pooling filters are parametrized as zero-mean Gaussians, with learnable
  std. They can be initialized with 0.4 to approximate a Hanning window.
  """

  def __init__(
      self,
      kernel_size,
      strides=1,
      padding=0,
      use_bias=True,
      kernel_initializer=nn.init.xavier_uniform_,
      kernel_regularizer=None,
      trainable=False,
  ):

    super(GaussianLowpass, self).__init__()
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.trainable = trainable

    kernel = self.kernel_initializer(torch.zeros(1, 1, kernel_size, 1))
    self._kernel = nn.Parameter(kernel, requires_grad=self.trainable)

  def forward(self, inputs):
    kernel = impulse_responses.gaussian_lowpass(self._kernel, self.kernel_size)
    kernel = kernel.squeeze(0)
    # kernel = kernel.permute(2, 0, 1)
    # import pdb;pdb.set_trace()
    # TODO: Check for padding, will not work due to 'same' is 0 in torch
    outputs = F.conv1d(inputs, kernel, stride=self.strides, groups=self.kernel_size,
      padding=self.padding)
    return tf.squeeze(outputs, axis=1)
