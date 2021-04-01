import math
import torch

def gabor_impulse_response(t: torch.Tensor, center: torch.Tensor,
                           fwhm: torch.Tensor) -> torch.Tensor:
  """Computes the gabor impulse response."""
  denominator = 1.0 / (torch.sqrt(2.0 * torch.tensor(math.pi)) * fwhm)
  gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm**2), -t**2, dims=0))
  center_frequency_complex = center.type(torch.complex64)
  t_complex = t.type(torch.complex64)
  sinusoid = torch.exp(
      1j * torch.tensordot(center_frequency_complex, t_complex, dims=0))
  denominator = denominator.type(torch.complex64).unsqueeze(1)
  gaussian = gaussian.type(torch.complex64)
  return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401) -> torch.Tensor:
  """Computes the gabor filters from its parameters for a given size.
  Args:
    kernel: torch.Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.
  Returns:
    A torch.Tensor<float>[filters, size].
  """
  return gabor_impulse_response(
      torch.arange(-(size // 2), (size + 1) // 2, dtype=torch.float32),
      center=kernel[:, 0], fwhm=kernel[:, 1])

def gaussian_lowpass(sigma: torch.Tensor, filter_size: int):
  """Generates gaussian windows centered in zero, of std sigma.

  Args:
    sigma: torch.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.

  Returns:
    A torch.Tensor<float>[1, filter_size, C, 1].
  """
  sigma = torch.clamp(sigma, 2. / filter_size, 0.5)
  t = torch.arange(0, filter_size).reshape(1, filter_size, 1, 1)
  numerator = t - 0.5 * (filter_size - 1)
  denominator = sigma * 0.5 * (filter_size - 1)
  return torch.exp(-0.5 * (numerator / denominator)**2)

