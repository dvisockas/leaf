"""Computes mel-filterbanks and corresponding gabor filters.

This script creates a class Gabor which takes as arguments parameters for a
mel-filterbank. A family of gabor wavelets is then created to match the
frequency response of the mel-filterbank.
"""

from leaf_torch import impulse_responses
import torch
import torch.nn.functional as F
import math

@gin.configurable
class Gabor:
  """This class creates gabor filters designed to match mel-filterbanks.

  Attributes:
    n_filters: number of filters
    min_freq: minimum frequency spanned by the filters
    max_freq: maximum frequency spanned by the filters
    sample_rate: samplerate (samples/s)
    window_len: window length in samples
    n_fft: number of frequency bins to compute mel-filters
    normalize_energy: boolean, True means that all filters have the same energy,
      False means that the higher the center frequency of a filter, the higher
      its energy
  """

  def __init__(self,
               n_filters: int = 40,
               min_freq: float = 0.,
               max_freq: float = 8000.,
               sample_rate: int = 16000,
               window_len: int = 401,
               n_fft: int = 512,
               normalize_energy: bool = False):

    self.n_filters = n_filters
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.sample_rate = sample_rate
    self.window_len = window_len
    self.n_fft = n_fft
    self.normalize_energy = normalize_energy

  @property
  def gabor_params_from_mels(self):
    """Retrieves center frequencies and standard deviations of gabor filters."""
    two = torch.tensor(2.)
    coeff = torch.sqrt(two * torch.log(two)) * self.n_fft
    sqrt_filters = torch.sqrt(self.mel_filters)
    center_frequencies = torch.argmax(sqrt_filters, dim=1).type(torch.float)
    peaks, _ = torch.max(sqrt_filters, dim=1, keepdim=True)
    half_magnitudes = peaks / 2.
    fwhms = torch.sum(
        (sqrt_filters >= half_magnitudes).type(torch.float), dim=1)
    return torch.stack(
        [center_frequencies * 2 * math.pi / self.n_fft, coeff / (math.pi * fwhms)],
        dim=1)

  def _mel_filters_areas(self, filters):
    """Area under each mel-filter."""
    peaks = torch.max(filters, dim=1, keepdim=True)
    return peaks * (
      torch.sum((filters > 0).type(torch.float), dim=1, keepdims=True) + 2
      ) * math.pi / self.n_fft

  @property
  def mel_filters(self):
    """Creates a bank of mel-filters."""
    # build mel filter matrix
    mel_filters = linear_to_mel_weight_matrix(
        num_mel_bins=self.n_filters,
        num_spectrogram_bins=self.n_fft // 2 + 1,
        sample_rate=self.sample_rate,
        lower_edge_hertz=self.min_freq,
        upper_edge_hertz=self.max_freq)
    mel_filters = mel_filters.T
    if self.normalize_energy:
      mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
    return mel_filters

  @property
  def gabor_filters(self):
    """Generates gabor filters that match the corresponding mel-filters."""
    gabor_filters = impulse_responses.gabor_filters(
        self.gabor_params_from_mels, size=self.window_len)
    return gabor_filters * (
        torch.sqrt(self._mel_filters_areas(self.mel_filters) * 2 *
        torch.sqrt(torch.tensor(math.pi)) * self.gabor_params_from_mels[:, 1:2])
      ).type(torch.complex64)

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def mel_to_hertz(mel_values):
  """Converts frequencies in `mel_values` from the mel scale to linear scale."""
  return _MEL_BREAK_FREQUENCY_HERTZ * (
      torch.exp(torch.tensor(mel_values) / _MEL_HIGH_FREQUENCY_Q) - 1.0)

def hertz_to_mel(frequencies_hertz):
  """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
  return _MEL_HIGH_FREQUENCY_Q * torch.log(
      1.0 + (torch.tensor(frequencies_hertz) / _MEL_BREAK_FREQUENCY_HERTZ))

def linear_to_mel_weight_matrix(num_mel_bins=20,
                                num_spectrogram_bins=129,
                                sample_rate=16000,
                                lower_edge_hertz=125.0,
                                upper_edge_hertz=3800.0):
  """Returns a matrix to warp linear scale spectrograms to the mel scale.
  Adapted from tf.contrib.signal.linear_to_mel_weight_matrix with a minimum
  band width (in Hz scale) of 1.5 * freq_bin. To preserve accuracy,
  we compute the matrix at float64 precision and then cast to `dtype`
  at the end. This function can be constant folded by graph optimization
  since there are no Tensor inputs.
  Args:
    num_mel_bins: Int, number of output frequency dimensions.
    num_spectrogram_bins: Int, number of input frequency dimensions.
    sample_rate: Int, sample rate of the audio.
    lower_edge_hertz: Float, lowest frequency to consider.
    upper_edge_hertz: Float, highest frequency to consider.
  Returns:
    PyTorch float32 tensor of shape [num_spectrogram_bins, num_mel_bins].
  Raises:
    ValueError: Input argument in the wrong range.
  """
  # Validate input arguments
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if num_spectrogram_bins <= 0:
    raise ValueError(
        'num_spectrogram_bins must be positive. Got: %s' % num_spectrogram_bins)
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if lower_edge_hertz < 0.0:
    raise ValueError(
        'lower_edge_hertz must be non-negative. Got: %s' % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got: %s for sample_rate: %s'
                     % (upper_edge_hertz, sample_rate))

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = torch.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:].unsqueeze(1)
  # spectrogram_bins_mel = hertz_to_mel(linear_frequencies)

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  band_edges_mel = torch.linspace(
      hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz),
      num_mel_bins + 2)

  lower_edge_mel = band_edges_mel[0:-2]
  center_mel = band_edges_mel[1:-1]
  upper_edge_mel = band_edges_mel[2:]

  freq_res = nyquist_hertz / float(num_spectrogram_bins)
  freq_th = 1.5 * freq_res
  # TODO: Check if this is needed as lower, center and upper are rewritten afterwards
  for i in range(0, num_mel_bins):
    center_hz = mel_to_hertz(center_mel[i])
    lower_hz = mel_to_hertz(lower_edge_mel[i])
    upper_hz = mel_to_hertz(upper_edge_mel[i])
    if upper_hz - lower_hz < freq_th:
      rhs = 0.5 * freq_th / (center_hz + _MEL_BREAK_FREQUENCY_HERTZ)
      dm = _MEL_HIGH_FREQUENCY_Q * torch.log(
        rhs + torch.sqrt(torch.tensor(1.0 + rhs**2)))
      lower_edge_mel[i] = center_mel[i] - dm
      upper_edge_mel[i] = center_mel[i] + dm

  lower_edge_hz = mel_to_hertz(lower_edge_mel).unsqueeze(0)
  center_hz = mel_to_hertz(center_mel).unsqueeze(0)
  upper_edge_hz = mel_to_hertz(upper_edge_mel).unsqueeze(0)

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (linear_frequencies - lower_edge_hz) / (
      center_hz - lower_edge_hz)
  upper_slopes = (upper_edge_hz - linear_frequencies) / (
      upper_edge_hz - center_hz)

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = torch.clamp(torch.min(lower_slopes, upper_slopes), min=0.0)

  # Re-add the zeroed lower bins we sliced out above.
  # [freq, mel]
  # TODO: Check if this is the same as
  # np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]],
  #                           'constant')
  mel_weights_matrix = F.pad(mel_weights_matrix, (bands_to_zero, 0, 0, 0))
  return mel_weights_matrix
