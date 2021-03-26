import torch
import torch.nn as nn

class Leaf(nn.Module):
  """PyTorch module that implements time-domain filterbanks.

  Creates a LEAF frontend, a learnable front-end that takes an audio
  waveform as input and outputs a learnable spectral representation. This layer
  can be initialized to replicate the computation of standard mel-filterbanks.
  A detailed technical description is presented in Section 3 of
  https://arxiv.org/abs/2101.08596 .

  """

  def __init__(
      self,
      learn_pooling: bool = True,
      learn_filters: bool = True,
      conv1d_cls=convolution.GaborConv1D,
      activation=SquaredModulus(),
      pooling_cls=pooling.GaussianLowpass,
      n_filters: int = 40,
      sample_rate: int = 16000,
      window_len: float = 25.,
      window_stride: float = 10.,
      compression_fn: _TensorCallable = postprocessing.PCENLayer(
          alpha=0.96,
          smooth_coef=0.04,
          delta=2.0,
          floor=1e-12,
          trainable=True,
          learn_smooth_coef=True,
          per_channel_smooth_coef=True),
      preemp: bool = False,
      preemp_init: _Initializer = initializers.PreempInit(),
      complex_conv_init: _Initializer = initializers.GaborInit(
          sample_rate=16000, min_freq=60.0, max_freq=7800.0),
      pooling_init: _Initializer = tf.keras.initializers.Constant(0.4),
      regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
      mean_var_norm: bool = False,
      spec_augment: bool = False,
      name='leaf'):
    super().__init__(name=name)
    window_size = int(sample_rate * window_len // 1000 + 1)
    window_stride = int(sample_rate * window_stride // 1000)
    if preemp:
      self._preemp_conv = tf.keras.layers.Conv1D(
          filters=1,
          kernel_size=2,
          strides=1,
          padding='SAME',
          use_bias=False,
          input_shape=(None, None, 1),
          kernel_initializer=preemp_init,
          kernel_regularizer=regularizer_fn if learn_filters else None,
          name='tfbanks_preemp',
          trainable=learn_filters)

    self._complex_conv = conv1d_cls(
        filters=2 * n_filters,
        kernel_size=window_size,
        strides=1,
        padding='SAME',
        use_bias=False,
        input_shape=(None, None, 1),
        kernel_initializer=complex_conv_init,
        kernel_regularizer=regularizer_fn if learn_filters else None,
        name='tfbanks_complex_conv',
        trainable=learn_filters)

    self._activation = activation
    self._pooling = pooling_cls(
        kernel_size=window_size,
        strides=window_stride,
        padding='SAME',
        use_bias=False,
        kernel_initializer=pooling_init,
        kernel_regularizer=regularizer_fn if learn_pooling else None,
        trainable=learn_pooling)

    self._instance_norm = None
    if mean_var_norm:
      self._instance_norm = tfa.layers.InstanceNormalization(
          axis=2,
          epsilon=1e-6,
          center=True,
          scale=True,
          beta_initializer='zeros',
          gamma_initializer='ones',
          name='tfbanks_instancenorm')

    self._compress_fn = compression_fn if compression_fn else tf.identity
    self._spec_augment_fn = postprocessing.SpecAugment(
    ) if spec_augment else tf.identity

    self._preemp = preemp

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Computes the Leaf representation of a batch of waveforms.

    Args:
      inputs: input audio of shape (batch_size, num_samples) or (batch_size,
        num_samples, 1).
      training: training mode, controls whether SpecAugment is applied or not.

    Returns:
      Leaf features of shape (batch_size, time_frames, freq_bins).
    """
    # Inputs should be [B, W] or [B, W, C]
    outputs = inputs[:, :, tf.newaxis] if inputs.shape.ndims < 3 else inputs
    if self._preemp:
      outputs = self._preemp_conv(outputs)
    outputs = self._complex_conv(outputs)
    outputs = self._activation(outputs)
    outputs = self._pooling(outputs)
    outputs = tf.maximum(outputs, 1e-5)
    outputs = self._compress_fn(outputs)
    if self._instance_norm is not None:
      outputs = self._instance_norm(outputs)
    if training:
      outputs = self._spec_augment_fn(outputs)
    return outputs