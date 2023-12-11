import numpy as np
import math
import torch

# from https://github.com/PAIR-code/saliency

# Output of the last convolution layer for the given input, including the batch
# dimension.
CONVOLUTION_LAYER_VALUES = 'CONVOLUTION_LAYER_VALUES'
# Gradients of the output being explained (the logit/softmax value) with respect
# to the last convolution layer, including the batch dimension.
CONVOLUTION_OUTPUT_GRADIENTS = 'CONVOLUTION_OUTPUT_GRADIENTS'
# Gradients of the output being explained (the logit/softmax value) with respect
# to the input. Shape should be the same shape as x_value_batch.
INPUT_OUTPUT_GRADIENTS = 'INPUT_OUTPUT_GRADIENTS'
# Value of the output being explained (the logit/softmax value).
OUTPUT_LAYER_VALUES = 'OUTPUT_LAYER_VALUES'

SHAPE_ERROR_MESSAGE = {
    CONVOLUTION_LAYER_VALUES: (
        'Expected outermost dimension of CONVOLUTION_LAYER_VALUES to be the '
        'same as x_value_batch - expected {}, actual {}'
    ),
    CONVOLUTION_OUTPUT_GRADIENTS: (
        'Expected outermost dimension of CONVOLUTION_OUTPUT_GRADIENTS to be the '
        'same as x_value_batch - expected {}, actual {}'
    ),
    INPUT_OUTPUT_GRADIENTS: (
        'Expected key INPUT_OUTPUT_GRADIENTS to be the same shape as input '
        'x_value_batch - expected {}, actual {}'
    ),
    OUTPUT_LAYER_VALUES: (
        'Expected outermost dimension of OUTPUT_LAYER_VALUES to be the same as'
        ' x_value_batch - expected {}, actual {}'
    ),
}


class CoreSaliency(object):
  r"""Base class for saliency methods. Alone, this class doesn't do anything."""

  def GetMask(self, x_value, call_model_function, call_model_args=None):
    """Returns an unsmoothed mask.
    Args:
      x_value: Input ndarray.
      call_model_function: A function that interfaces with a model to return
        specific output in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - Other arguments used to call and run the model.
          expected_keys - List of keys that are expected in the output. Possible
            keys in this list are CONVOLUTION_LAYER_VALUES, 
            CONVOLUTION_OUTPUT_GRADIENTS, INPUT_OUTPUT_GRADIENTS, and
            OUTPUT_LAYER_VALUES, and are explained in detail where declared.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
    """
    raise NotImplementedError('A derived class should implemented GetMask()')

  def GetSmoothedMask(self,
                      x_value,
                      call_model_function,
                      call_model_args=None,
                      stdev_spread=.15,
                      nsamples=25,
                      magnitude=True,
                      **kwargs):
    """Returns a mask that is smoothed with the SmoothGrad method.
    Args:
      x_value: Input ndarray.
      call_model_function: A function that interfaces with a model to return
        specific output in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - Other arguments used to call and run the model.
          expected_keys - List of keys that are expected in the output. Possible
            keys in this list are CONVOLUTION_LAYER_VALUES,
            CONVOLUTION_OUTPUT_GRADIENTS, INPUT_OUTPUT_GRADIENTS, and
            OUTPUT_LAYER_VALUES, and are explained in detail where declared.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min). Defaults to 15%.
      nsamples: Number of samples to average across to get the smooth gradient.
      magnitude: If true, computes the sum of squares of gradients instead of
                 just the sum. Defaults to true.
    """
    stdev = stdev_spread * (torch.max(x_value) - torch.min(x_value))

    total_gradients = torch.zeros_like(x_value)
    for _ in range(nsamples):
      noise = torch.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise
      grad = self.GetMask(x_plus_noise, call_model_function, call_model_args,
                          **kwargs)
      if magnitude:
        total_gradients += (grad * grad)
      else:
        total_gradients += grad

    return total_gradients / nsamples

  def format_and_check_call_model_output(self, output, input_shape, expected_keys):
    """Converts keys in the output into an np.ndarray, and confirms its shape.
    Args:
      output: The output dictionary of data to be formatted.
      input_shape: The shape of the input that yielded the output
      expected_keys: List of keys inside output to format/check for shape agreement.
    Raises:
        ValueError: If output shapes do not match expected shape."""
    # If key is in check_full_shape, the shape should be equal to the input shape (e.g. 
    # INPUT_OUTPUT_GRADIENTS, which gives gradients for each value of the input). Otherwise,
    # only checks the outermost dimension of output to match input_shape (i.e. the batch size
    # should be the same).
    check_full_shape = [INPUT_OUTPUT_GRADIENTS]
    for expected_key in expected_keys:
      output[expected_key] = output[expected_key]
      expected_shape = input_shape
      actual_shape = output[expected_key].shape
      if expected_key not in check_full_shape:
        expected_shape = expected_shape[0]
        actual_shape = actual_shape[0]
      if expected_shape != actual_shape:
        raise ValueError(SHAPE_ERROR_MESSAGE[expected_key].format(
                       expected_shape, actual_shape))

def VisualizeImageGrayscale(image_3d, percentile=99):
  r"""Returns a 3D tensor as a grayscale 2D tensor.
  This method sums a 3D tensor across the absolute value of axis=2, and then
  clips values at a given percentile.
  """

  image_2d = np.sum(np.abs(image_3d), axis=2)

  vmax = np.percentile(image_2d, percentile)
  vmin = np.min(image_2d)

  return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def VisualizeImageDiverging(image_3d, percentile=99):
  r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
  """
  image_2d = np.sum(image_3d, axis=2)

  span = abs(np.percentile(image_2d, percentile))
  vmin = -span
  vmax = span

  return np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)

# A very small number for comparing floating point values.
EPSILON = 1E-9

def l1_distance(x1, x2):
  """Returns L1 distance between two points."""
  return torch.abs(x1 - x2).sum()

def translate_x_to_alpha(x, x_input, x_baseline):
  """Translates a point on straight-line path to its corresponding alpha value.
  Args:
    x: the point on the straight-line path.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.
  Returns:
    The alpha value in range [0, 1] that shows the relative location of the
    point between x_baseline and x_input.
  """
  return torch.where(x_input - x_baseline != 0,
                  (x - x_baseline) / (x_input - x_baseline), torch.nan)

def translate_alpha_to_x(alpha, x_input, x_baseline):
  """Translates alpha to the point coordinates within straight-line interval.
   Args:
    alpha: the relative location of the point between x_baseline and x_input.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.
  Returns:
    The coordinates of the point within [x_baseline, x_input] interval
    that correspond to the given value of alpha.
  """
  assert 0 <= alpha <= 1.0
  return x_baseline + (x_input - x_baseline) * alpha

def guided_ig_impl(x_input, model, device, x_baseline, grad_func, steps=200, fraction=0.25,
    max_dist=0.02):
  """Calculates and returns Guided IG attribution.
  Args:
    x_input: model input that should be explained.
    x_baseline: chosen baseline for the input explanation.
    grad_func: gradient function that accepts a model input and returns
      the corresponding output gradients. In case of many class model, it is
      responsibility of the implementer of the function to return gradients
      for the specific class of interest.
    steps: the number of Riemann sum steps for path integral approximation.
    fraction: the fraction of features [0, 1] that should be selected and
      changed at every approximation step. E.g., value `0.25` means that 25% of
      the input features with smallest gradients are selected and changed at
      every step.
    max_dist: the relative maximum L1 distance [0, 1] that any feature can
      deviate from the straight line path. Value `0` allows no deviation and,
      therefore, corresponds to the Integrated Gradients method that is
      calculated on the straight-line path. Value `1` corresponds to the
      unbounded Guided IG method, where the path can go through any point within
      the baseline-input hyper-rectangular.
  """

  x_baseline = torch.asarray(x_baseline)
  x = x_baseline.clone()
  l1_total = l1_distance(x_input, x_baseline)
  attr = torch.zeros_like(x_input)

  # If the input is equal to the baseline then the attribution is zero.
  total_diff = x_input - x_baseline
  if torch.abs(total_diff).sum() == 0:
    return attr

  # Iterate through every step.
  for step in range(steps):
    # Calculate gradients and make a copy.
    grad_actual = grad_func(x, model, device)

    grad = grad_actual.clone()
    # Calculate current step alpha and the ranges of allowed values for this
    # step.
    alpha = (step + 1.0) / steps
    alpha_min = max(alpha - max_dist, 0.0)
    alpha_max = min(alpha + max_dist, 1.0)
    x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
    x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
    # The goal of every step is to reduce L1 distance to the input.
    # `l1_target` is the desired L1 distance after completion of this step.
    l1_target = l1_total * (1 - (step + 1) / steps)

    # Iterate until the desired L1 distance has been reached.
    gamma = float('inf')
    while gamma > 1.0:
      x_old = x.clone()
      x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
      x_alpha[torch.isnan(x_alpha)] = alpha_max
      # All features that fell behind the [alpha_min, alpha_max] interval in
      # terms of alpha, should be assigned the x_min values.
      x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

      # Calculate current L1 distance from the input.
      l1_current = l1_distance(x, x_input)
      # If the current L1 distance is close enough to the desired one then
      # update the attribution and proceed to the next step.
      if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
        attr += (x - x_old) * grad_actual
        break

      # Features that reached `x_max` should not be included in the selection.
      # Assign very high gradients to them so they are excluded.
      grad[x == x_max] = float('inf')

      # Select features with the lowest absolute gradient.
      threshold = torch.quantile(torch.abs(grad), fraction, interpolation='lower')
      s = torch.logical_and(torch.abs(grad) <= threshold, grad != float('inf'))

      # Find by how much the L1 distance can be reduced by changing only the
      # selected features.
      l1_s = (np.abs(x - x_max) * s).sum()

      # Calculate ratio `gamma` that show how much the selected features should
      # be changed toward `x_max` to close the gap between current L1 and target
      # L1.
      if l1_s > 0:
        gamma = (l1_current - l1_target) / l1_s
      else:
        gamma = float('inf')

      if gamma > 1.0:
        # Gamma higher than 1.0 means that changing selected features is not
        # enough to close the gap. Therefore change them as much as possible to
        # stay in the valid range.
        x[s] = x_max[s]
      else:
        assert gamma > 0, gamma
        x[s] = translate_alpha_to_x(gamma, x_max, x)[s]

      # Update attribution to reflect changes in `x`.
      attr += (x - x_old) * grad_actual
      
  return attr

def call_model_function(images, model, device, call_model_args = None, expected_keys = None):
  target_class_idx = call_model_args['class_idx_str']
  images = images.requires_grad_(True)
  output = model(images.to(device))

  m = torch.nn.Softmax(dim=1)
  output = m(output)

  if INPUT_OUTPUT_GRADIENTS in expected_keys:
    outputs = output[:,target_class_idx]
    grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))[0]
    gradients = grads.detach()
    images = images.requires_grad_(False)
    
    return {INPUT_OUTPUT_GRADIENTS: gradients}

class GuidedIG(CoreSaliency):
  """Implements ML framework independent version of Guided IG."""

  expected_keys = [INPUT_OUTPUT_GRADIENTS]

  def GetMask(self, x_value, model, device, call_model_function, call_model_args=None,
      x_baseline=None, x_steps=200, fraction=0.25, max_dist=0.02):

    """Computes and returns the Guided IG attribution.
    Args:
      x_value: an input (ndarray) for which the attribution should be computed.
      call_model_function: A function that interfaces with a model to return
        specific data in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - user defined arguments. The value of this argument
            is the value of `call_model_args` argument of the nesting method.
          expected_keys - List of keys that are expected in the output. For this
            method (Guided IG), the expected keys are
            INPUT_OUTPUT_GRADIENTS - Gradients of the output being
              explained (the logit/softmax value) with respect to the input.
              Shape should be the same shape as x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      fraction: the fraction of features [0, 1] that should be selected and
        changed at every approximation step. E.g., value `0.25` means that 25%
        of the input features with smallest gradients are selected and changed
        at every step.
      max_dist: the relative maximum L1 distance [0, 1] that any feature can
        deviate from the straight line path. Value `0` allows no deviation and;
        therefore, corresponds to the Integrated Gradients method that is
        calculated on the straight-line path. Value `1` corresponds to the
        unbounded Guided IG method, where the path can go through any point
        within the baseline-input hyper-rectangular.
    """

    if x_baseline is None:
      x_baseline = torch.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape

    return guided_ig_impl(
        x_input=x_value,
        model = model,
        device = device,
        x_baseline=x_baseline,
        grad_func=self._get_grad_func(call_model_function, call_model_args),
        steps=x_steps,
        fraction=fraction,
        max_dist=max_dist)

  def _get_grad_func(self, call_model_function, call_model_args):
    def _grad_func(x_value, model, device):
      call_model_output = call_model_function(
          x_value,
          model,
          device,
          call_model_args=call_model_args,
          expected_keys=self.expected_keys)
      return call_model_output[INPUT_OUTPUT_GRADIENTS]

    return _grad_func

class IntegratedGradients(CoreSaliency):
  """A CoreSaliency class that implements the integrated gradients method.
  https://arxiv.org/abs/1703.01365
  """

  expected_keys = [INPUT_OUTPUT_GRADIENTS]

  def GetMask(self, x_value, call_model_function, call_model_args=None,
              x_baseline=None, x_steps=25, batch_size=1):
    """Returns an integrated gradients mask.
    Args:
      x_value: Input ndarray.
      call_model_function: A function that interfaces with a model to return
        specific data in a dictionary when given an input and other arguments.
        Expected function signature:
        - call_model_function(x_value_batch,
                              call_model_args=None,
                              expected_keys=None):
          x_value_batch - Input for the model, given as a batch (i.e. dimension
            0 is the batch dimension, dimensions 1 through n represent a single
            input).
          call_model_args - Other arguments used to call and run the model.
          expected_keys - List of keys that are expected in the output. For this
            method (Integrated Gradients), the expected keys are
            INPUT_OUTPUT_GRADIENTS - Gradients of the output being
              explained (the logit/softmax value) with respect to the input.
              Shape should be the same shape as x_value_batch.
      call_model_args: The arguments that will be passed to the call model
        function, for every call of the model.
      x_baseline: Baseline value used in integration. Defaults to 0.
      x_steps: Number of integrated steps between baseline and x.
      batch_size: Maximum number of x inputs (steps along the integration path)
        that are passed to call_model_function as a batch.
    """

    if x_baseline is None:
      x_baseline = torch.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape

    x_diff = x_value - x_baseline

    total_gradients = torch.zeros_like(x_value)

    x_step_batched = []
    for alpha in torch.linspace(0, 1, x_steps):
      x_step = x_baseline + (alpha * x_diff)
      x_step_batched.append(x_step)

      if len(x_step_batched) == batch_size or alpha == 1:
        grads = call_model_function(
            x_step_batched,
            call_model_args=call_model_args,
            expected_keys=self.expected_keys)

        self.format_and_check_call_model_output(grads,
                                                x_step_batched.shape,
                                                self.expected_keys)

        total_gradients += grads[INPUT_OUTPUT_GRADIENTS].sum(axis=0)
          
        x_step_batched = []

    return total_gradients * x_diff / x_steps