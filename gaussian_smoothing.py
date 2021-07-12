import math
import torch
from torch import nn
from torch.nn import functional as F


class GaussianTargetSmoothing(nn.Module):
    """
    Apply gaussian smoothing on the target images.
    Smoothing is performed separately for each pixel in the input using a depthwise convolution.
    Arguments:
        total_pixels (int): Number of pixel of the input image (HWC).
        sigma (float): Standard deviation of the gaussian kernel.
        kernel_size (int): Size of the gaussian kernel, or number of "classes", default=256.
    """

    def __init__(self, total_pixels, sigma, kernel_size=256):
        super(GaussianTargetSmoothing, self).__init__()
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        x = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1) / 2
        kernel *= 1 / (sigma * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((x - mean) / sigma) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, -1)
        kernel = kernel.repeat(total_pixels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = total_pixels

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv1d(input, weight=self.weight, groups=self.groups, padding='same')
