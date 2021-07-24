import torch
from torch import nn


class ImageCrossEntropy(nn.Module):
    def forward(self, pred, target, temperature=1, im_average=True):
        """ Cross entropy that accepts soft targets
        Arguments:
             pred (torch.Tensor): predictions for neural network
             targets (torch.Tensor): targets, can be soft
             im_average: if false, sum is returned instead of mean
        """
        logsoftmax = nn.LogSoftmax(dim=2)

        pred = pred / temperature

        if im_average:
            return torch.mean(torch.sum(-target * logsoftmax(pred), dim=2))
        else:
            return torch.mean(torch.sum(-target * logsoftmax(pred), dim=(1, 2)))
