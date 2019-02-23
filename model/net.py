"""Defines the neural network, losss function and metrics"""

import torch.nn as nn
from torchvision import models
from utils.localization_utils import *


class ModelWrapper:
    """
    This class will return pre-trained models with modified last FC Layer for Localization
    """

    def __init__(self):
        self.model_ft = None
        self.number_output = 4

    def set_parameter_requires_grad(self):
        """
        Turns off gradient calculation in all layers as we are only learning the last FC layer.
        :return:
        """
        for param in self.model_ft.parameters():
            param.requires_grad = False

    def get_resnet18_network(self):
        """
        returns pre-trained resnet18 model with modified last layer.
        Resnet16's last FC layer has out_feature 1000 (same as image net)
        (fc): Linear(in_features=512, out_features=1000, bias=True)
        we need to reshape to out_features 4 and make the weights learnable.
        :return:
        """
        self.model_ft = models.resnet18(pretrained=True)
        # self.set_parameter_requires_grad()
        num_features = self.model_ft.fc.in_features
        # Parameters of newly constructed modules have requires_grad=True by default so no need to set auto grad=True
        self.model_ft.fc = nn.Linear(num_features, self.number_output)
        return self.model_ft


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    loss = nn.SmoothL1Loss()
    return loss(outputs,
                labels)


def accuracy(outputs, targets, img_sizes):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (torch tensor) dimension batch_size x 4
        targets: (torch.tensor) dimension batch_size x 6
        img_sizes: (torch.tensor) dimension batch_size x 2

    Returns: (float) accuracy in [0,1]
    """
    accuracy_val = compute_acc(preds=outputs, targets=targets, im_sizes=img_sizes)  # in pytorch data structure

    return accuracy_val.detach().numpy()


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
}
