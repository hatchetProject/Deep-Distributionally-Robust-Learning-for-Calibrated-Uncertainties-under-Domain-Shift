"""
Do test on our implementation of convolutional layer, compared with PyTorch implementation
Our one is not adopted with DRL formulation
"""
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import timeit
import torchvision.transforms as transforms
import util
import torch.nn.functional as F
from conv_layer import drl_Conv2d, drl_Conv2d_one_by_one

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def test_common():
    input = torch.randn(1, 3, 16, 16, requires_grad=True)
    tor_impl = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=5, stride=1)
    output_1 = tor_impl(input)
    s1 = torch.sum(output_1)

    drl_impl = drl_Conv2d(in_channels=3, out_channels=2, kernel_size=5, stride=1)
    with torch.no_grad():
        drl_impl.weight.data = tor_impl.weight.clone()
        drl_impl.bias.data = tor_impl.bias.clone()
    drl_label = torch.randn(1, 2, 7, 7)
    output_2 = drl_impl(input, drl_label, torch.ones((drl_label.shape[0], drl_label.shape[1], drl_label.shape[2])))
    s2 = torch.sum(output_2)

    output_1.register_hook(save_grad('output_1'))

    s1.backward()
    s2.backward()
    # print(output_1.requires_grad)
    # print("s1 grad:", s1.grad)
    # print("output1 grad:", grads["output_1"])

    print("tor_impl weight grad:", tor_impl.weight.grad)
    """
    input = np.ones((1, 1, 5, 5))
    input = torch.FloatTensor(input)
    input.requires_grad = True
    tor_impl = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
    print(tor_impl.weight.shape)
    output_1 = tor_impl(input)
    output_1.register_hook(save_grad('output_1'))
    input.register_hook(save_grad("input_1"))
    s1 = torch.sum(output_1)

    input = np.ones((1, 1, 5, 5))
    input = torch.FloatTensor(input)
    input.requires_grad = True
    drl_impl = drl_Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
    with torch.no_grad():
        drl_impl.weight.data = tor_impl.weight.clone()
        drl_impl.bias.data = tor_impl.bias.clone()
    drl_label = torch.randn(4, 2, 7, 7) # should be changed afterwards
    output_2 = drl_impl(input, drl_label, torch.ones((drl_label.shape[0], drl_label.shape[1], drl_label.shape[2])))
    output_2.register_hook(save_grad('output_2'))
    input.register_hook(save_grad("input_2"))
    s2 = torch.sum(output_2)
    s1.backward()
    s2.backward()
    print("tor_impl dout:", grads["output_1"])
    print("tor_impl weight grad:", tor_impl.weight.grad)
    print("tor_impl bias grad:", tor_impl.bias.grad)
    print("tor_impl input grad:", grads["input_1"])


    print("drl_impl dout:", grads["output_2"])
    print("drl_impl weight grad:", drl_impl.weight.grad)
    print("drl_impl bias grad:", drl_impl.bias.grad)
    print("drl_impl input grad:", grads["input_2"])
    """

class drl_model(nn.Module):
    def __init__(self):
        super(drl_model, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1)
        self.activate1 = nn.ReLU()
        #self.layer2 = drl_Conv2d_one_by_one(in_channels=3, out_channels=2, kernel_size=1, stride=1)
        self.layer2 = drl_Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=1)

    def forward(self, x, Y, r):
        x = self.layer1(x)
        x = self.activate1(x)
        x = self.layer2(x, Y, r)
        return x

def test_one_by_one():
    #input = np.ones((1, 3, 5, 5))
    #input = torch.FloatTensor(input)
    input = torch.randn((2, 3, 5, 5))
    input.requires_grad = True
    input_drl = input.clone()
    #print(input, input_drl)
    tor_impl = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2))
    output_1 = tor_impl(input)
    output_tmp = tor_impl[0](input)
    output_1.register_hook(save_grad('output_1'))
    output_tmp.register_hook(save_grad('mid_layer'))
    input.register_hook(save_grad("input_1"))
    s1 = torch.sum(output_1)

    #input = np.ones((1, 3, 5, 5))
    #input = torch.FloatTensor(input)
    #input.requires_grad = True
    drl_impl = drl_model()
    with torch.no_grad():
        drl_impl.layer1.weight.data = tor_impl[0].weight.clone()
        drl_impl.layer1.bias.data = tor_impl[0].bias.clone()
        drl_impl.layer2.weight.data = tor_impl[2].weight.clone()
        drl_impl.layer2.bias.data = tor_impl[2].bias.clone()
    drl_label = torch.randn(4, 2, 7, 7)  # should be changed afterwards
    output_2 = drl_impl(input_drl, drl_label, torch.ones((drl_label.shape[0], drl_label.shape[1], drl_label.shape[2])))
    output_tmp2 = drl_impl.layer1(input_drl)
    output_2.register_hook(save_grad('output_2'))
    output_tmp2.register_hook(save_grad('mid_layer_drl'))
    input_drl.register_hook(save_grad("input_2"))
    s2 = torch.sum(output_2)
    s1.backward()
    s2.backward()
    #print("input value:", input)
    #print("tor_impl dout:", grads["output_1"])
    #print("tor_impl weight:", tor_impl.weight)
    #print("tor_impl output:", output_1)
    #print("tor_impl weight grad:", tor_impl[0].weight.grad)
    #print("tor_impl bias grad:", tor_impl[0].bias.grad)
    #print("tor_impl input grad:", grads["input_1"])
    #print("tor_impl mid layer grad:", grads["mid_layer"])

    #print("drl_impl dout:", grads["output_2"])
    #print("drl_impl output:", output_2)
    #print("drl_impl weight grad:", drl_impl.layer1.weight.grad)
    #print("drl_impl bias grad:", drl_impl.layer1.bias.grad)
    #print("drl_impl input grad:", grads["input_2"])
    #print("drl_impl mid layer grad:", grads["mid_layer_drl"])

if __name__=="__main__":
    test_one_by_one()
