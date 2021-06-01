"""
This file includes the basic layers of the models for training ImageNet
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ClassificationFunctionAVH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, Y, r_st, bias=True):
        # Normalize the output of the classifier before the last layer using the criterion of AVH
        # code here
        exp_temp = input.mm(weight.t()).mul(r_st)
        #s = 1  # a hyperparameter for adjusting the scale of the output logits
        #exp_temp = s*avh_batch_score(input, weight.t()).mul(r_st)
        # forward output for confidence regularized training KL version, r is some ratio instead of density ratio
        #r = 0.001 # another hyperparameter that need to be tuned
        #new_exp_temp = (exp_temp + r*Y)/(r*Y + torch.ones(Y.shape).cuda())
        #exp_temp = new_exp_temp
        #orig_pred = torch.argmax(exp_temp, dim=1)
        #new_pred = torch.argmax(new_exp_temp, dim=1)
        #for idx in range(orig_pred.shape[0]):
        #    exp_temp[idx] = torch.where(orig_pred[idx] == new_pred[idx], new_exp_temp[idx], exp_temp[idx])

        # does bias matter? check this
        if bias is not None:
            exp_temp += bias.unsqueeze(0).expand_as(exp_temp)
        output = F.softmax(exp_temp, dim=1)
        ctx.save_for_backward(input, weight, bias, output, Y)
        return exp_temp

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output, Y = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_Y = grad_r = None
        if ctx.needs_input_grad[0]:
            # not negative here, which is different from math derivation
            grad_input = (output - Y).mm(weight)/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[1]:
            grad_weight = ((output.t() - Y.t()).mm(input))/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[2]:
            grad_Y = None
        if ctx.needs_input_grad[3]:
            grad_r = None
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_Y, grad_r, grad_bias

class ClassifierLayerAVH(nn.Module):
    """
    The last layer for C
    """
    def __init__(self, input_features, output_features, bias=True):
        super(ClassifierLayerAVH, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter("bias", None)

        # Weight initialization
        self.weight.data.uniform_(-1./math.sqrt(input_features), 1./math.sqrt(input_features))
        if bias is not None:
            self.bias.data.uniform_(-1./math.sqrt(input_features), 1./math.sqrt(input_features))

    def forward(self, input, Y, r):
        return ClassificationFunctionAVH.apply(input, self.weight, Y, r, self.bias)

    def extra_repr(self):
        return "in_features={}, output_features={}, bias={}".format(
            self.input_features, self.output_features, self.bias is not None
        )

class RatioEstimationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_output, prediction, pass_sign):
        """
        input: The density ratio
        nn_output: The output (a new feature) of the classification network, with shape (batch_size, n_classes)
        prediction: The probability, with shape (batch_size, n_classes)
        """
        ctx.save_for_backward(input, nn_output, prediction, pass_sign)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, nn_output, prediction, pass_sign = ctx.saved_tensors
        grad_input = grad_out = grad_pred = grad_pass = None
        if ctx.needs_input_grad[0]:
            if pass_sign is None:
                grad_input = grad_output.clone()
                #print grad_input
            else:
                grad_input = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1, 1)
                #print grad_input
        if ctx.needs_input_grad[1]:
            grad_out = None
        if ctx.needs_input_grad[2]:
            grad_pred = None
        if ctx.needs_input_grad[3]:
            grad_pass = None
        return grad_input, grad_out, grad_pred, grad_pass

class RatioEstimationLayer(nn.Module):
    """
    The last layer for D
    """
    def __init__(self):
        super(RatioEstimationLayer, self).__init__()

    def forward(self, input, nn_output, prediction, pass_sign):
        return RatioEstimationFunction.apply(input, nn_output, prediction, pass_sign)

    def extra_repr(self):
        return "Ratio Estimation Layer"

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SourceGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_output, prediction, p_t, sign_variable):
        ctx.save_for_backward(input, nn_output, prediction, p_t, sign_variable)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, nn_output, prediction, p_t, sign_variable = ctx.saved_tensors
        #print "Source back called", sign_variable
        grad_input = grad_out = grad_pred = grad_p_t = grad_sign = None
        if ctx.needs_input_grad[0]:
            if sign_variable is None:
                grad_input = grad_output
            else:
                grad_input = 1e3*torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1)/p_t
        if ctx.needs_input_grad[1]:
            grad_out = None
        if ctx.needs_input_grad[2]:
            grad_pred = None
        if ctx.needs_input_grad[3]:
            grad_p_t = None
        return grad_input, grad_out, grad_pred, grad_p_t, grad_sign

class SourceGradLayer(nn.Module):
    def __init__(self):
        super(SourceGradLayer, self).__init__()

    def forward(self, input, nn_output, prediction, p_t, sign_variable):
        return SourceGradFunction.apply(input, nn_output, prediction, p_t, sign_variable)

    def extra_repr(self):
        return "The Layer After Source Density Estimation, but different from SourceProbLayer"

class GradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_output, prediction, p_t, sign_variable):
        ctx.save_for_backward(input, nn_output, prediction, p_t, sign_variable)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, nn_output, prediction, p_t, sign_variable = ctx.saved_tensors
        grad_input = grad_out = grad_pred = grad_p_t = grad_sign = None
        if ctx.needs_input_grad[0]:
            # The parameters here controls the uncertainty measurement entropy of the results
            if sign_variable is None:
                grad_input = grad_output # original *1e2
            else:
                grad_source = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1)/p_t
                grad_target = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1) * (-(1-p_t)/p_t**2)
                grad_source /= prediction.shape[0]
                grad_target /= prediction.shape[0]
                grad_input = 1e-1 * torch.cat((grad_source, grad_target), dim=1)/p_t.shape[0]
            grad_input = 1e1 * grad_input # original 1e1
        if ctx.needs_input_grad[1]:
            grad_out = None
        if ctx.needs_input_grad[2]:
            grad_pred = None
        if ctx.needs_input_grad[3]:
            grad_p_t = None
        return grad_input, grad_out, grad_pred, grad_p_t, grad_sign

class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()

    def forward(self, input, nn_output, prediction, p_t, sign_variable):
        return GradFunction.apply(input, nn_output, prediction, p_t, sign_variable)

    def extra_repr(self):
        return "The Layer After Source Density Estimation"

