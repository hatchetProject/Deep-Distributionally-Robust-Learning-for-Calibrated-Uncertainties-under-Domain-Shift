"""
This file includes the basic layers of the models
"""

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

def avh_batch_score(x, w):
    """
    Actually computes the AVC score for a batch of samples;
    AVH score is used to replace the prediction probability
    x of shape (batch_size, feature_dim), w of shape (feature_dim, n_classes)
    :return: avh score of a single sample, with type float
    """
    cos_A = (x.mm(w)/((torch.norm(x, 2)*(torch.norm(w, 2, dim=0))).reshape(1, -1))).cpu().numpy()
    avh_pred = np.pi - np.arccos(cos_A)
    avh_pred = torch.FloatTensor(avh_pred)
    avh_pred = avh_pred.cuda()
    return avh_pred

class ClassificationFunctionAVH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, Y, r_st, bias=True):
        # Normalize the output of the classifier before the last layer using the criterion of AVH
        # code here
        exp_temp = input.mm(weight.t()).mul(r_st)
        #print("Forward: ",exp_temp[0])
        #print("weight: ", weight[0])
        #print("r_st: ", r_st[0])
        # forward output for confidence regularized training KL version, r is some ratio instead of density ratio
        r = 0.1 # another hyperparameter that need to be tuned
        new_exp_temp = (exp_temp + r*Y)/(r*Y + torch.ones(Y.shape).cuda())
        exp_temp = new_exp_temp

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
        #print("-----------------------")
        #print(output[0], Y[0])
        #print(Y[0])
        #print(weight[0])
        if ctx.needs_input_grad[0]:
            # not negative here, which is different from math derivation
            grad_input = -(output - Y).mm(weight)/(torch.norm(weight, 2)*torch.norm(input, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[1]:
            grad_weight = -((output.t() - Y.t()).mm(input))/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[2]:
            grad_Y = None
        if ctx.needs_input_grad[3]:
            grad_r = None
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0).squeeze(0)
        #print("Classification input: ", grad_input.reshape(-1)[:10])
        #print("Classification weight: ", grad_weight.reshape(-1)[:10])
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

class GradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nn_output, prediction, p_t, sign_variable):
        #print("beta input in: ", input[0][0])
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
                #print("!!!", nn_output.reshape(-1, )[:10])
                #print("???", prediction.reshape(-1, )[:10])
                #print("###", p_t.reshape(-1, )[:10])
                grad_source = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1)/p_t
                grad_target = torch.sum(nn_output.mul(prediction), dim=1).reshape(-1,1) * (-(1-p_t)/p_t**2)
                grad_source /= prediction.shape[0]
                grad_target /= prediction.shape[0]
                grad_input = 1e1 * torch.cat((grad_source, grad_target), dim=1)/p_t.shape[0]
                #print("grad 2: ", grad_input.reshape(-1, )[:6])
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
