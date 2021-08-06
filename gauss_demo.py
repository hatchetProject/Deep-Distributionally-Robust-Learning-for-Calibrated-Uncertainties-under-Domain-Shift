"""
Gaussian demo demonstrating that accurate density ratio estimation (plugin version)
is not as well as end2end version
"""
import sklearn
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from sklearn.neighbors import KernelDensity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClassificationFunctionAVH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, Y, r_st, bias=True):
        # Normalize the output of the classifier before the last layer using the criterion of AVH
        # code here
        exp_temp = input.mm(weight.t()).mul(r_st)
        # forward output for confidence regularized training KL version, r is some ratio instead of density ratio
        #r = 0.001 # another hyperparameter that need to be tuned
        #new_exp_temp = (exp_temp + r*Y)/(r*Y + torch.ones(Y.shape).cuda())
        #exp_temp = new_exp_temp
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
            grad_input = (output - Y).mm(weight)#/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
        if ctx.needs_input_grad[1]:
            grad_weight = ((output.t() - Y.t()).mm(input))#/(torch.norm(input, 2)*torch.norm(weight, 2))#/(output.shape[0]*output.shape[1])
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

class GaussData(Dataset):
    def __init__(self, input, label, r):
        self.input = input
        self.label = label
        self.r = r

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx]
        label = self.label[idx]
        r = self.r[idx]
        return input, label, r

class gauss_alpha(nn.Module):
    def __init__(self):
        super(gauss_alpha, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.clf = ClassifierLayerAVH(16, 2, bias=True)

    def forward(self, x, y, r):
        x = self.model(x)
        return self.clf(x, y, r)

class gauss_beta(nn.Module):
    def __init__(self):
        super(gauss_beta, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p

def train_accurate(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t):
    BATCH_SIZE=8
    epoch=80
    model_alpha = gauss_alpha()
    model_alpha = model_alpha.to(DEVICE)
    #optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-2, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_alpha = torch.optim.Adam(model_alpha.parameters(), lr=1e-3)
    trainset = GaussData(x_s, y_s, ground_r_s)
    testset = GaussData(x_t, y_t, ground_r_t)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    ce_loss = nn.CrossEntropyLoss()
    train_loss, train_acc, test_acc, test_loss = 0, 0, 0, 0
    test_num = 0
    for j in range(epoch):
        for i, (input, label, r) in enumerate(train_loader):
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            r = r.to(DEVICE)
            r = r.reshape((-1, 1))
            theta_out = model_alpha(input, label, r.float())
            source_pred = F.softmax(theta_out, dim=1)
            loss_theta = ce_loss(theta_out, torch.argmax(label, dim=1))

            optimizer_alpha.zero_grad()
            loss_theta.backward()
            optimizer_alpha.step()

            train_loss += loss_theta
            train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == torch.argmax(label, dim=1)).float() / input.shape[0]
        train_acc /= (i+1)
        #print("Train loss: {}, acc: {}".format(train_loss, train_acc*100.0))
        train_loss, train_acc = 0, 0

        with torch.no_grad():
            for i, (input, label, r) in enumerate(test_loader):
                input = input.to(DEVICE)
                label = label.to(DEVICE)
                test_num += input.shape[0]
                r = r.to(DEVICE)
                r = r.reshape((-1, 1))
                target_out = model_alpha(input, torch.ones((input.shape[0], label.shape[1])).cuda(), r.float())
                prediction_t = F.softmax(target_out, dim=1)
                test_loss += ce_loss(target_out, torch.argmax(label, dim=1))
                test_acc += torch.sum(
                    torch.argmax(prediction_t.detach(), dim=1) == torch.argmax(label, dim=1)).float() / input.shape[0]
        #if j % 10 == 0:
        #    print("Test acc: {}".format(test_acc*100.0/(i+1)))
        test_acc = 0
    test_loss /= test_num

    return model_alpha, test_loss.detach().cpu().numpy()

def train_end2end(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t):
    BATCH_SIZE = 8
    epoch = 100
    model_alpha = gauss_alpha()
    model_beta = gauss_beta()
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    #optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=5e-4)
    #optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_alpha = torch.optim.Adam(model_alpha.parameters(), lr=1e-3)
    optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=1e-3)
    trainset = GaussData(x_s, y_s, ground_r_s)
    testset = GaussData(x_t, y_t, ground_r_t)
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    train_loss, train_acc, test_acc, test_loss = 0, 0, 0, 0
    for j in range(epoch):
        iter_train = iter(train_loader)
        for i, (input_test, _, _) in enumerate(test_loader):
            input_train, label_train, _ = iter_train.next()
            input_train = input_train.to(DEVICE)
            label_train = label_train.to(DEVICE)
            input_test = input_test.to(DEVICE)
            input_concat = torch.cat([input_train, input_test], dim=0)
            label_concat = torch.cat(
                (torch.FloatTensor([1, 0]).repeat(input_train.shape[0], 1),
                 torch.FloatTensor([0, 1]).repeat(input_test.shape[0], 1)), dim=0)
            label_concat = label_concat.to(DEVICE)
            prob = model_beta(input_concat, None, None, None, None)
            assert (F.softmax(prob.detach(), dim=1).cpu().numpy().all() >= 0 and F.softmax(prob.detach(), dim=1).cpu().numpy().all() <= 1)
            loss_dis = bce_loss(prob, label_concat)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            # Separate source sample density ratios from target sample density ratios
            r_source = r[:BATCH_SIZE].reshape(-1, 1)
            r_target = r[BATCH_SIZE:].reshape(-1, 1)
            p_t_source = p_t[:BATCH_SIZE]
            p_t_target = p_t[BATCH_SIZE:]

            theta_out = model_alpha(input_train, label_train, r_source)
            source_pred = F.softmax(theta_out, dim=1)
            nn_out = model_alpha(input_test, torch.ones((input_test.shape[0], 2)).cuda(),
                                 r_target.detach().cuda())
            pred_target = F.softmax(nn_out, dim=1)
            prob_grad_r = model_beta(input_test, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                     sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape).cuda()))
            loss_theta = ce_loss(theta_out, torch.argmax(label_train, dim=1))


            optimizer_beta.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_beta.step()

            optimizer_beta.zero_grad()
            loss_r.backward(retain_graph=True)
            optimizer_beta.step()

            optimizer_alpha.zero_grad()
            loss_theta.backward()
            optimizer_alpha.step()

            train_loss += loss_theta
            train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == torch.argmax(label_train, dim=1)).float()/input_train.shape[0]
        train_acc /= (i + 1)
        #print("Train loss: {}, acc: {}".format(train_loss, train_acc*100.0))
        train_loss, train_acc = 0, 0
        with torch.no_grad():
            for i, (input, label, _) in enumerate(test_loader):
                input = input.to(DEVICE)
                label = input.to(DEVICE)
                prob = model_beta(input, None, None, None, None)
                r = prob[:, 0] / prob[:, 1]
                r = r.reshape((-1, 1))
                target_out = model_alpha(input, torch.ones((input.shape[0], label.shape[1])).cuda(), r)
                test_loss += ce_loss(target_out, torch.argmax(label, dim=1))
                prediction_t = F.softmax(target_out, dim=1)
                test_acc += torch.sum(
                    torch.argmax(prediction_t.detach(), dim=1) == torch.argmax(label, dim=1)).float() / input.shape[0]

            print("Test acc: {}, test loss: {}".format(test_acc*100.0 / (i + 1), test_loss / (i+1)))
            test_acc, test_loss = 0, 0
    return model_alpha, model_beta

def plot_fig_accurate(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t, path="tmp.png"):
    x_1 = np.load("data/gaussian3/x_s.npy")
    x_2 = np.load("data/gaussian3/x_t.npy")
    y_1 = np.load("data/gaussian3/y_s.npy")
    y_2 = np.load("data/gaussian3/y_t.npy")
    mu_s = np.array([7, 7])
    var_s = np.array([[3, -2], [-2, 3]])
    mu_t = np.array([7, 7])
    var_t = np.array([[3, 2], [2, 3]])

    maxs = 12.5
    mins = 2
    x_dim1, x_dim2 = np.meshgrid(np.arange(mins, maxs+0.02, 0.02), np.arange(mins, maxs+0.02, 0.02))
    dim = int((maxs - mins) / 0.02 + 1)
    x_pos = x_1[np.nonzero(y_1[:, 0] == 0)[0], :]
    x_neg = x_1[np.nonzero(y_1[:, 0] == 1)[0], :]
    x_pos_t = x_2[np.nonzero(y_2[:, 0] == 0)[0], :]
    x_neg_t = x_2[np.nonzero(y_2[:, 0] == 1)[0], :]

    idx = np.where(x_pos[:, 0] < 11)[0]
    x_pos = x_pos[idx]
    idx = np.where(x_pos[:, 1] < 12)[0]
    x_pos = x_pos[idx]
    idx = np.where(x_neg[:, 1] > 3)[0]
    x_neg = x_neg[idx]
    idx = np.where(x_neg[:, 0] > 3.5)[0]
    x_neg = x_neg[idx]
    idx = np.where(x_pos_t[:, 0] < 10.5)[0]
    x_pos_t = x_pos_t[idx]
    idx = np.where(x_pos_t[:, 1] < 11)[0]
    x_pos_t = x_pos_t[idx]
    idx = np.where(x_neg_t[:, 0] > 4)[0]
    x_neg_t = x_neg_t[idx]

    model, test_loss = train_accurate(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t)
    print("Test loss: {}".format(test_loss))
    prediction = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            x_t = np.array([[x_dim1[i, j], x_dim2[i, j]]])
            #x_t = np.concatenate((np.ones((x_t.shape[0], 1)), x_t), axis=1)
            r = multivariate_normal.pdf(x_t, mu_s, var_s) / multivariate_normal.pdf(x_t, mu_t, var_t)
            r = r.reshape((1, 1))
            r = torch.FloatTensor(r)
            output = model(torch.tensor(x_t).float().cuda(), torch.ones((1, 2)).cuda(), r.float().cuda())
            output = F.softmax(output, dim=1)
            output = output.detach().cpu().numpy()
            prediction[i, j] = output[0][1]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5))
    plt = fig.add_subplot(111)
    sc = plt.pcolor(x_dim1, x_dim2, prediction, cmap="seismic")
    sc.set_clim(0, 1)
    plt.scatter(x_pos_t[:, 0], x_pos_t[:, 1], marker="^", label="Target 1", c="orange", s=24)
    plt.scatter(x_neg_t[:, 0], x_neg_t[:, 1], marker="v", label="Target 0", c="darkgreen", s=24)
    #position = fig.add_axes([0, 0.11, 0.03, 0.77]) # left, down, right, up
    #fig.colorbar(sc, cax=position, orientation="vertical")
    ell1 = Ellipse(xy=(mu_s[0], mu_s[1]), width=9, height=4, angle=-360 * (math.atan(1.5) / (2 * math.pi)), fill=False,
                   linewidth=3)
    ell2 = Ellipse(xy=(mu_t[0], mu_t[1]), width=9, height=4, angle=360 * (math.atan(1.5) / (2 * math.pi)), fill=False,
                   linewidth=3.2, linestyle="dotted")
    plt.add_patch(ell1)
    plt.add_patch(ell2)
    plt.axis("off")
    plt.set_title("Log-likelihood = -935", fontdict={"weight":"normal", "size": 24})
    fig.savefig(path)
    return test_loss

def plot_fig_end2end(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t, path="tmp.png"):
    x_1 = np.load("data/gaussian1/x_s.npy")
    x_2 = np.load("data/gaussian1/x_t.npy")
    y_1 = np.load("data/gaussian1/y_s.npy")
    y_2 = np.load("data/gaussian1/y_t.npy")
    mu_s = np.array([6, 6])
    var_s = np.array([[3, -2], [-2, 3]])
    mu_t = np.array([7, 7])
    var_t = np.array([[3, 2], [2, 3]])

    maxs = 15
    mins = -5
    x_dim1, x_dim2 = np.meshgrid(np.arange(mins, maxs+0.1, 0.1), np.arange(mins, maxs+0.1, 0.1))
    dim = int((maxs - mins) / 0.1 + 1)
    x_pos = x_1[np.nonzero(y_1[:, 0] == 0)[0], :]
    x_neg = x_1[np.nonzero(y_1[:, 0] == 1)[0], :]
    x_pos_t = x_2[np.nonzero(y_2[:, 0] == 0)[0], :]
    x_neg_t = x_2[np.nonzero(y_2[:, 0] == 1)[0], :]

    #x_1 = torch.FloatTensor(np.concatenate((np.ones((x_1.shape[0], 1)), x_1), axis=1))
    #x_2 = torch.FloatTensor(np.concatenate((np.ones((x_2.shape[0], 1)), x_2), axis=1))
    #y_1 = torch.FloatTensor(y_1)
    #y_2 = torch.FloatTensor(y_2)
    #if not trained:
    #    _ , _, _, _, _ = trainNew(x_1, y_1, x_2, y_2)
    model_alpha, model_beta = train_end2end(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t)
    prediction = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            x_t = np.array([[x_dim1[i, j], x_dim2[i, j]]])
            #x_t = np.concatenate((np.ones((x_t.shape[0], 1)), x_t), axis=1)
            pred = model_beta(torch.tensor(x_t).float().cuda(), None, None, None, None)
            r = pred[:, 0] / pred[:, 1]
            output = model_alpha(torch.tensor(x_t).float().cuda(), torch.ones((1, 2)).cuda(), r.cuda())
            output = F.softmax(output, dim=1)
            output = output.detach().cpu().numpy()
            prediction[i, j] = output[0][1]

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    sc = plt.pcolor(x_dim1, x_dim2, prediction, cmap="RdBu")
    sc.set_clim(0, 1)
    ax1.scatter(x_pos[:, 0], x_pos[:, 1], marker="+")
    ax1.scatter(x_neg[:, 0], x_neg[:, 1], marker="o")
    fig.colorbar(sc)
    ell1 = Ellipse(xy=(mu_s[0], mu_s[1]), width=9, height=4, angle=-360*(math.atan(1.5)/(2*math.pi)), fill=False)
    ax1.add_patch(ell1)
    ell2 = Ellipse(xy=(mu_t[0], mu_t[1]), width=9, height=4, angle=360*(math.atan(1.5)/(2*math.pi)), fill=False)
    ax1.add_patch(ell2)

    ax2 = fig.add_subplot(122)
    sc = plt.pcolor(x_dim1, x_dim2, prediction, cmap="RdBu")
    sc.set_clim(0, 1)
    ax2.scatter(x_pos_t[:, 0], x_pos_t[:, 1], marker="+")
    ax2.scatter(x_neg_t[:, 0], x_neg_t[:, 1], marker="o")
    fig.colorbar(sc)
    ell1 = Ellipse(xy=(mu_s[0], mu_s[1]), width=9, height=4, angle=-360*(math.atan(1.5)/(2*math.pi)), fill=False)
    ell2 = Ellipse(xy=(mu_t[0], mu_t[1]), width=9, height=4, angle=360*(math.atan(1.5)/(2*math.pi)), fill=False)
    ax2.add_patch(ell1)
    ax2.add_patch(ell2)
    plt.savefig(path)

def kde_estimation(x_s, x_t, band=1.0):
    kde = KernelDensity(bandwidth=band, kernel="gaussian")
    kde = kde.fit(np.concatenate([x_s, x_t]))
    liklihood = (kde.score(x_t) + kde.score(x_s)) / 2
    src_weight = np.exp(kde.score_samples(x_s))
    tgt_weight = np.exp(kde.score_samples(x_t))
    return liklihood, src_weight*1e2, tgt_weight*1e2

def plot_fig3_a():
    x_s = np.load("data/gaussian3/x_s.npy")
    x_t = np.load("data/gaussian3/x_t.npy")
    y_s = np.load("data/gaussian3/y_s.npy")
    y_t = np.load("data/gaussian3/y_t.npy")
    mu_s = np.array([7, 7])
    var_s = np.array([[3, -2], [-2, 3]])
    mu_t = np.array([7, 7])
    var_t = np.array([[3, 2], [2, 3]])

    x_pos = x_s[np.nonzero(y_s[:, 0] == 0)[0], :]
    x_neg = x_s[np.nonzero(y_s[:, 0] == 1)[0], :]
    x_pos_t = x_t[np.nonzero(y_t[:, 0] == 0)[0], :]
    x_neg_t = x_t[np.nonzero(y_t[:, 0] == 1)[0], :]

    idx = np.where(x_pos[:, 0]<11)[0]
    x_pos = x_pos[idx]
    idx = np.where(x_pos[:, 1] < 12)[0]
    x_pos = x_pos[idx]
    idx = np.where(x_neg[:, 1] > 3)[0]
    x_neg = x_neg[idx]
    idx = np.where(x_neg[:, 0] > 3.5)[0]
    x_neg = x_neg[idx]
    idx = np.where(x_pos_t[:, 0] < 10.5)[0]
    x_pos_t = x_pos_t[idx]
    idx = np.where(x_pos_t[:, 1] < 11)[0]
    x_pos_t = x_pos_t[idx]
    idx = np.where(x_neg_t[:, 0] > 4)[0]
    x_neg_t = x_neg_t[idx]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5))
    plt = fig.add_subplot(111)
    plt.scatter(x_pos[:, 0], x_pos[:, 1], marker="o", label="Source 1", c="r", s=24)
    plt.scatter(x_neg[:, 0], x_neg[:, 1], marker="*", label="Source 0", c="b", s=24)

    plt.scatter(x_pos_t[:, 0], x_pos_t[:, 1], marker="^", label="Target 1", c="orange", s=24)
    plt.scatter(x_neg_t[:, 0], x_neg_t[:, 1], marker="v", label="Target 0", c="darkgreen", s=24)
    ell1 = Ellipse(xy=(mu_s[0], mu_s[1]), width=9, height=4, angle=-360 * (math.atan(1.5) / (2 * math.pi)), fill=False, linewidth=3)
    ell2 = Ellipse(xy=(mu_t[0], mu_t[1]), width=9, height=4, angle=360 * (math.atan(1.5) / (2 * math.pi)), fill=False, linewidth=3.2, linestyle="dotted")
    plt.add_patch(ell1)
    plt.add_patch(ell2)
    plt.set_xlim((2, 12.5))
    plt.set_ylim((2, 12.5))
    fig.legend(loc=(0.755, 0.76), fontsize=12)
    plt.axis("off")
    fig.savefig("log/finals/fig3_a.jpg", bbox_inches='tight')


def plot_logloss(likelihood, mean_loss, std_loss):
    fig = plt.figure(figsize=(18, 12))
    x_axis = np.rint(likelihood)
    #logloss = pow(logloss, 1/2)
    x_new = np.log(x_axis + 2500)
    x_new[1] = x_new[1] - 0.12
    x_new[2] = x_new[2] - 0.06
    plt.plot(x_new, mean_loss, linewidth="6.5", marker='o', markersize=16, c="darkblue")
    plt.fill_between(x_new, mean_loss - std_loss, mean_loss + std_loss, facecolor="b", alpha=0.25)
    x_axis = [str(x)[:-2] for x in x_axis]
    plt.xticks(x_new, x_axis, rotation=0)
    plt.tick_params(labelsize=30)
    plt.xlabel("KDE log-likelihood", fontdict={"weight": "normal", "size": 36})
    plt.ylabel("Sqrt(Logloss)", fontdict={"weight": "normal", "size": 36})
    directory = "log/finals/"
    import os
    if not os.path.exists(directory):
        os.mkdir(directory)
    plt.savefig(directory + "fig3_d.jpg", bbox_inches="tight")


if __name__=="__main__":
    x_s = np.load("data/gaussian3/x_s.npy")
    x_t = np.load("data/gaussian3/x_t.npy")
    y_s = np.load("data/gaussian3/y_s.npy")
    y_t = np.load("data/gaussian3/y_t.npy")

    x_s, x_t, y_s, y_t = torch.FloatTensor(x_s), torch.FloatTensor(x_t), torch.FloatTensor(y_s), torch.FloatTensor(y_t)
    print (x_s.shape, y_s.shape, x_t.shape, y_t.shape)
    ground_r_s = multivariate_normal.pdf(x_s, mu_s, var_s) / multivariate_normal.pdf(x_s, mu_t, var_t)
    ground_r_t = multivariate_normal.pdf(x_t, mu_s, var_s) / multivariate_normal.pdf(x_t, mu_t, var_t)

    train_end2end(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t)
    train_accurate(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t)
    plot_fig_accurate(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t, "log/gauss_plot/accurate_plot.jpg")
    plot_fig_end2end(x_s, y_s, x_t, y_t, ground_r_s, ground_r_t, "log/gauss_plot/end2end_plot1.jpg")
    
    bandwidths = [0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005]
    test_loss = []
    likelihood_list = []
    for i in range(len(bandwidths)):
        likelihood, src_weight, tgt_weight = kde_estimation(x_s, x_t, band=bandwidths[i])
        print("Bandwidth {} has likelihood {}".format(bandwidths[i], likelihood))
        x_s, x_t, y_s, y_t = torch.FloatTensor(x_s), torch.FloatTensor(x_t), torch.FloatTensor(y_s), torch.FloatTensor(y_t)
        src_weight, tgt_weight = torch.tensor(src_weight), torch.tensor(tgt_weight)
        loss = plot_fig_accurate(x_s, y_s, x_t, y_t, src_weight, tgt_weight, "log/gauss_plot/band_"+str(bandwidths[i])+".jpg")
        likelihood_list.append(likelihood)
        test_loss.append(loss)
    print("Bandwidth: ", bandwidths)
    print("Likelihood: ", likelihood_list)
    print("Test loss: ", test_loss)
    
    
    plot_fig3_a()

    """
    # plot data, the likelihood and losses are put here for faster visualization
    # this part also does the KDE estimation
    bandwidths = np.array([0.2, 0.05, 0.04, 0.03, 0.02, 0.01, 0.006, 0.004, 0.002, 0.001])
    likelihood = np.array(
        [-934.9857878579751, -482.7166757106748, -381.8943244564515, -246.66452098330416, -50.24848524856729,
         291.49231954766816, 545.6785097285135, 748.057816382672, 1094.2679419167853,
         1440.75641079062])
    loss1 = np.array([0.014756339602172375, 0.01483641, 0.01881391, 0.01591324, 0.01637949, 0.02924049, 0.05493251, 0.1471393, 0.52957857, 1.9488393])
    loss2 = np.array([0.01867685094475746, 0.01596146, 0.01798576, 0.02261054, 0.01778168, 0.02881819, 0.04272154, 0.14143793, 0.45200148, 2.6351626])
    loss3 = np.array([0.018818695098161697, 0.01725437, 0.0111, 0.01575122, 0.01857065, 0.01766134, 0.04312471, 0.16487613, 0.6151211, 3.0907857])
    loss1 = np.sqrt(loss1)
    loss2 = np.sqrt(loss2)
    loss3 = np.sqrt(loss3)
    mean_loss = (loss1 + loss2 + loss3) / 3
    std_loss = np.zeros(loss1.shape)
    for i in range(loss1.shape[0]):
        std_loss[i] = np.std([loss1[i], loss2[i], loss3[i]], ddof=1)
    bandwidth = 0.2
    likelihood, src_weight, tgt_weight = kde_estimation(x_s, x_t, band=bandwidth)
    print("Bandwidth {} has likelihood {}".format(bandwidth, likelihood))
    x_s, x_t, y_s, y_t = torch.FloatTensor(x_s), torch.FloatTensor(x_t), torch.FloatTensor(y_s), torch.FloatTensor(y_t)
    src_weight, tgt_weight = torch.tensor(src_weight), torch.tensor(tgt_weight)
    loss = plot_fig_accurate(x_s, y_s, x_t, y_t, src_weight, tgt_weight, "log/finals/fig3_b.jpg")
    """

