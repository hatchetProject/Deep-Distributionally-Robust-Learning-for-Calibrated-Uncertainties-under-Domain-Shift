"""
Do the accuracy-confidence plot
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np
from model_layers import ClassifierLayer, RatioEstimationLayer, Flatten, GradLayer, IWLayer
import math
import warnings
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

## Model with aligned data
class Discriminator(nn.Module):
    def __init__(self, n_features):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )
        self.grad_r = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.net(x)
        p = self.grad_r(p, nn_output, prediction, p_t, pass_sign)
        return p

class thetaNet_align(nn.Module):
    def __init__(self, n_features, n_output):
        super(thetaNet_align, self).__init__()
        self.extractor = torch.nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            #nn.Linear(512, 256),
            #nn.Tanh(),
        )
        self.classifier = ClassifierLayer(512, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x_s = self.extractor(x_s)
        x = self.classifier(x_s, y_s, r)
        return x

class thetaNet(nn.Module):
    """
    Defines C network
    """
    def __init__(self, n_features, n_output):
        super(thetaNet, self).__init__()
        self.extractor = torch.nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.Tanh(),
            #nn.Linear(1024, 512),
            #nn.Tanh(),
        )
        self.classifier = ClassifierLayer(1024, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x_s = self.extractor(x_s)
        x = self.classifier(x_s, y_s, r)
        return x

## Model with image data
class Discriminator_e2e(nn.Module):
    def __init__(self):
        super(Discriminator_e2e, self).__init__()
        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                       self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )
        self.grad_r = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        x = self.sharedNet(x)
        x = x.view(-1, 2048)
        p = self.net(x)
        p = self.grad_r(p, nn_output, prediction, p_t, pass_sign)
        return p

class thetaNet_e2e(nn.Module):
    def __init__(self, n_output):
        super(thetaNet_e2e, self).__init__()
        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                       self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.extractor = torch.nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Tanh(),
            #nn.Linear(1024, 512),
            #nn.Tanh(),
            #nn.Linear(512, 256),
            #nn.Tanh(),
        )
        self.classifier = ClassifierLayer(1024, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x_s = self.sharedNet(x_s)
        x_s = x_s.view(-1, 2048)
        x_s = self.extractor(x_s)
        x = self.classifier(x_s, y_s, r)
        return x

## Importance weighting models
class Discriminator_IW(nn.Module):
    def __init__(self):
        super(Discriminator_IW, self).__init__()
        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                       self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            # nn.Linear(512, 512),
            # nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            # nn.Linear(64, 64),
            # nn.Tanh(),
            # nn.Linear(16, 8),
            # nn.Sigmoid(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.sharedNet(x)
        x = x.view(-1, 2048)
        p = self.net(x)
        return p

class IWNet(nn.Module):
    def __init__(self, n_output):
        super(IWNet, self).__init__()
        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                       self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.extractor = torch.nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            # nn.Linear(512, 512),
            # nn.Tanh(),
            # nn.Linear(512, 512),
            # nn.Tanh(),
            # nn.Linear(512, 512),
            # nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            # torch.nn.Linear(1024, 64),
        )
        self.IW = IWLayer(256, n_output)

    def forward(self, x_s, y_s, r):
        x_s = self.sharedNet(x_s)
        x_s = x_s.view(-1, 2048)
        x_s = self.extractor(x_s)
        x = self.IW(x_s, y_s, r)
        return x

## IID
class source_net(nn.Module):
    def __init__(self, num_classes):
        super(source_net, self).__init__()
        self.isTrain = True

        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.cls_fc = nn.Linear(2048, num_classes)
        self.cls_fc.weight.data.normal_(0, 0.005)

    def forward(self, source):
        source = self.sharedNet(source)
        source = source.view(source.size(0), 2048)
        clf = self.cls_fc(source)
        return clf

## DeepCORAL
def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE).mm(source)
    cs = (source.t().mm(source) - (tmp_s.t().mm(tmp_s)) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE).mm(target)
    ct = (target.t().mm(target) - (tmp_t.t().mm(tmp_t)) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

class DeepCoral(nn.Module):
    def __init__(self, num_classes, backbone):
        super(DeepCoral, self).__init__()
        self.isTrain = True
        self.backbone = backbone
        if self.backbone == 'resnet50':
            model_resnet = torchvision.models.resnet50(pretrained=True)
            #self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.layer2 = model_resnet.layer2
            self.layer3 = model_resnet.layer3
            self.layer4 = model_resnet.layer4
            self.avgpool = model_resnet.avgpool

            self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
            #self.compress = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(2048, num_classes)
            self.fc = nn.Linear(1,1)
        elif self.backbone == 'alexnet':
            model_alexnet = torchvision.models.alexnet(pretrained=True)
            self.sharedNet = model_alexnet.features
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )
            self.cls_fc = nn.Linear(4096, num_classes)
        elif self.backbone == "None":
            self.sharedNet = nn.Linear(256, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        self.cls_fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        coral_loss = 0
        source = self.sharedNet(source)
        source = source.view(source.size(0), 2048)
        #source = self.compress(source)
        if self.backbone == 'alexnet':
            source = self.fc(source)
        if self.isTrain:
            target = self.sharedNet(target)
            target = target.view(target.size(0), 2048)
            target = self.compress(target)
            if self.backbone == 'alexnet':
                target = self.fc(target)

            coral_loss = CORAL(source, target)

        clf = self.cls_fc(source)
        return clf, coral_loss

## Relaxed DeepCORAL
def Relaxed_CORAL(source, target, beta=0.5):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE).mm(source)
    cs = (source.t().mm(source) - (tmp_s.t().mm(tmp_s)) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE).mm(target)
    ct = (target.t().mm(target) - (tmp_t.t().mm(tmp_t)) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - (1+beta)*ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

class Relaxed_DeepCORAL(nn.Module):
    def __init__(self, num_classes):
        super(Relaxed_DeepCORAL, self).__init__()
        self.isTrain = True
        model_resnet = torchvision.models.resnet50(pretrained=True)
        #self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #n_features = model_resnet.fc.in_features
        self.compress = nn.Linear(2048, 256)
        self.cls_fc = nn.Linear(256, num_classes)
        self.fc = nn.Linear(1,1)
        self.cls_fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target, beta):
        coral_loss = 0
        source = self.sharedNet(source)
        source = source.view(source.size(0), 2048)
        source = self.compress(source)
        if self.isTrain:
            target = self.sharedNet(target)
            target = target.view(target.size(0), 2048)
            target = self.compress(target)
            coral_loss = Relaxed_CORAL(source, target, beta=beta)

        clf = self.cls_fc(source)
        return clf, coral_loss


## Bayesian network model
import pyro
from pyro.distributions import Normal, Categorical

class NN(nn.Module):
    def __init__(self, num_classes):
        super(NN, self).__init__()
        self.isTrain = True
        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                       self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.cls_fc = nn.Linear(2048, num_classes)
        self.cls_fc.weight.data.normal_(0, 0.005)

    def forward(self, source):
        source = self.sharedNet(source)
        source = source.view(-1, 2048)
        clf = self.cls_fc(source)
        return clf

def guide(net):
    softplus = torch.nn.Softplus()
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.cls_fc.weight)
    outw_sigma = torch.randn_like(net.cls_fc.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.cls_fc.bias)
    outb_sigma = torch.randn_like(net.cls_fc.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'out.weight': outw_prior, 'out.bias': outb_prior}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()

num_samples = 10
def predict(x, net):
    sampled_models = [guide(net) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean

## Temperature scaling model (imported from temperature_scaling.py)
## Used directly in function

def dataloader(root_path, dir, batch_size, train):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor()])
        }
    data = datasets.ImageFolder(root=root_path + dir, transform=transform['train' if train else 'test'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
    return data_loader

def entropy(p):
    p[p<1e-20] = 1e-20 # Deal with numerical issues
    return -torch.sum(p.mul(torch.log2(p)))

CONFIG = {
    "n_classes": 65,
    "batch_size": 32,
}

torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def brier_score(source, target):
    # You need to run the training process first, we directly load the saved model
    print('Using device:', DEVICE)
    torch.manual_seed(200)
    N_CLASSES = CONFIG["n_classes"]
    BATCH_SIZE = CONFIG["batch_size"]
    print("Source distribution: %s; Target distribution: %s" % (source, target))
    task = source[0]+target[0]

    # Aligned data with alternative training
    theta = thetaNet(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_aligned_"+task+".pkl"))
    discriminator = torch.load("models/dis_rba_alter_aligned_"+task+".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
    y_t = y_t.reshape(-1, 1)
    enc = OneHotEncoder(categories="auto")
    y_t = enc.fit_transform(y_t.numpy()).toarray()
    y_t = torch.FloatTensor(y_t)
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_num = 0
    b_score = 0
    with torch.no_grad():
        theta.eval()
        discriminator.eval()
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            for j in range(data.shape[0]):
                b_score += brier_score_loss(label[j].cpu().numpy(), prediction_t[j].cpu().numpy())
    print("Brier score for aligned data with alternative training: ", b_score/test_num)

    # Relaxed DCORAL aligned data with alternative training
    theta = thetaNet(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_aligned_relaxed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_alter_aligned_relaxed_" + task + ".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAlignedRelaxed.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetYRelaxed.pkl")
    y_t = y_t.reshape(-1, 1)
    enc = OneHotEncoder(categories="auto")
    y_t = enc.fit_transform(y_t.numpy()).toarray()
    y_t = torch.FloatTensor(y_t)
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_num = 0
    b_score = 0
    with torch.no_grad():
        theta.eval()
        discriminator.eval()
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            for j in range(data.shape[0]):
                b_score += brier_score_loss(label[j].cpu().numpy(), prediction_t[j].cpu().numpy())
    print("Brier score for relaxed aligned data with alternative training: ", b_score / test_num)

    
    # Aligned data with fixed R
    theta = thetaNet_align(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_fixed_aligned_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_fixed_aligned_" + task + ".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
    y_t = y_t.reshape(-1, 1)
    enc = OneHotEncoder(categories="auto")
    y_t = enc.fit_transform(y_t.numpy()).toarray()
    y_t = torch.FloatTensor(y_t)
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            for j in range(data.shape[0]):
                b_score += brier_score_loss(label[j].cpu().numpy(), prediction_t[j].cpu().numpy())
    print("Brier score for aligned data with fixed r: ", b_score / test_num)

    test_loader = dataloader("office/", target, 32, False)
    # IID
    theta = torch.load("models/sourceOnly_"+source+"_"+target+".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = theta(data).detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES-label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for IID: ", b_score / test_num)

    # DeepCORAL
    theta = torch.load("models/deepcoral_"+source+"_"+target+".pkl")
    theta.isTrain = False
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out, _ = theta(data, None)
            target_out = target_out.detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for DeepCORAL: ", b_score / test_num)

    # DeepCORAL with relaxed r
    theta = torch.load("models/relaxed_deepcoral_" + source + "_" + target + ".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out, _ = theta(data, None, None)
            target_out = target_out.detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for DeepCORAL with relaxed r: ", b_score / test_num)
    
    # Bayesian network
    theta = torch.load("models/bnn_"+task+".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = predict(data, theta)
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for BNN: ", b_score / test_num)
    
    # Temperature scaling
    from temperature_scaling import ModelWithTemperature, _ECELoss
    orig_model = torch.load("models/sourceOnly_" + source + "_" + target + ".pkl")
    valid_loader = dataloader("office/", source, 32, True)
    scaled_model = ModelWithTemperature(orig_model)
    scaled_model.set_temperature(valid_loader)
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = scaled_model(data)
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for temperature scaling: ", b_score / test_num)

    # IW + Fixed R
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_fixed_"+task+".pkl"))
    discriminator = torch.load("models/dis_iw_fixed_"+task+".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data).detach(), dim=1)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for IW with fixed r: ", b_score / test_num)

    # IW + Alternative Training
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_alter_" + task + ".pkl"))
    discriminator = torch.load("models/dis_iw_alter_" + task + ".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data).detach(), dim=1)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for IW with alternative training: ", b_score / test_num)


    # End2end with fixed r
    theta = thetaNet_e2e(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_fixed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_fixed_" + task + ".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for End2end with fixed r: ", b_score / test_num)

    # End2end with alternative r
    theta = thetaNet_e2e(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_alter_" + task + ".pkl")
    test_num = 0
    b_score = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
            for j in range(data.shape[0]):
                y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
    print("Brier score for End2end with alternative r: ", b_score / test_num)
    

if __name__=="__main__":
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    source = "RealWorld"
    for target in domains:
        if source != target:
            brier_score(source, target)

 
