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
from sklearn.preprocessing import OneHotEncoder
import heapq

warnings.filterwarnings('ignore')

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
            self.compress = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
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
        source = self.compress(source)
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

"""
## Bayesian network model
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, SGD
log_softmax = nn.LogSoftmax(dim=1)
softplus = torch.nn.Softplus()


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
net = NN(31)
net = net.cuda()

def model(x_data, y_data):
    outw_prior = Normal(loc=torch.zeros_like(net.cls_fc.weight), scale=torch.ones_like(net.cls_fc.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.cls_fc.bias), scale=torch.ones_like(net.cls_fc.bias))

    priors = {'out.weight': outw_prior, 'out.bias': outb_prior}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))

    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

def guide(x_data, y_data):
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
num_iterations = 30
train_loader = dataloader("office/", "webcam", 32, True)
optim = Adam({"lr": 1e-3})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean

for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        # calculate the loss and take a gradient step
        loss += svi.step(data[0].cuda(), data[1].cuda())
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = loss / normalizer_train

    print("Epoch ", j, " Loss ", total_epoch_loss_train)
## Temperature scaling model (imported from temperature_scaling.py)
## Used directly in function
"""
def entropy(p):
    p[p<1e-20] = 1e-20 # Deal with numerical issues
    return -torch.sum(p.mul(torch.log2(p)))

torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def confidence_accuracy_plot(source, target):
    # You need to run the training process first and then plot the graph, we directly load the saved model
    print('Using device:', DEVICE)
    torch.manual_seed(200)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N_CLASSES = 65
    BATCH_SIZE = CONFIG["batch_size"]
    baseline_int = [0, 1]
    ce_func = nn.CrossEntropyLoss()
    plt.figure(figsize=(15, 10))
    plt.plot(baseline_int, np.zeros(len(baseline_int)), c='k', linestyle='--')
    print("Source distribution: %s; Target distribution: %s" % (source, target))
    task = source[0]+target[0]

    """
    test_loader = dataloader("OfficeHome/", target, 32, False)
    # End2end with fixed r
    theta = thetaNet_e2e(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_fixed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_fixed_" + task + ".pkl")
    theta.eval()
    discriminator.eval()
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (("End2end training with fixed R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.998, 1]  # aw
    intervals = [0.0, 0.52, 0.64, 0.67, 0.7, 0.8, 0.84, 0.86, 0.9, 0.92, 0.95, 0.955,
                 0.965, 0.97, 0.975, 0.99, 0.992, 0.995, 0.997, 0.998, 1]
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print (x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="DRBA", marker="d", color="#7e1e9c")

    # End2end with alternative r
    theta = thetaNet_e2e(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_alter_" + task + ".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print ((
                   "End2end training with alternative R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.77, 0.8, 0.91, 0.92, 0.94,
                 0.96, 0.97, 0.975, 0.98, 0.983, 0.987, 0.99, 0.992, 0.995, 0.997, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="E2C2", marker="d", color="#033500")
    """

    # Relaxed DeepCORAL with alternative training
    theta = thetaNet(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_aligned_relaxed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_alter_aligned_relaxed_" + task + ".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAlignedRelaxed.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetYRelaxed.pkl")
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        theta.eval()
        discriminator.eval()
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (("Relaxed aligned data with alternatively trained R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))

    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="E2C2-R-DCORAL", marker="d", color="#0343df")

    # Aligned data with alternative training
    theta = thetaNet(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_aligned_"+task+".pkl"))
    discriminator = torch.load("models/dis_rba_alter_aligned_"+task+".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    theta.eval()
    discriminator.eval()
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (("Aligned data with alternatively trained R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas /cor_num))

    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    intervals = [0, 0.27, 0.33, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.78, 0.79, 0.8, 0.83, 0.85,
                 0.87, 0.90, 1]  # aw
    #intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95,
    #             0.955, 0.96, 0.97, 1]  # aw
    for i in range(len(intervals)-1):
        int_idx = ((confidence>=intervals[i]) == (confidence<intervals[i+1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx])/np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx]))/np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5)/100., y_value, label="E2C2-DCORAL", marker="d", color="#15b01a")

    # Aligned data with fixed R
    theta = thetaNet(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_fixed_aligned_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_fixed_aligned_" + task + ".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (("Aligned data with Fixed R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 0.98, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="DRBA-DCORAL", marker="d", color="#653700")


    test_loader = dataloader("office/", target, 32, False)
    # IID
    theta = torch.load("models/sourceOnly_"+source+"_"+target+".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = theta(data).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == label))
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (("IID: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5)/100., y_value, label="Source Only", marker=".", color="#e50000")

    # DeepCORAL
    theta = torch.load("models/deepcoral_"+source+"_"+target+".pkl", encoding="iso-8859-1")
    theta.isTrain = False
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out, _ = theta(data, None)
            target_out = target_out.detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == label))
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (("DeepCORAL: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="DCORAL", marker=".", color="#95d0fc")

    # relaxed DeepCORAL
    theta = torch.load("models/relaxed_deepcoral_" + source + "_" + target + ".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out, _ = theta(data, None, None)
            target_out = target_out.detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == label))
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (("Relaxed DeepCORAL: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="R-DCORAL", marker=".", color="#029386")

    """
    # Bayesian network
    theta = torch.load("models/bnn_"+task+".pkl")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = predict(data, theta)
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (("BNN: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="LL-SVI", marker=".", color="#f97306")
    """

    # Temperature scaling
    from temperature_scaling import ModelWithTemperature, _ECELoss
    orig_model = torch.load("models/sourceOnly_" + source + "_" + target + ".pkl", encoding="iso-8859-1")
    valid_loader = dataloader("OfficeHome/", source, 32, True)
    scaled_model = ModelWithTemperature(orig_model)
    scaled_model.set_temperature(valid_loader)
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = scaled_model(data)
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print("Temperature scaling: test_loss: %.3f, test_acc: %.4f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f" % (
        test_loss * 1e3 / test_num, test_acc / test_num, entropy_clas / test_num, mis_entropy_clas / mis_num,
        cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="Source-TS", marker=".", color="#96f97b")

    # IW + Fixed R
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_fixed_"+task+".pkl"))
    discriminator = torch.load("models/dis_iw_fixed_"+task+".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (("IW with fixed R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5)/100., y_value, label="IW", marker=".", color="#c20078")

    # IW + Alternative Training
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_alter_" + task + ".pkl"))
    discriminator = torch.load("models/dis_iw_alter_" + task + ".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (("IW with alternatively trained R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="E2IW", marker=".", color="#ff81c0")

    x_value = [str(x)[:5] for x in x_value]
    plt.xticks(np.arange(0, 110, 10) / 100., x_value[1::2], rotation=0)
    plt.xlabel("Confidence")
    plt.ylabel("Confidence - Accuracy")
    plt.grid(axis="both")
    plt.legend()
    #task_dic = {"aw": "Amazon -> Webcam", "ad": "Amazon -> Dslr", "wa": "Webcam -> Amazon", "wd": "Webcam -> Dslr", "da": "Dslr -> Amazon", "dw": "Dslr -> Webcam"}
    task_dic = {"AC": "Art -> Clipart", "AP": "Art -> Product", "AR": "Art -> RealWorld", "CA": "Clipart -> Art", "CP": "Clipart -> Product", "CR": "Clipart -> RealWorld",
                "PA": "Product -> Art", "PC": "Product -> Clipart", "PR": "Product -> RealWorld", "RA": "RealWorld -> Art", "RC": "RealWorld -> Clipart", "RP": "RealWorld -> Product"}
    plt.title("Confidence-accuracy plot on task: "+task_dic[task])
    plt.savefig("rec/conf_acc_"+task+".png")

CONFIG = {
    #"lr1": 5e-4,
    #"lr2": 5e-5,
    "lr1":1e-5,
    "lr2": 1e-5,
    "wd1": 1e-7,
    "wd2": 1e-7,
    "max_iter": 3000,
    "out_iter": 10,
    "n_classes": 31,
    "batch_size": 32,
    "upper_threshold": 1.2,
    "lower_threshold": 0.83,
    "source_prob": torch.FloatTensor([1., 0.]),
    "interval_prob": torch.FloatTensor([0.5, 0.5]),
    "target_prob": torch.FloatTensor([0., 1.]),
}

def acc_conf_plot(source, target):
    ## Plot the graphs for different methods (6 lines in total)
    print('Using device:', DEVICE)
    torch.manual_seed(200)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N_CLASSES = 31
    BATCH_SIZE = CONFIG["batch_size"]
    baseline_int = [0, 1]
    ce_func = nn.CrossEntropyLoss()
    plt.figure(figsize=(15, 10))
    plt.plot(baseline_int, np.zeros(len(baseline_int)), c='k', linestyle='--', linewidth=2)
    print("Source distribution: %s; Target distribution: %s" % (source, target))
    task = source[0] + target[0]
    intervals = [0.0, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 1]
    #intervals = [0.0, 0.6, 0.9, 0.94, 0.99, 1] #wd

    ### Source only
    test_loader = dataloader("office/", target, 32, False)
    # IID
    theta = torch.load("models/sourceOnly_" + source + "_" + target + ".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = theta(data).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == label))
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (("IID: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="Source", marker="^", color="#e50000", linewidth=2.5, markersize=12)

    # TS
    from temperature_scaling import ModelWithTemperature, _ECELoss
    orig_model = torch.load("models/sourceOnly_" + source + "_" + target + ".pkl", encoding="iso-8859-1")
    valid_loader = dataloader("office/", source, 32, True)
    scaled_model = ModelWithTemperature(orig_model)
    scaled_model.set_temperature(valid_loader)
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = scaled_model(data)
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print("Temperature scaling: test_loss: %.3f, test_acc: %.4f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f" % (
                test_loss * 1e3 / test_num, test_acc / test_num, entropy_clas / test_num, mis_entropy_clas / mis_num,
                cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="TS", marker="v", color="#96f97b", linewidth=2.5, markersize=12)
    """
    # BNN
    test_num = 0
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out = predict(data)
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (("BNN: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="LL-SVI", marker="o", color="#f97306", linewidth=2.5, markersize=12)
    """
    # IW
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_fixed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_iw_fixed_" + task + ".pkl", encoding="iso-8859-1")
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            pred = F.softmax(discriminator(data).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print ((
                   "IW with fixed R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="IW", marker="s", color="#c20078", linewidth=2.5, markersize=12)
    """
    # DeepCORAL
    theta = torch.load("models/deepcoral_" + source + "_" + target + ".pkl", encoding="iso-8859-1")
    theta.isTrain = False
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            data = data.cuda()
            target_out, _ = theta(data, None)
            target_out = target_out.detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == label))
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print ((
               "DeepCORAL: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
           (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
            mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="DCORAL", marker="D", color="#95d0fc", linewidth=2.5, markersize=12)
    """

    # E2C2-DCORAL
    theta = thetaNet(2048, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_aligned_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_alter_aligned_" + task + ".pkl")
    x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
    y_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
    x_t, y_t = x_t.to(DEVICE), y_t.to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    mis_num, cor_num, train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        theta.eval()
        discriminator.eval()
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, label))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == label).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == label).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print ((
                   "Aligned data with alternatively trained R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="$\mathregular{RESCUE}$", marker="d", color="#15b01a", linewidth=2.5, markersize=12)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    #x_axis = np.array(x_value)
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=24)
    plt.xlabel("Confidence (max prob)", fontdict={"weight":"normal", "size":24})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight":"normal", "size":24})
    #plt.grid(axis="both")
    plt.legend(shadow=True, fontsize='x-large')
    task_dic = {"aw": "Amazon -> Webcam", "ad": "Amazon -> Dslr", "wa": "Webcam -> Amazon", "wd": "Webcam -> Dslr",
                "da": "Dslr -> Amazon", "dw": "Dslr -> Webcam"}
    #task_dic = {"AC": "Art -> Clipart", "AP": "Art -> Product", "AR": "Art -> RealWorld", "CA": "Clipart -> Art",
    #            "CP": "Clipart -> Product", "CR": "Clipart -> RealWorld",
    #            "PA": "Product -> Art", "PC": "Product -> Clipart", "PR": "Product -> RealWorld",
    #            "RA": "RealWorld -> Art", "RC": "RealWorld -> Clipart", "RP": "RealWorld -> Product"}
    plt.title("Confidence-accuracy plot on task: " + task_dic[task], fontdict={"weight":"normal", "size":24})
    #plt.savefig("rec/conf_acc_part_" + task + ".png")
    import os
    if not os.path.exists("log/fig7/"):
        os.mkdir("log/fig7/")
    plt.savefig("log/fig7/conf_acc_rescue_"+task+".png")

if __name__=="__main__":
    torch.manual_seed(200)
    source = "amazon"
    target = "webcam"
    #confidence_accuracy_plot(source, target)
    acc_conf_plot(source, target)
    """
    a-d, w-a, d-a need to retrain DeepCORAL for inconsistent network structure
    """