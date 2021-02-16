import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data
import numpy as np


torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print('Using device:', DEVICE)
torch.manual_seed(200)

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


import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, SGD

log_softmax = nn.LogSoftmax(dim=1)


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

softplus = torch.nn.Softplus()

def guide(x_data, y_data):
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

#optim = SGD({"lr": 1e-3, "momentum":0.9, "weight_decay": 0})
optim = Adam({"lr": 1e-3})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


num_iterations = 30
loss = 0
N_CLASSES = 31

num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    x = x.cuda()
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean

def entropy(p):
    p[p<1e-20] = 1e-20
    return -torch.sum(p.mul(torch.log2(p)))

import math
ce_func = nn.CrossEntropyLoss()

from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
#domains = ["Art", "Clipart", "Product", "RealWorld"]
domains = ["amazon", "webcam", "dslr"]
for i in range(3):
    for j in range(3):
        if i != j:
            source = domains[i]
            target = domains[j]
            print("Task: "+source+" to "+target)

            train_loader = dataloader("office/", source, 32, True)
            test_loader = dataloader("office/", target, 32, False)

            net = NN(N_CLASSES)
            for j in range(num_iterations):
                loss = 0
                for batch_id, data in enumerate(train_loader):
                    # calculate the loss and take a gradient step
                    loss += svi.step(data[0].cuda(), data[1].cuda())
                    normalizer_train = len(train_loader.dataset)
                    total_epoch_loss_train = loss / normalizer_train

                print("Epoch ", j, " Loss ", total_epoch_loss_train)
            entropy_clas, test_loss, test_acc, mis_entropy_clas, mis_num, cor_entropy_clas, cor_num, num_test = 0, 0, 0, 0, 0, 0, 0, 0
            b_score = 0
            for data, label in test_loader:
                data, label = data.cuda(), label.cuda()
                num_test += data.shape[0]
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
                enc = OneHotEncoder(categories="auto")
                label = label.cpu().numpy().reshape(-1, 1)
                label = enc.fit_transform(label).toarray()
                for j in range(data.shape[0]):
                    y_true = np.concatenate((label[j], np.zeros(N_CLASSES - label[j].shape[0])))
                    b_score += brier_score_loss(y_true, prediction_t[j].cpu().numpy())
            print("test_loss: %.3f, test_acc: %.4f, mis_ent_clas: %.3f" %
                  (test_loss * 1e3 / num_test,  test_acc / num_test, mis_entropy_clas / mis_num))
            print("Brier score: ", b_score / num_test)

            print('Prediction when network is forced to predict')
            correct = 0
            total = 0
            for j, data in enumerate(test_loader):
                images, labels = data
                predicted = np.argmax(predict(images).cpu().numpy(), axis=1)
                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum().item()
            acc = 100. * correct / total
            print("accuracy: %.2f %%" % acc)
            torch.save(net, "models/bnn_"+source[0]+target[0]+".pkl")

