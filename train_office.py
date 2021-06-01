"""
Do training and testing on the Office-31 dataset
"""
import numpy as np
import torch
import math
import torch.nn as nn
import torchvision
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import torch.utils.data as Data
from model_layers import ClassifierLayer, RatioEstimationLayer, Flatten, GradLayer, IWLayer
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.manifold import TSNE
from deep_coral import DeepCoral, dataloader
import heapq
from sklearn.metrics import brier_score_loss


torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

class Discriminator_IW(nn.Module):
    def __init__(self, n_features):
        super(Discriminator_IW, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512),
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
        p = self.net(x)
        return p

class thetaNet(nn.Module):
    def __init__(self, n_features, n_output):
        super(thetaNet, self).__init__()
        self.extractor = torch.nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.Tanh(),
            #nn.Linear(1024, 512),
            #nn.Tanh(),
            #nn.Linear(512, 256),
            #nn.Tanh(),
        )
        self.classifier = ClassifierLayer(1024, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x_s = self.extractor(x_s)
        x = self.classifier(x_s, y_s, r)
        return x

class iid_theta(nn.Module):
    def __init__(self, n_features, n_output):
        super(iid_theta, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),

            #nn.Linear(512, 512),
            #nn.Tanh(),
            #nn.Linear(512, 512),
            #nn.Tanh(),
            #nn.Linear(512, 512),
            #nn.Tanh(),
            torch.nn.Linear(256, n_output),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class IWNet(nn.Module):
     def __init__(self, n_features, n_output):
        super(IWNet, self).__init__()
        self.extractor = torch.nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            #nn.Linear(512, 512),
            #nn.Tanh(),
            #nn.Linear(512, 512),
            #nn.Tanh(),
            #nn.Linear(512, 512),
            #nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            #torch.nn.Linear(1024, 64),
        )
        self.IW = IWLayer(256, n_output)

     def forward(self, x_s, y_s, r):
         x_s = self.extractor(x_s)
         x = self.IW(x_s, y_s, r)
         return x

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
        x = x.view(x.size(0), 2048)
        p = self.net(x)
        p = self.grad_r(p, nn_output, prediction, p_t, pass_sign)
        return p

class theta_e2e(nn.Module):
    def __init__(self, n_output):
        super(theta_e2e, self).__init__()
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
        x_s = x_s.view(x_s.size(0), 2048)
        x_s = self.extractor(x_s)
        x = self.classifier(x_s, y_s, r)
        return x

def entropy(p):
    p[p<1e-20] = 1e-20 # Deal with numerical issues
    return -torch.sum(p.mul(torch.log2(p)))

CONFIG = {
    #"lr1": 5e-4,
    #"lr2": 5e-5,
    "lr1":1e-3,
    "lr2": 1e-3,
    "wd1": 1e-7,
    "wd2": 1e-7,
    "max_iter": 5000,
    "out_iter": 10,
    "n_classes": 31,
    "batch_size": 64,
    "upper_threshold": 1.2,
    "lower_threshold": 0.83,
    "source_prob": torch.FloatTensor([1., 0.]),
    "interval_prob": torch.FloatTensor([0.5, 0.5]),
    "target_prob": torch.FloatTensor([0., 1.]),
}

LOGDIR = os.path.join("runs", datetime.now().strftime("%Y%m%d%H%M%S"))

def iid_baseline(x_s, y_s, x_t, y_t, task="aw_coral"):
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = CONFIG["max_iter"]
    OUT_ITER = CONFIG["out_iter"]
    N_FEATURES= x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    theta = iid_theta(N_FEATURES, N_CLASSES)
    theta = theta.cuda()
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-08)
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    ce_func = nn.CrossEntropyLoss()
    y_s = y_s.long()
    y_t = y_t.long()
    train_dataset = Data.TensorDataset(x_s, y_s)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    batch_num_test = len(test_loader.dataset)
    batch_num_train = len(train_loader)
    train_loss, train_acc, test_loss, test_acc, entropy_clas, mis_entropy_clas, cor_entropy_clas = 0, 0, 0, 0, 0, 0, 0
    early_stop = 0
    best_train_loss = 1e8
    writer = SummaryWriter(LOGDIR)
    for epoch in range(MAX_ITER):
        theta.train()
        for batch_x_s, batch_y_s in train_loader:
            prob = theta(batch_x_s)
            prediction = (F.softmax(prob, dim=1)).detach()
            loss = ce_func(prob, torch.argmax(batch_y_s, dim=1))
            optimizer_theta.zero_grad()
            loss.backward()
            optimizer_theta.step()

            train_loss += float(loss.detach())
            train_acc += float(torch.sum(torch.argmax(prediction, dim=1) == torch.argmax(batch_y_s, dim=1)))/BATCH_SIZE

        if (epoch+1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train)
            train_acc /= (OUT_ITER * batch_num_train)
            if train_loss >= best_train_loss:
                early_stop += 1
            else:
                early_stop = 0
                best_train_loss = train_loss
                torch.save(theta.state_dict(), "models/theta_office_param_iid_"+task+".pkl")
            #theta.eval()
            mis_num = 0
            cor_num = 0
            with torch.no_grad():
                for data, label in test_loader:
                    target_out = theta(data).detach()
                    prediction_t = F.softmax(target_out, dim=1)
                    entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                    test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                    test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
                    mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    mis_pred = prediction_t[mis_idx]
                    cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    cor_pred = prediction_t[cor_idx]
                    mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                    mis_num += mis_idx.shape[0]
                    cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                    cor_num += cor_idx.shape[0]
            print (
                "{} epochs: train_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                (epoch + 1), train_loss * 1e3, test_loss * 1e3 / batch_num_test, train_acc,
                             test_acc / batch_num_test, entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
            )
            writer.add_scalars("train_loss", {"iid": train_loss}, (epoch+1))
            writer.add_scalars("test_loss", {"iid": test_loss/batch_num_test}, (epoch+1))
            writer.add_scalars("train_acc", {"iid": train_acc}, (epoch+1))
            writer.add_scalars("test_acc", {"iid": test_acc/batch_num_test}, (epoch+1))
            writer.add_scalars("ent", {"iid": entropy_clas/batch_num_test}, (epoch+1))
            writer.add_scalars("mis_ent", {"iid": mis_entropy_clas/batch_num_test}, (epoch+1))
            train_loss, train_acc, test_loss, test_acc, entropy_clas, mis_entropy_clas, cor_entropy_clas = 0, 0, 0, 0, 0, 0, 0
        if early_stop > 3:
            print "Training Process Already Converges At Epoch %s" % (epoch+1)
            break
    writer.close()

def train_iw(x_s, y_s, x_t, y_t, task="aw_coral"):
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = CONFIG["max_iter"]
    OUT_ITER = CONFIG["out_iter"]
    N_FEATURES= x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator_IW(N_FEATURES)
    theta = IWNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.99, 0.999), eps=1e-8)
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    batch_num_train = max(n_train, n_test)/BATCH_SIZE + 1
    ce_func = nn.CrossEntropyLoss()
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    batch_num_test = len(test_loader.dataset)
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    early_stop = 0
    best_train_loss = 1e8
    writer = SummaryWriter(LOGDIR)
    for epoch in range(MAX_ITER):
        theta.train()
        discriminator.train()
        train_sample_order = np.arange(n_train)
        test_sample_order = np.arange(n_test)
        np.random.shuffle(train_sample_order)
        np.random.shuffle(test_sample_order)
        for step in range(batch_num_train):
            if (step * BATCH_SIZE)%n_train > ((step + 1) * BATCH_SIZE)%n_train:
                batch_id_s = train_sample_order[(step * BATCH_SIZE)%n_train:-1]
                batch_id_s = np.concatenate((batch_id_s, train_sample_order[0:BATCH_SIZE - batch_id_s.shape[0]]))
            else:
                batch_id_s = train_sample_order[step * BATCH_SIZE % n_train:(step + 1) * BATCH_SIZE % n_train]
            if (step * BATCH_SIZE)%n_test > ((step+1) * BATCH_SIZE)%n_test:
                batch_id_t = test_sample_order[(step * BATCH_SIZE)%n_test:-1]
                batch_id_t = np.concatenate((batch_id_t, test_sample_order[0:BATCH_SIZE - batch_id_t.shape[0]]))
            else:
                batch_id_t = test_sample_order[(step * BATCH_SIZE)%n_test:((step + 1) * BATCH_SIZE)%n_test]
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_t = x_t[batch_id_t]
            batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2*BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_y = batch_y[shuffle_idx]
            prob = discriminator(batch_x)
            loss_dis = ce_func(prob, batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_t/p_s
            pos_source, pos_target = np.zeros((BATCH_SIZE, )), np.zeros((BATCH_SIZE, ))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx==idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            for idx in range(BATCH_SIZE, 2*BATCH_SIZE):
                pos_target[idx-BATCH_SIZE] = np.where(shuffle_idx==idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)

            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)

            optimizer_dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_dis.step()

            loss_theta = torch.sum(theta_out)
            optimizer_theta.zero_grad()
            loss_theta.backward()
            optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float()/BATCH_SIZE
            dis_loss += float(loss_dis)
            dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == batch_y).float()/(2*BATCH_SIZE)
            writer.add_scalars("rba_iw", {"source": torch.mean(r_source.detach()), "target": torch.mean(r_target.detach())}, (epoch+1))

        if (epoch+1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train)
            dis_acc /=(OUT_ITER * batch_num_train)
            if train_loss >= best_train_loss:
                early_stop += 1
            else:
                early_stop = 0
                best_train_loss = train_loss
                torch.save(discriminator, "models/dis_office_iw_"+task+".pkl")
                torch.save(theta.state_dict(), "models/theta_office_param_iw_"+task+".pkl")
            #theta.eval()
            #discriminator.eval()
            mis_num = 0
            cor_num = 0
            with torch.no_grad():
                for data, label in test_loader:
                    pred = F.softmax(discriminator(data).detach(), dim=1)
                    entropy_dis += entropy(pred)
                    r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
                    target_out = theta(data, None, r_target).detach()
                    prediction_t = F.softmax(target_out, dim=1)
                    entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                    test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                    test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
                    mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    mis_pred = prediction_t[mis_idx]
                    cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    cor_pred = prediction_t[cor_idx]
                    mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                    mis_num += mis_idx.shape[0]
                    cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                    cor_num += cor_idx.shape[0]
                print (
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / batch_num_test, train_acc,
                                 test_acc / batch_num_test, dis_acc, entropy_dis / batch_num_test, \
                                 entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
            writer.add_scalars("train_loss", {"iw": train_loss}, (epoch+1))
            writer.add_scalars("test_loss", {"iw": test_loss/batch_num_test}, (epoch+1))
            writer.add_scalars("train_acc", {"iw": train_acc}, (epoch+1))
            writer.add_scalars("test_acc", {"iw": test_acc/batch_num_test}, (epoch+1))
            writer.add_scalars("dis_loss", {"iw": dis_loss}, (epoch+1))
            writer.add_scalars("dis_acc", {"iw": dis_acc}, (epoch+1))
            writer.add_scalars("ent", {"iw": entropy_clas/batch_num_test}, (epoch+1))
            writer.add_scalars("mis_ent", {"iw": mis_entropy_clas/batch_num_test}, (epoch+1))
            writer.add_scalars("dis_ent", {"iw": entropy_dis/batch_num_test}, (epoch+1))
            train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cor_entropy_clas = 0
        if early_stop > 3:
            print "Training Process Already Converges At Epoch %s" % (epoch+1)
            break
    writer.close()

def train_aligned(x_s, y_s, x_t, y_t, x_s_orig, x_t_orig):
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 300
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.99, 0.999), eps=1e-6)
    test_dataset = Data.TensorDataset(x_t, x_t_orig, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    batch_num_train = n_train / BATCH_SIZE + 1
    batch_num_test = len(test_loader.dataset)
    ce_func = nn.CrossEntropyLoss()
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    early_stop = 0
    best_train_loss = 1e8
    best_dis_loss = 1e8
    best_test_loss = 1e8
    writer = SummaryWriter(LOGDIR)
    """
    for epoch in range(100):
        discriminator.train()
        for step in range(batch_num_train):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s_orig = x_s_orig[batch_id_s]
            batch_x_t_orig = x_t_orig[batch_id_t]
            batch_x = torch.cat((batch_x_s_orig, batch_x_t_orig), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_y = batch_y[shuffle_idx]
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = ce_func(prob, batch_y)
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
            prediction = F.softmax(prob, dim=1).detach()
            dis_loss += float(loss_dis.detach())
            dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == batch_y).float() / (2 * BATCH_SIZE)
        if (epoch+1) % OUT_ITER == 0:
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            discriminator.eval()
            with torch.no_grad():
                for data, data_orig, label in test_loader:
                    pred = F.softmax(discriminator(data_orig, None, None, None, None).detach(), dim=1)
            print "Epoch: %d, Discriminator loss: %.4f, accuracy %.4f:" % ((epoch+1), dis_loss, dis_acc)
            dis_loss = 0
            dis_acc = 0
    """
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        train_sample_order = np.arange(n_train)
        test_sample_order = np.arange(n_test)
        np.random.shuffle(train_sample_order)
        np.random.shuffle(test_sample_order)
        for step in range(batch_num_train):
            if (step * BATCH_SIZE) % n_train > ((step + 1) * BATCH_SIZE) % n_train:
                batch_id_s = train_sample_order[(step * BATCH_SIZE) % n_train:-1]
                batch_id_s = np.concatenate((batch_id_s, train_sample_order[0:BATCH_SIZE - batch_id_s.shape[0]]))
            else:
                batch_id_s = train_sample_order[step * BATCH_SIZE % n_train:(step + 1) * BATCH_SIZE % n_train]
            if (step * BATCH_SIZE) % n_test > ((step + 1) * BATCH_SIZE) % n_test:
                batch_id_t = test_sample_order[(step * BATCH_SIZE) % n_test:-1]
                batch_id_t = np.concatenate((batch_id_t, test_sample_order[0:BATCH_SIZE - batch_id_t.shape[0]]))
            else:
                batch_id_t = test_sample_order[(step * BATCH_SIZE) % n_test:((step + 1) * BATCH_SIZE) % n_test]
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_s_orig = x_s_orig[batch_id_s]
            batch_x_t_orig = x_t_orig[batch_id_t]
            batch_x_t = x_t[batch_id_t]
            batch_x = torch.cat((batch_x_s_orig, batch_x_t_orig), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2 * BATCH_SIZE)
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = ce_func(prob, batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source, pos_target = np.zeros((BATCH_SIZE,)), np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx == idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            for idx in range(BATCH_SIZE, 2 * BATCH_SIZE):
                pos_target[idx - BATCH_SIZE] = np.where(shuffle_idx == idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)
            p_t_target = p_t[pos_target]
            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)

            nn_out = theta(batch_x_t, None, r_target.detach())
            pred_target = F.softmax(nn_out, dim=1)

            prob_grad_r = discriminator(batch_x_t_orig, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                        sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))

            optimizer_dis.zero_grad()
            loss_r.backward(retain_graph=True)
            optimizer_dis.step()

            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

            loss_theta = torch.sum(theta_out)
            optimizer_theta.zero_grad()
            loss_theta.backward()
            optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())
            dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == batch_y).float() / (2 * BATCH_SIZE)
            writer.add_scalars("rba_r",
                               {"source": torch.mean(r_source.detach()), "target": torch.mean(r_target.detach())},
                               (epoch + 1))
        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            mis_num = 0
            cor_num = 0
            with torch.no_grad():
                for data, data_orig, label in test_loader:
                    pred = F.softmax(discriminator(data_orig, None, None, None, None).detach(), dim=1)
                    entropy_dis += entropy(pred)
                    r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                    target_out = theta(data, None, r_target).detach()
                    prediction_t = F.softmax(target_out, dim=1)
                    entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                    test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                    test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
                    mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    mis_pred = prediction_t[mis_idx]
                    cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    cor_pred = prediction_t[cor_idx]
                    mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                    mis_num += mis_idx.shape[0]
                    cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                    cor_num += cor_idx.shape[0]
                print (
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / batch_num_test, train_acc,
                                 test_acc / batch_num_test, dis_acc, entropy_dis / batch_num_test, \
                                 entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                """
                test_loss, test_acc, entropy_dis, entropy_clas, mis_entropy_clas, \
                cor_entropy_clas = test_mixed_1(discriminator, theta, x_s, y_s, x_t, y_t, x_s_orig, x_t_orig)
                print (
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3, train_acc,
                    test_acc, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas, cor_entropy_clas
                )
                """
                writer.add_scalars("train_loss", {"rba": train_loss}, (epoch + 1))
                writer.add_scalars("test_loss", {"rba": test_loss / batch_num_test}, (epoch + 1))
                writer.add_scalars("train_acc", {"rba": train_acc}, (epoch + 1))
                writer.add_scalars("test_acc", {"rba": test_acc / batch_num_test}, (epoch + 1))
                writer.add_scalars("dis_loss", {"rba": dis_loss}, (epoch + 1))
                writer.add_scalars("dis_acc", {"rba": dis_acc}, (epoch + 1))
                writer.add_scalars("ent", {"rba": entropy_clas / batch_num_test}, (epoch + 1))
                writer.add_scalars("mis_ent", {"rba": mis_entropy_clas / batch_num_test}, (epoch + 1))
                writer.add_scalars("dis_ent", {"rba": entropy_dis / batch_num_test}, (epoch + 1))
                if train_loss >= best_train_loss and dis_loss >= best_dis_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    best_dis_loss = dis_loss
                    best_test_loss = test_loss
                    early_stop = 0
                    torch.save(discriminator, "models/dis_office_rba.pkl")
                    torch.save(theta.state_dict(), "models/theta_office_param_rba.pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 20:
            print "Training Process Converges Until Epoch %s" % (epoch + 1)
            break
    writer.close()

def test_mixed_1(discriminator, theta, x_s, y_s, x_t, y_t, x_s_orig, x_t_orig):
    BATCH_SIZE = 64
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    batch_num_test = x_t.shape[0]/BATCH_SIZE + 1
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    ce_func = nn.CrossEntropyLoss()
    test_loss, test_acc, entropy_dis, entropy_clas, mis_entropy_clas, cor_entropy_clas = 0, 0, 0, 0, 0, 0
    mis_num= 0
    cor_num = 0
    with torch.no_grad():
        for iter in range(batch_num_test):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_s_orig = x_s_orig[batch_id_s]
            batch_x_t_orig = x_t_orig[batch_id_t]
            batch_x_t = x_t[batch_id_t]
            batch_y_t = y_t[batch_id_t]
            batch_x = torch.cat((batch_x_s_orig, batch_x_t_orig), dim=0)

            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            prob = discriminator(batch_x, None, None, None, None)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source, pos_target = np.zeros((BATCH_SIZE,)), np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE, 2 * BATCH_SIZE):
                pos_target[idx - BATCH_SIZE] = np.where(shuffle_idx == idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)
            target_out = theta(batch_x_t, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(batch_y_t, dim=1)))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(batch_y_t, dim=1)).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(batch_y_t, dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(batch_y_t, dim=1)).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
    #print r_target
    return test_loss/(batch_num_test*BATCH_SIZE), test_acc/(batch_num_test*BATCH_SIZE), \
           entropy_dis/(batch_num_test*BATCH_SIZE), entropy_clas/(batch_num_test*BATCH_SIZE), \
           mis_entropy_clas/mis_num, cor_entropy_clas/cor_num

def test_mixed_2(x_s, y_s, x_t, y_t, x_s_orig, x_t_orig):
    BATCH_SIZE = 32
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    batch_num_test = x_t.shape[0] / BATCH_SIZE + 1
    theta = thetaNet(N_FEATURES, N_CLASSES)
    discriminator = torch.load("models/dis_office_rba.pkl")
    theta.load_state_dict(torch.load("models/theta_office_param_rba.pkl"))
    discriminator.eval()
    theta.eval()
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    ce_func = nn.CrossEntropyLoss()
    test_loss, test_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0
    with torch.no_grad():
        for iter in range(batch_num_test):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_s_orig = x_s_orig[batch_id_s]
            batch_x_t_orig = x_t_orig[batch_id_t]
            batch_x_t = x_t[batch_id_t]
            batch_y_t = y_t[batch_id_t]
            batch_x_orig = torch.cat((batch_x_s_orig, batch_x_t_orig), dim=0)
            batch_x = torch.cat((batch_x_s, batch_x_t),dim=0)
            batch_y = torch.cat((batch_y_s, batch_y_t), dim=0)

            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_x_orig = batch_x_orig[shuffle_idx]
            batch_y = batch_y[shuffle_idx]

            prob = discriminator(batch_x_orig, None, None, None, None)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source, pos_target = np.zeros((BATCH_SIZE,)), np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE, 2 * BATCH_SIZE):
                pos_target[idx - BATCH_SIZE] = np.where(shuffle_idx == idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)

            target_out = theta(batch_x, None, r).detach()
            target_out = target_out[pos_target]
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(batch_y[pos_target], dim=1)))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(batch_y[pos_target], dim=1)).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(batch_y[pos_target], dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
    return test_loss, test_acc, entropy_dis, entropy_clas, mis_entropy_clas

def plot_2d(x_s, x_t, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model = torch.load("models/dis_office_rba.pkl")
    print list(model.net.children())
    new_net = nn.Sequential(*list(model.net.children())[:-1])
    BATCH_SIZE = 64
    batch_num_train = x_s.shape[0] / BATCH_SIZE + 1
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    #for name, param in new_classifier.named_parameters():
    #    print name, param
    X = torch.FloatTensor(np.ones((1, 2048)))
    Y = torch.LongTensor(np.ones((1, )))
    with torch.no_grad():
        new_net.eval()
        for step in range(batch_num_train):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s_orig = x_s[batch_id_s]
            batch_x_t_orig = x_t[batch_id_t]
            batch_x = torch.cat((batch_x_s_orig, batch_x_t_orig), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_y = batch_y[shuffle_idx]
            #out = new_net(batch_x)
            out = batch_x
            X = torch.cat((X, out.cpu()), dim=0)
            Y = torch.cat((Y, batch_y.cpu()), dim=0)
    X = X[1:]
    Y = Y[1:]
    print X.shape, Y.shape
    X = X.detach().numpy()
    Y = Y.detach().numpy()
    X_transformed = TSNE(n_components=2).fit_transform(X)
    print X_transformed.shape
    source_x = X[np.where(Y==0)[0]]
    target_x = X[np.where(Y==1)[0]]
    np.save("data/s_x.npy", source_x)
    np.save("data/t_x.npy", target_x)


    plt.scatter(source_x[:, 0], source_x[:, 1], label="source")
    plt.scatter(target_x[:, 0], target_x[:, 1], label="target")
    plt.legend()
    plt.savefig(path)

def early_fix(x_s, y_s, x_t, y_t):
    BATCH_SIZE = 64
    MAX_ITER = 2000
    OUT_ITER = 10
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    #optimizer_theta = torch.optim.Adagrad(theta.parameters(), lr=CONFIG["lr1"], lr_decay=1e-7, weight_decay=CONFIG["wd1"])
    #optimizer_dis = torch.optim.Adagrad(discriminator.parameters(), lr=CONFIG["lr2"], lr_decay=1e-7, weight_decay=CONFIG["wd2"])
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=CONFIG["lr1"], betas=(0.99, 0.999), eps=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr2"], betas=(0.99, 0.999), eps=1e-6)
    batch_num_train = x_s.shape[0] / BATCH_SIZE + 1
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    batch_num_test = len(test_loader.dataset)
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    ce_func = nn.CrossEntropyLoss()
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    early_stop = 0
    best_test_loss = 1e8
    best_train_loss = 1e8
    writer = SummaryWriter(LOGDIR)
    early_fix_point = -1
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        early_fix_point += 1
        for step in range(batch_num_train):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_s_orig = x_s[batch_id_s]
            batch_x_t_orig = x_t[batch_id_t]
            batch_x_t = x_t[batch_id_t]
            batch_x = torch.cat((batch_x_s_orig, batch_x_t_orig), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_y = batch_y[shuffle_idx]
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = ce_func(prob, batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source, pos_target = np.zeros((BATCH_SIZE,)), np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx == idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            for idx in range(BATCH_SIZE, 2 * BATCH_SIZE):
                pos_target[idx - BATCH_SIZE] = np.where(shuffle_idx == idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)
            p_t_target = p_t[pos_target]
            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)

            nn_out = theta(batch_x_t, None, r_target.detach())
            pred_target = F.softmax(nn_out, dim=1)

            prob_grad_r = discriminator(batch_x_t_orig, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                        sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))

            if early_fix_point < 20:
                optimizer_dis.zero_grad()
                loss_r.backward(retain_graph=True)
                optimizer_dis.step()

                optimizer_dis.zero_grad()
                loss_dis.backward()
                optimizer_dis.step()

            loss_theta = torch.sum(theta_out)
            optimizer_theta.zero_grad()
            loss_theta.backward()
            optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())
            dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == batch_y).float() / (2 * BATCH_SIZE)
            writer.add_scalars("rba_r",
                               {"source": torch.mean(r_source.detach()), "target": torch.mean(r_target.detach())},
                               (epoch + 1))

        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            discriminator.eval()
            theta.eval()
            mis_num = 0
            cor_num = 0
            with torch.no_grad():
                for data, label in test_loader:
                    pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
                    entropy_dis += entropy(pred)
                    r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                    target_out = theta(data, None, r_target).detach()
                    prediction_t = F.softmax(target_out, dim=1)
                    entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                    test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                    test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
                    mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    mis_pred = prediction_t[mis_idx]
                    cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    cor_pred = prediction_t[cor_idx]
                    mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                    mis_num += mis_idx.shape[0]
                    cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                    cor_num += cor_idx.shape[0]
                print (
                    "{} epoches: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / batch_num_test, train_acc,
                                 test_acc / batch_num_test, dis_acc, entropy_dis / batch_num_test, \
                                 entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas /cor_num
                )
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    # best_dis_loss = dis_loss
                    best_test_loss = test_loss
                    early_stop = 0
                    if early_fix_point < 20:
                        torch.save(discriminator, "models/ef_dis_office_rba_"+task_name+".pkl")
                    torch.save(theta.state_dict(), "models/ef_theta_office_param_rba_"+task_name+".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 20:
            print "Training Process Converges At Epoch %s, reaches best at Epoch %s" % ((epoch+1), (epoch+1-200))
            break


def confidence_accuracy_plot(x_t, y_t, x_t_orig, task="aw_coral"):
    # You need to run the training process first and then plot the graph, we directly load the saved model
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #import seaborn as sns
    #styles = ['darkgrid', 'dark', 'white', 'whitegrid', 'tricks']
    #sns.set_style(styles[0])
    N_CLASSES = CONFIG["n_classes"]
    N_FEATURES = x_t.shape[1]
    BATCH_SIZE = CONFIG["batch_size"]

    # RBA
    theta = thetaNet(N_FEATURES, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_office_param_rba_"+task+".pkl"))
    discriminator = torch.load("models/dis_office_rba_"+task+".pkl")
    ce_func = nn.CrossEntropyLoss()
    test_dataset = Data.TensorDataset(x_t, x_t_orig, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    #discriminator.eval()
    #theta.eval()
    mis_num = 0
    cor_num = 0
    batch_num_test = len(test_loader.dataset)
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, data_orig, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data_orig, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (
            "RBA: test_loss:{:.3f}, test_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
            test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, \
                         entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas /cor_num
        )
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    #intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.993, 0.998, 1]
    #intervals = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    #             0.90, 0.95, 1]
    intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 1]
    baseline_int = [0, 1]
    x_value = []
    y_value = []
    for i in range(len(intervals)-1):
        int_idx = ((confidence>=intervals[i]) == (confidence<intervals[i+1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx])/np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx]))/np.sum(int_idx))
    plt.figure()
    plt.plot(np.arange(0, 100, 5)/100., y_value, label="RBA", marker=".")
    plt.plot(baseline_int, np.zeros(len(baseline_int)), c='k', linestyle='--')

    # IID
    theta = iid_theta(N_FEATURES, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_office_param_iid_"+task+".pkl"))
    ce_func = nn.CrossEntropyLoss()
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    #theta.eval()
    mis_num = 0
    cor_num = 0
    batch_num_test = len(test_loader.dataset)
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            target_out = theta(data).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)))
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (
        "IID: test_loss:{:.3f}, test_acc: {:.4f}, ent_cla: {:.3f}, mis_ent_clas: {:.3f}, ent_clas_cor: {:.3f}").format(
         test_loss * 1e3 / test_num, test_acc / test_num,
                     entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
    )
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    plt.plot(np.arange(0, 100, 5)/100., y_value, label="IID", marker=".")

    # Original,don't do it with TCA features...
    """
    tmp_dic = {"a": "amazon", "w": "webcam", "d": "dslr", "A": "Art", "C": "Clipart", "P": "Product", "R": "RealWorld"}
    src = tmp_dic[task[0]]
    trg = tmp_dic[task[1]]
    model = torch.load("models/deepcoral_" + src + "_" + trg + ".pkl")
    model.eval()
    ce_func = nn.CrossEntropyLoss()
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    mis_num = 0
    cor_num = 0
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            target_out = model.cls_fc(data)
            target_out = target_out.detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)))
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
    print (
        "Original Model: test_loss:{:.3f}, test_acc: {:.4f}, ent_cla: {:.3f}, ent_cla_mis: {:.3f}, ent_clas_cor: {:.3f}").format(
        test_loss * 1e3 / test_num, test_acc / test_num,
        entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
    )
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    plt.plot(np.arange(0, 100, 5) / 100., y_value, label="Original", marker=".")
    """

    # IW
    theta = IWNet(N_FEATURES, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_office_param_iw_"+task+".pkl"))
    discriminator = torch.load("models/dis_office_iw_"+task+".pkl")
    ce_func = nn.CrossEntropyLoss()
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    #discriminator.eval()
    #theta.eval()
    mis_num = 0
    cor_num = 0
    batch_num_test = len(test_loader.dataset)
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    test_num = 0
    with torch.no_grad():
        for data, label in test_loader:
            test_num += data.shape[0]
            pred = F.softmax(discriminator(data).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 1] / pred[:, 0]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (
            "IW: test_loss:{:.3f}, test_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
            test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, \
            entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
        )
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    plt.plot(np.arange(0, 100, 5)/100., y_value, label="IW", marker=".")
    x_value = [str(x)[:5] for x in x_value]
    plt.xticks(np.arange(0, 110, 10) / 100., x_value[1::2], rotation=0)
    plt.xlabel("Confidence")
    plt.ylabel("Confidence - Accuracy")
    plt.grid(axis="both")
    plt.legend()
    task_dic = {"aw": "(Amazon -> Webcam)", "ad": "(Amazon -> Dslr)", "wa": "(Webcam -> Amazon)", "wd": "(Webcam -> Dslr)", "da": "(Dslr -> Amazon)", "dw": "(Dslr -> Webcam)"}
    if len(task) == 6:
        plt.title("Office31 based on TCA "+task_dic[task[:2]])
    else:
        plt.title("Office31 based on DeepCORAL "+task_dic[task[:2]])
    plt.savefig("rec/office_conf_acc_"+task+".png")

def softlabels(x_s, y_s, x_t, y_t, task="aw_coral"):
    ## Changes the hard labels of the original dataset to soft ones (probabilities), such as (0.5, 0.5) for samples with
    ## large density ratio in the target domain
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 3000
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=CONFIG["lr1"], betas=(0.99, 0.999), eps=1e-4, weight_decay=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr2"], betas=(0.99, 0.999), eps=1e-4, weight_decay=1e-8)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    ce_func = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction="mean")
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    early_stop = 0
    best_train_loss = 1e8
    best_dis_loss = 1e8
    best_test_loss = 1e8
    writer = SummaryWriter(LOGDIR)
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat((torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print "Originally %d source data, %d target data" % (n_train, n_test)
    train_loss_list = []
    test_loss_list = []
    dis_loss_list = []
    test_acc_list = []
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        train_sample_order = np.arange(n_train)
        test_sample_order = np.arange(n_test)
        #np.random.shuffle(train_sample_order)
        #np.random.shuffle(test_sample_order)
        convert_data_idx_s = torch.eq(whole_label_dis[0:n_train, 0], CONFIG["interval_prob"][0]).nonzero().view(-1, ).cpu().numpy()
        remain_data_idx_t = torch.eq(whole_label_dis[n_train:n_train+n_test, 1], 1).nonzero().view(-1, ).cpu().numpy()
        if (epoch+1) % OUT_ITER == 0:
            interval_s = convert_data_idx_s.shape[0]
            remain_target = remain_data_idx_t.shape[0]
            print "Currently %d removed source data, %d remained target data, %d interval source data, %d interval target data" % (
                n_train-interval_s, remain_target, interval_s, n_test-remain_target
            )
        batch_num_train = max(n_train, n_test) / BATCH_SIZE + 1
        for step in range(batch_num_train):
            if convert_data_idx_s.shape[0] < BATCH_SIZE:
                batch_id_s = np.random.choice(train_sample_order, BATCH_SIZE, replace=False)
            else:
                batch_id_s = np.random.choice(convert_data_idx_s, BATCH_SIZE, replace=False)
            if remain_data_idx_t.shape[0] < BATCH_SIZE:
                batch_id_t = np.random.choice(test_sample_order, BATCH_SIZE, replace=False)
            else:
                batch_id_t = np.random.choice(remain_data_idx_t, BATCH_SIZE, replace=False)
            batch_id_t = batch_id_t + n_train
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_t = whole_data[batch_id_t]
            batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
            batch_y = torch.cat((whole_label_dis[batch_id_s], whole_label_dis[batch_id_t]), dim=0)
            batch_y = batch_y.to(DEVICE)
            shuffle_idx = np.arange(2 * BATCH_SIZE)

            # Feed Forward
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = bce_loss(F.softmax(prob, dim=1), batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source, pos_target = np.zeros((BATCH_SIZE,)), np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx == idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            for idx in range(BATCH_SIZE, 2 * BATCH_SIZE):
                pos_target[idx - BATCH_SIZE] = np.where(shuffle_idx == idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)
            p_t_target = p_t[pos_target]
            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)
            nn_out = theta(batch_x_t, None, r_target.detach())
            pred_target = F.softmax(nn_out, dim=1)
            prob_grad_r = discriminator(batch_x_t, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                        sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))
            loss_theta = torch.sum(theta_out)

            # Backpropagate
            if (step+1) % 1 == 0:
                optimizer_dis.zero_grad()
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()

            if (step+1) % 1 == 0:
                optimizer_dis.zero_grad()
                loss_r.backward(retain_graph=True)
                optimizer_dis.step()

            if (step+1) % 1 == 0:
                optimizer_theta.zero_grad()
                loss_theta.backward()
                optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())
            #dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == torch.argmax(batch_y, dim=1)).float() / (2 * BATCH_SIZE)
            writer.add_scalars("rba_r",
                               {"source": torch.mean(r_source.detach()), "target": torch.mean(r_target.detach())},
                               (epoch + 1))

        ## Change source to interval section, and only use the changed ones for training
        if (epoch + 1) % 15 == 0:
            whole_label_dis = torch.cat((torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
            pred_tmp = F.softmax(discriminator(whole_data, None, None, None, None).detach(), dim=1)
            r = (pred_tmp[:, 0] / pred_tmp[:, 1]).reshape(-1, 1)
            #print r[:10]
            #print r[n_train:10+n_train]
            pos_source = np.arange(n_train)
            source_ratio = r[pos_source].view(-1, ).cpu().numpy()
            num_convert = int(source_ratio.shape[0] * 0.5)
            int_convert = heapq.nsmallest(num_convert, range(len(source_ratio)), source_ratio.take)
            invert_idx = pos_source[int_convert].astype(np.int32)
            whole_label_dis[invert_idx] = CONFIG["interval_prob"]

            pos_target = np.arange(n_train, n_train+n_test)
            target_ratio = r[pos_target].view(-1, ).cpu().numpy()
            num_convert = int(target_ratio.shape[0] * 0.0)
            int_convert = heapq.nlargest(num_convert, range(len(target_ratio)), target_ratio.take)
            invert_idx = pos_target[int_convert].astype(np.int32)
            whole_label_dis[invert_idx] = CONFIG["interval_prob"]

        #r_list.append(r[-10])
        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            # Do not do eval(), fucking bug; Probably can be solved by normalizing input data
            # and since we do not have batch_norm, there should be no influence with not using it
            #discriminator.eval()
            #theta.eval()
            mis_num = 0
            cor_num = 0
            test_num = 0
            with torch.no_grad():
                for data, label in test_loader:
                    test_num += data.shape[0]
                    pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
                    entropy_dis += entropy(pred)
                    r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                    #r_target += torch.FloatTensor(enlarge_id).to(DEVICE)
                    target_out = theta(data, None, r_target).detach()
                    prediction_t = F.softmax(target_out, dim=1)
                    entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                    test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                    test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
                    mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    mis_pred = prediction_t[mis_idx]
                    cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    cor_pred = prediction_t[cor_idx]
                    mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                    mis_num += mis_idx.shape[0]
                    cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                    cor_num += cor_idx.shape[0]
                #print r_target
                print (
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num, \
                                 entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                dis_loss_list.append(dis_loss)
                test_acc_list.append(test_acc)
                writer.add_scalars("train_loss", {"rba": train_loss}, (epoch + 1))
                writer.add_scalars("test_loss", {"rba": test_loss / test_num}, (epoch + 1))
                writer.add_scalars("train_acc", {"rba": train_acc}, (epoch + 1))
                writer.add_scalars("test_acc", {"rba": test_acc / test_num}, (epoch + 1))
                writer.add_scalars("dis_loss", {"rba": dis_loss}, (epoch + 1))
                writer.add_scalars("dis_acc", {"rba": dis_acc}, (epoch + 1))
                writer.add_scalars("ent", {"rba": entropy_clas / test_num}, (epoch + 1))
                writer.add_scalars("mis_ent", {"rba": mis_entropy_clas / test_num}, (epoch + 1))
                writer.add_scalars("dis_ent", {"rba": entropy_dis / test_num}, (epoch + 1))
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    best_dis_loss = dis_loss
                    best_test_loss = test_loss
                    early_stop = 0
                    torch.save(discriminator, "models/dis_office_rba_"+task+".pkl")
                    torch.save(theta.state_dict(), "models/theta_office_param_rba_"+task+".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 10:
            print "Training Process Converges Until Epoch %s" % (epoch + 1)
            break
    writer.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.plot(np.arange(len(test_loss_list)), test_loss_list)
    plt.plot(np.arange(len(dis_loss_list)), dis_loss_list)
    plt.plot(np.arange(len(test_acc_list)), test_acc_list)
    plt.savefig("data/alt_train_plot.png")

def test_model(x_t, y_t, x_t_orig):
    # You need to run the training process first and then plot the graph, we directly load the saved model
    print "\n\nTest model performance"
    N_CLASSES = CONFIG["n_classes"]
    N_FEATURES = x_t.shape[1]
    BATCH_SIZE = CONFIG["batch_size"]

    # RBA
    theta = thetaNet(N_FEATURES, N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_office_param_rba.pkl"))
    discriminator = torch.load("models/dis_office_rba.pkl")
    ce_func = nn.CrossEntropyLoss()
    test_dataset = Data.TensorDataset(x_t, x_t_orig, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    discriminator.eval()
    theta.eval()
    mis_num = 0
    cor_num = 0
    batch_num_test = len(test_loader.dataset)
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    confidence = torch.FloatTensor([1])
    accuracy = torch.FloatTensor([0])
    with torch.no_grad():
        for data, data_orig, label in test_loader:
            pred = F.softmax(discriminator(data_orig, None, None, None, None).detach(), dim=1)
            entropy_dis += entropy(pred)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = theta(data, None, r_target).detach()
            prediction_t = F.softmax(target_out, dim=1)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
            test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
            mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
            batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
            batch_accuracy = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).cpu().float()
            confidence = torch.cat((confidence, batch_confidence), dim=0)
            accuracy = torch.cat((accuracy, batch_accuracy), dim=0)
        print (
            "test_loss:{:.3f}, test_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
            test_loss * 1e3 / batch_num_test, test_acc / batch_num_test, entropy_dis / batch_num_test, \
                         entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas /cor_num
        )

def pretrain_ratio(x_s, y_s, x_t, y_t, task="aw_coral"):
    # Train the network separately, with pretrained density ratios
    # Purpose of demonstrating why we train C and D simultaneously
    # Different from fixed_ratio. Fixed ratio fixes D by not updating through optimizer_dis, this one get rids of loss_dis
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 3000
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-2, betas=(0.99, 0.999), eps=1e-4,
                                       weight_decay=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-4,
                                     weight_decay=1e-8)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    n_train = x_s.shape[0]
    n_test = x_t.shape[0]
    ce_func = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction="mean")
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    early_stop = 0
    best_train_loss = 1e8
    dis_converge = False
    best_dis_loss = 1e8
    best_test_loss = 1e8
    writer = SummaryWriter(LOGDIR)
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat(
        (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print "Originally %d source data, %d target data" % (n_train, n_test)
    train_loss_list = []
    test_loss_list = []
    dis_loss_list = []
    test_acc_list = []
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        train_sample_order = np.arange(n_train)
        test_sample_order = np.arange(n_test)
        # np.random.shuffle(train_sample_order)
        # np.random.shuffle(test_sample_order)
        convert_data_idx_s = torch.eq(whole_label_dis[0:n_train, 0], CONFIG["interval_prob"][0]).nonzero().view(
            -1, ).cpu().numpy()
        remain_data_idx_t = torch.eq(whole_label_dis[n_train:n_train + n_test, 1], 1).nonzero().view(-1, ).cpu().numpy()
        if (epoch + 1) % OUT_ITER == 0:
            interval_s = convert_data_idx_s.shape[0]
            remain_target = remain_data_idx_t.shape[0]
            print "Currently %d removed source data, %d remained target data, %d interval source data, %d interval target data" % (
                n_train - interval_s, remain_target, interval_s, n_test - remain_target
            )
        batch_num_train = max(n_train, n_test) / BATCH_SIZE + 1
        for step in range(batch_num_train):
            if convert_data_idx_s.shape[0] < BATCH_SIZE:
                batch_id_s = np.random.choice(train_sample_order, BATCH_SIZE, replace=False)
            else:
                batch_id_s = np.random.choice(convert_data_idx_s, BATCH_SIZE, replace=False)
            if remain_data_idx_t.shape[0] < BATCH_SIZE:
                batch_id_t = np.random.choice(test_sample_order, BATCH_SIZE, replace=False)
            else:
                batch_id_t = np.random.choice(remain_data_idx_t, BATCH_SIZE, replace=False)
            batch_id_t = batch_id_t + n_train
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_t = whole_data[batch_id_t]
            batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
            batch_y = torch.cat((whole_label_dis[batch_id_s], whole_label_dis[batch_id_t]), dim=0)
            batch_y = batch_y.to(DEVICE)
            shuffle_idx = np.arange(2 * BATCH_SIZE)

            # Feed Forward
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = bce_loss(F.softmax(prob, dim=1), batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source, pos_target = np.zeros((BATCH_SIZE,)), np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx == idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            for idx in range(BATCH_SIZE, 2 * BATCH_SIZE):
                pos_target[idx - BATCH_SIZE] = np.where(shuffle_idx == idx)[0][0]
            r_target = r[pos_target].reshape(-1, 1)
            p_t_target = p_t[pos_target]
            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)
            nn_out = theta(batch_x_t, None, r_target.detach())
            pred_target = F.softmax(nn_out, dim=1)
            prob_grad_r = discriminator(batch_x_t, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                        sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))
            loss_theta = torch.sum(theta_out)

            # Backpropagate
            if not dis_converge:
                if (step + 1) % 1 == 0:
                    optimizer_dis.zero_grad()
                    loss_dis.backward(retain_graph=True)
                    optimizer_dis.step()

            if dis_converge:
                if (step + 1) % 1 == 0:
                    optimizer_dis.zero_grad()
                    loss_r.backward(retain_graph=True)
                    optimizer_dis.step()

                if (step + 1) % 1 == 0:
                    optimizer_theta.zero_grad()
                    loss_theta.backward()
                    optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())
            # dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == torch.argmax(batch_y, dim=1)).float() / (2 * BATCH_SIZE)
            writer.add_scalars("rba_r",
                               {"source": torch.mean(r_source.detach()), "target": torch.mean(r_target.detach())},
                               (epoch + 1))

        ## Change source to interval section, and only use the changed ones for training
        if (epoch + 1) % 15 == 0:
            whole_label_dis = torch.cat(
                (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
            pred_tmp = F.softmax(discriminator(whole_data, None, None, None, None).detach(), dim=1)
            r = (pred_tmp[:, 0] / pred_tmp[:, 1]).reshape(-1, 1)
            # print r[:10]
            # print r[n_train:10+n_train]
            pos_source = np.arange(n_train)
            source_ratio = r[pos_source].view(-1, ).cpu().numpy()
            num_convert = int(source_ratio.shape[0] * 0.5)
            int_convert = heapq.nsmallest(num_convert, range(len(source_ratio)), source_ratio.take)
            invert_idx = pos_source[int_convert].astype(np.int32)
            whole_label_dis[invert_idx] = CONFIG["interval_prob"]

            pos_target = np.arange(n_train, n_train + n_test)
            target_ratio = r[pos_target].view(-1, ).cpu().numpy()
            num_convert = int(target_ratio.shape[0] * 0.0)
            int_convert = heapq.nlargest(num_convert, range(len(target_ratio)), target_ratio.take)
            invert_idx = pos_target[int_convert].astype(np.int32)
            whole_label_dis[invert_idx] = CONFIG["interval_prob"]

        # r_list.append(r[-10])
        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            # Do not do eval(), fucking bug; Probably can be solved by normalizing input data
            # and since we do not have batch_norm, there should be no influence with not using it
            # discriminator.eval()
            # theta.eval()
            mis_num = 0
            cor_num = 0
            test_num = 0
            with torch.no_grad():
                for data, label in test_loader:
                    test_num += data.shape[0]
                    pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
                    entropy_dis += entropy(pred)
                    r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                    # r_target += torch.FloatTensor(enlarge_id).to(DEVICE)
                    target_out = theta(data, None, r_target).detach()
                    prediction_t = F.softmax(target_out, dim=1)
                    entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                    test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                    test_acc += torch.sum(torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).float()
                    mis_idx = (torch.argmax(prediction_t, dim=1) != torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    mis_pred = prediction_t[mis_idx]
                    cor_idx = (torch.argmax(prediction_t, dim=1) == torch.argmax(label, dim=1)).nonzero().reshape(-1, )
                    cor_pred = prediction_t[cor_idx]
                    mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                    mis_num += mis_idx.shape[0]
                    cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                    cor_num += cor_idx.shape[0]
                # print r_target
                print (
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num, \
                                 entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                dis_loss_list.append(dis_loss)
                test_acc_list.append(test_acc)
                writer.add_scalars("train_loss", {"rba": train_loss}, (epoch + 1))
                writer.add_scalars("test_loss", {"rba": test_loss / test_num}, (epoch + 1))
                writer.add_scalars("train_acc", {"rba": train_acc}, (epoch + 1))
                writer.add_scalars("test_acc", {"rba": test_acc / test_num}, (epoch + 1))
                writer.add_scalars("dis_loss", {"rba": dis_loss}, (epoch + 1))
                writer.add_scalars("dis_acc", {"rba": dis_acc}, (epoch + 1))
                writer.add_scalars("ent", {"rba": entropy_clas / test_num}, (epoch + 1))
                writer.add_scalars("mis_ent", {"rba": mis_entropy_clas / test_num}, (epoch + 1))
                writer.add_scalars("dis_ent", {"rba": entropy_dis / test_num}, (epoch + 1))
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    if dis_loss < best_dis_loss:
                        best_dis_loss = dis_loss
                    else:
                        dis_converge = True
                    best_test_loss = test_loss
                    early_stop = 0
                    torch.save(discriminator, "models/fr_dis_office_rba_" + task + ".pkl")
                    torch.save(theta.state_dict(), "models/fr_theta_office_param_rba_" + task + ".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 10:
            print "Training Process Converges Until Epoch %s, Reaches Best At %s" % ((epoch + 1), (epoch+1-100))
            break
    writer.close()

def brier_score():
    tasks = [("amazon", "webcam"), ("amazon", "dslr"), ("webcam", "amazon"), ("webcam", "dslr"), ("dslr", "amazon"), ("dslr", "webcam")]
    ### DeepCoral
    for t in tasks:
        source, target = t[0], t[1]
        print ("Task:", source, " to ", target)
        target_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
        target_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
        enc = OneHotEncoder(categories="auto")
        target_y = target_y.reshape(-1, 1)
        target_y = enc.fit_transform(target_y).toarray()
        target_y = torch.tensor(target_y).to(torch.float32).to(DEVICE)
        target_x = target_x.to(DEVICE)
        test_dataset = Data.TensorDataset(target_x, target_y)
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=64,
            shuffle=True,
        )
        task_name = source[0]+target[0]+"_deepcoral"
        theta = thetaNet(2048, 31)
        theta.load_state_dict(torch.load("models/dis_office_rba_"+task_name+".pkl"))
        discriminator = torch.load("models/dis_office_rba_"+task_name+".pkl")
        score = 0
        with torch.no_grad():
            for data, label in test_loader:
                pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
                r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                target_out = theta(data, None, r_target).detach()
                prediction_t = F.softmax(target_out, dim=1)
                label = label.cpu().numpy()
                prediction_t = prediction_t.cpu().numpy()
                batch_score = brier_score_loss(label, prediction_t)
                score += batch_score
        score /= len(test_loader.dataset)
        print (score)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print 'Using device:', DEVICE
    torch.manual_seed(200)

    """
    brier_score()
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="dslr", help="Source domain")
    parser.add_argument("-t", "--target", default="webcam", help="Target domain")
    parser.add_argument("-d", "--dataset", default="deepcoral", help="Dataset choice")
    args = parser.parse_args()
    source = args.source
    target = args.target
    dataset = args.dataset
    print "Source distribution: %s; Target distribution: %s, Dataset: %s" % (source, target, dataset)
    task_name = source[0]+target[0]
    if dataset == "tca":
        source_data = torch.FloatTensor(np.loadtxt("data/office31_resnet50/" + source + "_" + source + ".csv", delimiter=",").astype("float32"))
        target_data = torch.FloatTensor(np.loadtxt("data/office31_resnet50/" + source + "_" + target + ".csv", delimiter=",").astype("float32"))
        source_x, source_y = torch.load("aligned_data/tca/"+source[0]+target[0]+"_"+source+"_"+source+".pkl"), source_data[:, -1]
        target_x, target_y = torch.load("aligned_data/tca/"+source[0]+target[0]+"_"+source+"_"+target+".pkl"), target_data[:, -1]
        original_x_s = source_data[:, :-1].to(DEVICE)
        original_x_t = target_data[:, :-1].to(DEVICE)
        task_name += "_tca"
    elif dataset =="deepcoral":
        source_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceAligned.pkl")
        source_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceY.pkl")
        target_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
        target_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")
        original_x_s = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceOriginal.pkl")
        original_x_t = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetOriginal.pkl")
        task_name += "_coral"
    enc = OneHotEncoder(categories="auto")
    source_y, target_y = source_y.reshape(-1, 1), target_y.reshape(-1, 1)
    source_y = enc.fit_transform(source_y).toarray()
    target_y = enc.fit_transform(target_y).toarray()
    source_y = torch.tensor(source_y).to(torch.float32)
    target_y = torch.tensor(target_y).to(torch.float32).to(DEVICE)
    source_x = source_x.to(DEVICE)
    target_x = target_x.to(DEVICE)
    """
    #print "\n\nStart Training Aligned+Original RBA"
    #train_aligned(source_x, source_y, target_x, target_y, source_x, target_x)
    #print "\n\n"
    #train_aligned(original_x_s, source_y, original_x_t, target_y, original_x_s, original_x_t)
    #plot_2d(source_x, target_x, "data/dis_plot.png")
    #plot_2d(original_x_s, original_x_t, "data/dis_plot_orig.png")
    #plot_2d(source_x, target_x, "data/dis_plot_dc.png")
    #plot_2d(original_x_s, original_x_t, "data/dis_plot_orig_dc.png")
    #print "\n\nTraining Fixed Discrimiantor (After Few Epoches)"
    #early_fix(source_x, source_y, target_x, target_y)
    #print "\nStart Training IID"
    #iid_baseline(source_x, source_y, target_x, target_y, task_name)
    #print "\nStart Training IW"
    #train_iw(source_x, source_y, target_x, target_y, task_name)
    #print "\nTraining with soft labels"
    #softlabels(source_x, source_y, target_x, target_y, task_name)
    #confidence_accuracy_plot(target_x, target_y, target_x, task_name)
