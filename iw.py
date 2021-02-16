"""
Experiment on different datasets with Importance Weighting method
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
from torchvision import transforms, datasets

torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    "lr1":1e-4,
    "lr2": 1e-4,
    "wd1": 1e-7,
    "wd2": 1e-7,
    "max_iter": 5000,
    "out_iter": 10,
    "n_classes": 65,
    "batch_size": 64,
    "upper_threshold": 1.2,
    "lower_threshold": 0.83,
}

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

def entropy(p):
    p[p<1e-20] = 1e-20 # Deal with numerical issues
    return -torch.sum(p.mul(torch.log2(p)))

def dataloader(root_path, dir, batch_size, train, kwargs):
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

def train_iw_alter(source, target, task_name):
    BATCH_SIZE = 32
    MAX_ITER = 5
    OUT_ITER = 5
    N_CLASSES = CONFIG["n_classes"]
    discriminator = Discriminator_IW()
    theta = IWNet(N_CLASSES)
    # optimizer_theta = torch.optim.Adagrad(theta.parameters(), lr=CONFIG["lr1"], lr_decay=1e-7, weight_decay=CONFIG["wd1"])
    # optimizer_dis = torch.optim.Adagrad(discriminator.parameters(), lr=CONFIG["lr2"], lr_decay=1e-7, weight_decay=CONFIG["wd2"])
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=CONFIG["lr1"], betas=(0.99, 0.999), eps=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr2"], betas=(0.99, 0.999), eps=1e-6)
    train_loader = dataloader("OfficeHome/", source, 32, True, None)
    test_tmp_loader = dataloader("OfficeHome/", target, 32, True, None)
    n_train = len(train_loader.dataset)
    n_test = len(test_tmp_loader.dataset)
    ce_func = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction="mean")
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    early_stop = 0
    best_train_loss = 1e8
    x_s = torch.zeros((1, 3, 224, 224)).cuda()
    y_s = torch.zeros((1)).cuda().long()
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()
        x_s = torch.cat([x_s, img], dim=0)
        y_s = torch.cat([y_s, label], dim=0)
    x_s = x_s[1:]
    y_s = y_s[1:]
    x_t = torch.zeros((1, 3, 224, 224)).cuda()
    y_t = torch.zeros((1)).cuda().long()
    for img, label in test_tmp_loader:
        img = img.cuda()
        label = label.cuda()
        x_t = torch.cat([x_t, img], dim=0)
        y_t = torch.cat([y_t, label], dim=0)
    x_t = x_t[1:]
    y_t = y_t[1:]
    enc = OneHotEncoder(categories="auto")
    y_s, y_t = y_s.cpu().numpy().reshape(-1, 1), y_t.cpu().numpy().reshape(-1, 1)
    y_s = enc.fit_transform(y_s).toarray()
    y_t = enc.fit_transform(y_t).toarray()
    y_s = torch.tensor(y_s).to(torch.float32).to(DEVICE)
    y_t = torch.tensor(y_t).to(torch.float32).to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    x_s = x_s.to(DEVICE)
    x_t = x_t.to(DEVICE)
    y_s = y_s.to(DEVICE)
    ce_func = nn.CrossEntropyLoss()
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    early_stop = 0
    best_train_loss = 1e8
    batch_num_train = max(n_train, n_test) / BATCH_SIZE + 1
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        for step in range(batch_num_train):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_t = x_t[batch_id_t]
            batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_y = batch_y[shuffle_idx]
            prob = discriminator(batch_x)
            loss_dis = ce_func(prob, batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source = np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx == idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)

            prob_grad_r = discriminator(batch_x_t)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))

            if (step + 1) % 1 == 0:
                optimizer_dis.zero_grad()
                loss_r.backward(retain_graph=True)
                optimizer_dis.step()

            if (step + 1) % 1 == 0:
                optimizer_dis.zero_grad()
                loss_dis.backward()
                optimizer_dis.step()

            if (step + 1) % 1 == 0:
                loss_theta = torch.sum(theta_out)
                optimizer_theta.zero_grad()
                loss_theta.backward()
                optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())
            dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == batch_y).float() / (2 * BATCH_SIZE)

        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            discriminator.eval()
            theta.eval()
            mis_num = 0
            cor_num = 0
            batch_num_test = len(test_loader.dataset)
            with torch.no_grad():
                for data, label in test_loader:
                    pred = F.softmax(discriminator(data).detach(), dim=1)
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
                                 entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0
                    torch.save(discriminator, "models/dis_iw_alter_" + task_name + ".pkl")
                    torch.save(theta.state_dict(), "models/theta_iw_alter_" + task_name + ".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 20:
            print "Training Process Converges At Epoch %s, reaches best at Epoch %s" % ((epoch + 1), (epoch + 1 - 200))
            break

def train_iw_fixed(source, target, task_name):
    BATCH_SIZE = 32
    MAX_ITER = 25
    OUT_ITER = 5
    N_CLASSES = CONFIG["n_classes"]
    discriminator = Discriminator_IW()
    theta = IWNet(N_CLASSES)
    # optimizer_theta = torch.optim.Adagrad(theta.parameters(), lr=CONFIG["lr1"], lr_decay=1e-7, weight_decay=CONFIG["wd1"])
    # optimizer_dis = torch.optim.Adagrad(discriminator.parameters(), lr=CONFIG["lr2"], lr_decay=1e-7, weight_decay=CONFIG["wd2"])
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=CONFIG["lr1"], betas=(0.99, 0.999), eps=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr2"], betas=(0.99, 0.999), eps=1e-6)
    train_loader = dataloader("OfficeHome/", source, 32, True, None)
    test_tmp_loader = dataloader("OfficeHome/", target, 32, True, None)
    n_train = len(train_loader.dataset)
    n_test = len(test_tmp_loader.dataset)
    ce_func = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction="mean")
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    early_stop = 0
    best_train_loss = 1e8
    x_s = torch.zeros((1, 3, 224, 224)).cuda()
    y_s = torch.zeros((1)).cuda().long()
    for img, label in train_loader:
        img = img.cuda()
        label = label.cuda()
        x_s = torch.cat([x_s, img], dim=0)
        y_s = torch.cat([y_s, label], dim=0)
    x_s = x_s[1:]
    y_s = y_s[1:]
    x_t = torch.zeros((1, 3, 224, 224)).cuda()
    y_t = torch.zeros((1)).cuda().long()
    for img, label in test_tmp_loader:
        img = img.cuda()
        label = label.cuda()
        x_t = torch.cat([x_t, img], dim=0)
        y_t = torch.cat([y_t, label], dim=0)
    x_t = x_t[1:]
    y_t = y_t[1:]
    enc = OneHotEncoder(categories="auto")
    y_s, y_t = y_s.cpu().numpy().reshape(-1, 1), y_t.cpu().numpy().reshape(-1, 1)
    y_s = enc.fit_transform(y_s).toarray()
    y_t = enc.fit_transform(y_t).toarray()
    y_s = torch.tensor(y_s).to(torch.float32).to(DEVICE)
    y_t = torch.tensor(y_t).to(torch.float32).to(DEVICE)
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    x_s = x_s.to(DEVICE)
    x_t = x_t.to(DEVICE)
    y_s = y_s.to(DEVICE)
    ce_func = nn.CrossEntropyLoss()
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    early_stop = 0
    best_train_loss = 1e8
    early_fix_point = -1
    batch_num_train = max(n_train, n_test) / BATCH_SIZE + 1
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        early_fix_point += 1
        for step in range(batch_num_train):
            batch_id_s = np.random.choice(np.arange(n_train), BATCH_SIZE, replace=False)
            batch_id_t = np.random.choice(np.arange(n_test), BATCH_SIZE, replace=False)
            batch_x_s = x_s[batch_id_s]
            batch_y_s = y_s[batch_id_s]
            batch_x_t = x_t[batch_id_t]
            batch_x = torch.cat((batch_x_s, batch_x_t), dim=0)
            batch_y = torch.cat((torch.zeros(BATCH_SIZE, ), torch.ones(BATCH_SIZE, )), dim=0).long()
            shuffle_idx = np.arange(2 * BATCH_SIZE)
            np.random.shuffle(shuffle_idx)
            batch_x = batch_x[shuffle_idx]
            batch_y = batch_y[shuffle_idx]
            prob = discriminator(batch_x)
            loss_dis = ce_func(prob, batch_y)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            pos_source = np.zeros((BATCH_SIZE,))
            for idx in range(BATCH_SIZE):
                pos_source[idx] = np.where(shuffle_idx == idx)[0][0]
            r_source = r[pos_source].reshape(-1, 1)
            theta_out = theta(batch_x_s, batch_y_s, r_source.detach())
            source_pred = F.softmax(theta_out, dim=1)

            prob_grad_r = discriminator(batch_x_t)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))

            if early_fix_point < 20:
                optimizer_dis.zero_grad()
                loss_r.backward(retain_graph=True)
                optimizer_dis.step()

                optimizer_dis.zero_grad()
                loss_dis.backward()
                optimizer_dis.step()
            else:
                loss_theta = torch.sum(theta_out)
                optimizer_theta.zero_grad()
                loss_theta.backward()
                optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())
            dis_acc += torch.sum(torch.argmax(prediction.detach(), dim=1) == batch_y).float() / (2 * BATCH_SIZE)

        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            discriminator.eval()
            theta.eval()
            mis_num = 0
            cor_num = 0
            batch_num_test = len(test_loader.dataset)
            with torch.no_grad():
                for data, label in test_loader:
                    pred = F.softmax(discriminator(data).detach(), dim=1)
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
                                 entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0
                    if early_fix_point < 20:
                        torch.save(discriminator, "models/dis_iw_fixed_" + task_name + ".pkl")
                    torch.save(theta.state_dict(), "models/theta_iw_fixed_" + task_name + ".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 20:
            print "Training Process Converges At Epoch %s, reaches best at Epoch %s" % ((epoch + 1), (epoch + 1 - 200))
            break


if __name__=="__main__":
    print ('Using device:', DEVICE)
    torch.manual_seed(200)
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    for source in domains:
        for target in domains:
            if source != target:
                print ("Source distribution: %s; Target distribution: %s" % (source, target))
                task_name = source[0] + target[0]
                train_iw_alter(source, target, task_name)



