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
import heapq
from torchvision import transforms, datasets
from sklearn.metrics import brier_score_loss

torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_end2end_alter(source, target, task):
    BATCH_SIZE = 16
    MAX_ITER = 200
    OUT_ITER = 5
    N_CLASSES = 65
    discriminator = Discriminator_e2e()
    theta = thetaNet_e2e(N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=CONFIG["lr1"], betas=(0.99, 0.999), eps=1e-4,
                                       weight_decay=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr2"], betas=(0.99, 0.999), eps=1e-4,
                                     weight_decay=1e-8)
    train_loader = dataloader("OfficeHome/", source, 32, True, None)
    test_tmp_loader = dataloader("OfficeHome/", target, 32, True, None)
    n_train = len(train_loader.dataset)
    n_test = len(test_tmp_loader.dataset)
    ce_func = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction="mean")
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
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
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat(
        (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print ("Originally %d source data, %d target data" % (n_train, n_test))
    train_loss_list = []
    test_loss_list = []
    dis_loss_list = []
    test_acc_list = []
    for epoch in range(MAX_ITER):
        discriminator.train()
        theta.train()
        train_sample_order = np.arange(n_train)
        test_sample_order = np.arange(n_test)
        convert_data_idx_s = torch.eq(whole_label_dis[0:n_train, 0], CONFIG["interval_prob"][0]).nonzero().view(
            -1, ).cpu().numpy()
        remain_data_idx_t = torch.eq(whole_label_dis[n_train:n_train + n_test, 1], 1).nonzero().view(-1, ).cpu().numpy()
        if (epoch + 1) % OUT_ITER == 0:
            interval_s = convert_data_idx_s.shape[0]
            remain_target = remain_data_idx_t.shape[0]
            print ("Currently %d removed source data, %d remained target data, %d interval source data, %d interval target data" % (
                n_train - interval_s, remain_target, interval_s, n_test - remain_target
            ))
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
            shuffle_idx = np.arange(2 * BATCH_SIZE)

            # Feed Forward
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = bce_loss(F.softmax(prob, dim=1), batch_y.cuda())
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
            if (step + 1) % 1 == 0:
                optimizer_dis.zero_grad()
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()

            if (step + 1) % 5 == 0:
                optimizer_dis.zero_grad()
                loss_r.backward(retain_graph=True)
                optimizer_dis.step()

            if (step + 1) % 5 == 0:
                optimizer_theta.zero_grad()
                loss_theta.backward()
                optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())

        ## Change source to interval section, and only use the changed ones for training
        if (epoch + 1) % 15 == 0:
            whole_label_dis = torch.cat(
                (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
            pred_tmp = torch.zeros((1, 2))
            for i in range(whole_data.shape[0]):
                pred_tmp = torch.cat((pred_tmp, F.softmax(discriminator(whole_data[i].reshape(1, 3, 224, 224), None, None, None, None).detach(), dim=1)), dim=0)
            pred_tmp = pred_tmp[1:]
            r = (pred_tmp[:, 0] / pred_tmp[:, 1]).reshape(-1, 1)
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
            mis_num = 0
            cor_num = 0
            test_num = 0
            b_score = 0
            discriminator.eval()
            theta.eval()
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
                    for j in range(data.shape[0]):
                        b_score += brier_score_loss(label[j].cpu().numpy(), prediction_t[j].cpu().numpy())
                print("Brier score: ", b_score / test_num)
                # print r_target
                print ((
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num, \
                                 entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                ))
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                dis_loss_list.append(dis_loss)
                test_acc_list.append(test_acc)
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 3:
            print ("Training Process Converges Until Epoch %s" % (epoch + 1))
            break
    #torch.save(discriminator, "models/dis_rba_alter_" + task + ".pkl")
    #torch.save(theta.state_dict(), "models/theta_rba_alter_" + task + ".pkl")

def train_end2end_fixed(source, target, task):
    BATCH_SIZE = 8
    MAX_ITER = 20
    #MAX_ITER = 25
    OUT_ITER = 5
    N_CLASSES = 65
    discriminator = Discriminator_e2e()
    theta = thetaNet_e2e(N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=CONFIG["lr1"], betas=(0.99, 0.999), eps=1e-4,
                                       weight_decay=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr2"], betas=(0.99, 0.999), eps=1e-4,
                                     weight_decay=1e-8)
    train_loader = dataloader("OfficeHome/", source, 32, True, None)
    test_tmp_loader = dataloader("OfficeHome/", target, 32, True, None)
    n_train = len(train_loader.dataset)
    n_test = len(test_tmp_loader.dataset)
    ce_func = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction="mean")
    train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
    cor_entropy_clas = 0
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
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
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat(
        (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print ("Originally %d source data, %d target data" % (n_train, n_test))
    early_fix_point = -1
    for epoch in range(MAX_ITER):
        torch.manual_seed(200)
        torch.cuda.manual_seed(200)
        discriminator.train()
        theta.train()
        early_fix_point += 1
        train_sample_order = np.arange(n_train)
        test_sample_order = np.arange(n_test)
        convert_data_idx_s = torch.eq(whole_label_dis[0:n_train, 0], CONFIG["interval_prob"][0]).nonzero().view(
            -1, ).cpu().numpy()
        remain_data_idx_t = torch.eq(whole_label_dis[n_train:n_train + n_test, 1], 1).nonzero().view(-1, ).cpu().numpy()
        if (epoch + 1) % OUT_ITER == 0:
            interval_s = convert_data_idx_s.shape[0]
            remain_target = remain_data_idx_t.shape[0]
            print ("Currently %d removed source data, %d remained target data, %d interval source data, %d interval target data" % (
                n_train - interval_s, remain_target, interval_s, n_test - remain_target
            ))
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
            shuffle_idx = np.arange(2 * BATCH_SIZE)

            # Feed Forward
            prob = discriminator(batch_x, None, None, None, None)
            loss_dis = bce_loss(F.softmax(prob, dim=1), batch_y.cuda())
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
            prob_grad_r = discriminator(batch_x_t, nn_out.detach(), pred_target.detach(), p_t_target.detach(), sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))
            loss_theta = torch.sum(theta_out)

            # Backpropagate
            if early_fix_point < 10:
                if (step + 1) % 1 == 0:
                    optimizer_dis.zero_grad()
                    loss_dis.backward(retain_graph=True)
                    optimizer_dis.step()

                if (step + 1) % 1 == 0:
                    optimizer_dis.zero_grad()
                    loss_r.backward(retain_graph=True)
                    optimizer_dis.step()
            else:
                if (step + 1) % 1 == 0:
                    optimizer_theta.zero_grad()
                    loss_theta.backward()
                    optimizer_theta.step()

            train_loss += float(ce_func(theta_out.detach(), torch.argmax(batch_y_s, dim=1)))
            train_acc += torch.sum(
                torch.argmax(source_pred.detach(), dim=1) == torch.argmax(batch_y_s, dim=1)).float() / BATCH_SIZE
            dis_loss += float(loss_dis.detach())

        ## Change source to interval section, and only use the changed ones for training
        if (epoch + 1) % 15 == 0:
            whole_label_dis = torch.cat(
                (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
            pred_tmp = torch.zeros((1, 2))
            for i in range(whole_data.shape[0]):
                pred_tmp = torch.cat((pred_tmp, F.softmax(
                    discriminator(whole_data[i].reshape(1, 3, 224, 224), None, None, None, None).detach(), dim=1)),
                                     dim=0)
            pred_tmp = pred_tmp[1:]
            r = (pred_tmp[:, 0] / pred_tmp[:, 1]).reshape(-1, 1)
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

        if (epoch + 1) % OUT_ITER == 0:
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)
            mis_num = 0
            cor_num = 0
            test_num = 0
            b_score = 0
            theta.eval()
            discriminator.eval()
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
                    for j in range(data.shape[0]):
                        b_score += brier_score_loss(label[j].cpu().numpy(), prediction_t[j].cpu().numpy())
                print("Brier score: ", b_score / test_num)
                # print r_target
                print ((
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num,
                    entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                ))
            train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cor_entropy_clas = 0

        if early_stop > 3:
            print ("Training Process Converges Until Epoch %s" % (epoch + 1))
            break
    torch.save(discriminator, "models/dis_rba_fixed_" + task + ".pkl")
    torch.save(theta.state_dict(), "models/theta_rba_fixed_" + task + ".pkl")
    return theta, discriminator


def load_model(source, target, task):
    print("Test saved model")
    #test_loader = dataloader("office/", target, 32, True, None)
    test_tmp_loader = dataloader("office/", target, 32, False, None)
    x_t = torch.zeros((1, 3, 224, 224)).cuda()
    y_t = torch.zeros((1)).cuda().long()
    for img, label in test_tmp_loader:
        img = img.cuda()
        label = label.cuda()
        x_t = torch.cat([x_t, img], dim=0)
        y_t = torch.cat([y_t, label], dim=0)
    x_t = x_t[1:]
    y_t = y_t[1:]
    test_dataset = Data.TensorDataset(x_t, y_t)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=True,
    )
    ce_func = nn.CrossEntropyLoss()
    N_CLASSES = 31
    # End2end with fixed r
    theta = thetaNet_e2e(N_CLASSES)
    theta.eval()
    theta.load_state_dict(torch.load("models/theta_rba_fixed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_fixed_" + task + ".pkl")
    discriminator.eval()
    torch.manual_seed(200)
    torch.cuda.manual_seed(200)
    for name, param in theta.named_parameters():
        if name == "extractor.0.weight":
            print(name, param)
    for name, param in discriminator.named_parameters():
        if name == "net.0.weight":
            print(name, param)

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
        print(test_num, mis_num)
        print (("End2end training with fixed R: test_loss:%.3f, test_acc: %.4f, ent_dis: %.3f, ent_clas: %.3f, mis_ent_clas: %.3f, cor_ent: %.3f") %
               (test_loss * 1e3 / test_num, test_acc / test_num, entropy_dis / test_num, entropy_clas / test_num,
                mis_entropy_clas / mis_num, cor_entropy_clas / cor_num))

if __name__=="__main__":
    print ('Using device:', DEVICE)
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    for source in domains:
        for target in domains:
            if source != target:
                print ("Source distribution: %s; Target distribution: %s" % (source, target))
                task_name = source[0]+target[0]
                train_end2end_fixed(source, target, task_name)
   
