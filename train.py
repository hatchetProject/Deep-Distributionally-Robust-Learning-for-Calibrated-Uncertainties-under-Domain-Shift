"""
Do training and testing on the Office-Home dataset
Train TCA features with 1 layer FCN, DeepCORAL with more layers
"""

import numpy as np
import torch
import math
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import torch.utils.data as Data
from model_layers import ClassifierLayer, RatioEstimationLayer, Flatten, GradLayer, IWLayer
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import heapq
import torchvision

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Discriminator(nn.Module):
    """
    Defines D network
    """
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

class iid_theta(nn.Module):
    def __init__(self, n_features, n_output):
        super(iid_theta, self).__init__()
        self.net = torch.nn.Sequential(
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
            #nn.Linear(512, 256),
            #nn.Tanh(),
            #torch.nn.Linear(256, 64),
            torch.nn.Linear(512, n_output),
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
            #nn.Linear(512, 256),
            #nn.Tanh(),
            #torch.nn.Linear(1024, 64),
        )
        self.IW = IWLayer(512, n_output)

     def forward(self, x_s, y_s, r):
         x_s = self.extractor(x_s)
         x = self.IW(x_s, y_s, r)
         return x

def entropy(p):
    p[p<1e-20] = 1e-20
    return -torch.sum(p.mul(torch.log2(p)))

CONFIG = {
    "lr1": 1e-3,
    "lr2": 1e-4,
    "wd1": 1e-7,
    "wd2": 1e-7,
    "max_iter": 150,
    "out_iter": 10,
    "n_classes": 65,
    "batch_size": 64,
    "upper_threshold": 1.5,
    "lower_threshold": 0.67,
    "source_prob": torch.FloatTensor([1., 0.]),
    "interval_prob": torch.FloatTensor([0.5, 0.5]),
    "target_prob": torch.FloatTensor([0., 1.]),
}

LOGDIR = os.path.join("runs", datetime.now().strftime("%Y%m%d%H%M%S"))


def softlabels(x_s, y_s, x_t, y_t, task):
    ## RBA training
    ## Changes the hard labels of the original dataset to soft ones (probabilities), such as (0.5, 0.5) for samples with large density ratio in the target domain
    ## Trained with adversarial principle
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 300
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-8,
                                       weight_decay=0)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8,
                                     weight_decay=0)
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
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat(
        (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print "Originally %d source data, %d target data" % (n_train, n_test)
    train_loss_list = []
    test_loss_list = []
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
            # Separate source sample density ratios from target sample density ratios
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
            pred_tmp = F.softmax(discriminator(whole_data, None, None, None, None).detach(), dim=1)
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
            # Test current model for every OUT_ITER epochs, save the model as well
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)

            discriminator.eval()
            theta.eval()
            mis_num = 0
            cor_num = 0
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
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num, entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                train_loss_list.append(train_loss * 1e3)
                test_loss_list.append(test_loss *1e3 / test_num)
                test_acc_list.append(test_acc.cpu().numpy() / test_num)
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0
                #torch.save(discriminator, "models/dis_rba_alter_aligned_" + task + ".pkl")
                #torch.save(theta.state_dict(), "models/theta_rba_alter_aligned_" + task + ".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0
    print (train_loss_list)
        #if early_stop > 5:
        #    print "Training Process Converges Until Epoch %s" % (epoch + 1)
        #    break

def no_softlabel(x_s, y_s, x_t, y_t, task):
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 300
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-8,
                                       weight_decay=0)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8,
                                     weight_decay=0)
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
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat(
        (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print "Originally %d source data, %d target data" % (n_train, n_test)
    train_loss_list = []
    test_loss_list = []
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
            # Separate source sample density ratios from target sample density ratios
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
        if (epoch + 1) % 1000 == 0:
            whole_label_dis = torch.cat(
                (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
            pred_tmp = F.softmax(discriminator(whole_data, None, None, None, None).detach(), dim=1)
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
            # Test current model for every OUT_ITER epochs, save the model as well
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)

            discriminator.eval()
            theta.eval()
            mis_num = 0
            cor_num = 0
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
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num, entropy_clas / test_num,
                                 mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                train_loss_list.append(train_loss * 1e3)
                test_loss_list.append(test_loss * 1e3 / test_num)
                test_acc_list.append(test_acc.cpu().numpy() / test_num)
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0
                # torch.save(discriminator, "models/dis_rba_alter_aligned_" + task + ".pkl")
                # torch.save(theta.state_dict(), "models/theta_rba_alter_aligned_" + task + ".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0
    print (train_loss_list)
        #if early_stop > 5:
        #    print "Training Process Converges Until Epoch %s" % (epoch + 1)
        #    break

def softlabels_relaxed(x_s, y_s, x_t, y_t, task):
    from sklearn.metrics import brier_score_loss
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 60
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8,
                                       weight_decay=0)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.99, 0.999), eps=1e-8,
                                     weight_decay=0)
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
    whole_data = torch.cat((x_s, x_t), dim=0)
    whole_label_dis = torch.cat(
        (torch.FloatTensor([1, 0]).repeat(n_train, 1), torch.FloatTensor([0, 1]).repeat(n_test, 1)), dim=0)
    print "Originally %d source data, %d target data" % (n_train, n_test)
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
            # Separate source sample density ratios from target sample density ratios
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
            pred_tmp = F.softmax(discriminator(whole_data, None, None, None, None).detach(), dim=1)
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
            # Test current model for every OUT_ITER epochs, save the model as well
            train_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            train_acc /= (OUT_ITER * batch_num_train)
            dis_loss /= (OUT_ITER * batch_num_train * BATCH_SIZE)
            dis_acc /= (OUT_ITER * batch_num_train)

            discriminator.eval()
            theta.eval()
            mis_num = 0
            cor_num = 0
            test_num = 0
            b_score = 0
            with torch.no_grad():
                for data, label in test_loader:
                    test_num += data.shape[0]
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
                    "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                 test_acc / test_num, dis_acc, entropy_dis / test_num, entropy_clas / test_num,
                                 mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                )
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0

                #torch.save(discriminator, "models/dis_rba_alter_aligned_relaxed_" + task + ".pkl")
                #torch.save(theta.state_dict(), "models/theta_rba_alter_aligned_relaxed_" + task + ".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0
        if early_stop > 5:
            print "Training Process Converges Until Epoch %s" % (epoch + 1)
            break

def early_fix(x_s, y_s, x_t, y_t, task_name):
    BATCH_SIZE = 64
    MAX_ITER = 100
    OUT_ITER = 10
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    #optimizer_theta = torch.optim.Adagrad(theta.parameters(), lr=CONFIG["lr1"], lr_decay=1e-7, weight_decay=CONFIG["wd1"])
    #optimizer_dis = torch.optim.Adagrad(discriminator.parameters(), lr=CONFIG["lr2"], lr_decay=1e-7, weight_decay=CONFIG["wd2"])
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-8)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-6)
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
    best_train_loss = 1e8
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
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()

                optimizer_dis.zero_grad()
                loss_r.backward(retain_graph=True)
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
                    "{} epoches: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                    (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / batch_num_test, train_acc,
                                 test_acc / batch_num_test, entropy_dis / batch_num_test, \
                                 entropy_clas / batch_num_test, mis_entropy_clas / mis_num, cor_entropy_clas /cor_num
                )
                if train_loss >= best_train_loss:
                    early_stop += 1
                else:
                    best_train_loss = train_loss
                    early_stop = 0
                    #if early_fix_point < 20:
                    #    torch.save(discriminator, "models/dis_rba_fixed_aligned_"+task_name+".pkl")
                    #torch.save(theta.state_dict(), "models/theta_rba_fixed_aligned_"+task_name+".pkl")
                train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
                cor_entropy_clas = 0

        if early_stop > 10:
            print "Training Process Converges At Epoch %s" % (epoch+1)
            break



if __name__=="__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print 'Using device:', DEVICE
    torch.manual_seed(200)
    # Run different tasks by command: python train_home.py -s XXX -t XXX -d XXX
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-s", "--source", default="RealWorld", help="Source domain")
    #parser.add_argument("-t", "--target", default="Product", help="Target domain")
    #args = parser.parse_args()

    #source = args.source
    #target = args.target
    source = "Product"
    target = "Art"
    task_name = source[0] + target[0]
    print "Source distribution: %s; Target distribution: %s" % (source, target)
    # Load the dataset for relaxed alignment
    source_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceAligned.pkl")
    source_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceY.pkl")
    target_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAligned.pkl")
    target_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetY.pkl")

    # Change the labels into one-hot encoding
    enc = OneHotEncoder(categories="auto")
    source_y, target_y = source_y.reshape(-1, 1), target_y.reshape(-1, 1)
    source_y = enc.fit_transform(source_y).toarray()
    target_y = enc.fit_transform(target_y).toarray()
    source_y = torch.tensor(source_y).to(torch.float32)
    target_y = torch.tensor(target_y).to(torch.float32).cuda()
    source_x, target_x = source_x.cuda(), target_x.cuda()
    #print "Training Fixed Discrimiantor (After Few Epoches)"
    #early_fix(source_x, source_y, target_x, target_y, task_name)
    print("\n\nTrained without softlabels")
    no_softlabel(source_x, source_y, target_x, target_y, task_name)
    print("\n\nTrain with soft labels")
    softlabels(source_x, source_y, target_x, target_y, task_name)

    print ("\n\nTrain soft labels with relaxed alignment")
    source_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceAlignedRelaxed.pkl")
    source_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_sourceYRelaxed.pkl")
    target_x = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetAlignedRelaxed.pkl")
    target_y = torch.load("aligned_data/deepcoral/" + source + "_" + target + "_targetYRelaxed.pkl")
    enc = OneHotEncoder(categories="auto")
    source_y, target_y = source_y.reshape(-1, 1), target_y.reshape(-1, 1)
    source_y = enc.fit_transform(source_y).toarray()
    target_y = enc.fit_transform(target_y).toarray()
    source_y = torch.tensor(source_y).to(torch.float32)
    target_y = torch.tensor(target_y).to(torch.float32).cuda()
    source_x, target_x = source_x.cuda(), target_x.cuda()

    print "\n\nTraining Fixed Discrimiantor (After Few Epoches)"
    early_fix(source_x, source_y, target_x, target_y, task_name)
    print("\n\nTrain with soft labels")
    softlabels(source_x, source_y, target_x, target_y, task_name)
    print ("\n\nTrain soft labels with relaxed alignment")
    softlabels_relaxed(source_x, source_y, target_x, target_y, task_name)


