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

    def forward(self, x_s, y_s, r, p_t = None):
        x_s = self.extractor(x_s)
        x = self.classifier(x_s, y_s, r, p_t)
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
    "n_classes": 31,
    "batch_size": 64,
    "upper_threshold": 1.5,
    "lower_threshold": 0.67,
    "source_prob": torch.FloatTensor([1., 0.]),
    "interval_prob": torch.FloatTensor([0.5, 0.5]),
    "target_prob": torch.FloatTensor([0., 1.]),
}

LOGDIR = os.path.join("runs", datetime.now().strftime("%Y%m%d%H%M%S"))

def avh_score(x, w):
    """
    Actually computes the AVC score for a single sample;
    AVH score is used to replace the prediction probability
    x with shape (1, num_features), w with shape (num_features, n_classes)
    :return: avh score of a single sample, with type float
    """
    avc_score = np.pi - np.arccos(np.dot(x, w.t())/(np.linalg.norm(x)*np.linalg.norm(w)))
    avc_score = avc_score / np.sum(avc_score)
    return avc_score


def iid_baseline(x_s, y_s, x_t, y_t, task="AC_tca"):
    ## IID training
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

def train_iw(x_s, y_s, x_t, y_t, task="AC_tca"):
    ## IW training
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

            train_loss, train_acc, test_loss, test_acc, dis_loss, dis_acc, entropy_dis, entropy_clas, mis_entropy_clas = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cor_entropy_clas = 0
        if early_stop > 3:
            print "Training Process Already Converges At Epoch %s" % (epoch+1)
            break
    writer.close()

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
            print(r)
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

def self_training_softlabels(x_s, y_s, x_t, y_t, task):
    ## RBA self-training
    ## Training with self-training criteria
    BATCH_SIZE = CONFIG["batch_size"]
    MAX_ITER = 40
    OUT_ITER = 5
    N_FEATURES = x_s.shape[1]
    N_CLASSES = y_s.shape[1]
    discriminator = Discriminator(N_FEATURES)
    theta = thetaNet(N_FEATURES, N_CLASSES)
    optimizer_theta = torch.optim.Adam(theta.parameters(), lr=1e-3, betas=(0.99, 0.999), eps=1e-8,
                                       weight_decay=0)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.99, 0.999), eps=1e-8,
                                     weight_decay=0)

    stop_iter = 10
    iter_num = -1
    x_s_clas = x_s # The data for classifier, now we train the classifier and estimator separately, with different x
    y_s_clas = y_s # The label for classifier, corresponding to x_s_clas
    y_t_not_onehot = torch.argmax(y_t, dim=1)
    separate_count = np.zeros(CONFIG["n_classes"])
    for label in range(CONFIG["n_classes"]):
        separate_count[label] = torch.sum(y_t_not_onehot == label)

    while (iter_num < stop_iter):
        print ("\n\n Current number of training examples: ", x_s.shape[0])
        test_dataset = Data.TensorDataset(x_t, y_t)
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
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

        iter_num += 1
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
                p_t_source = p_t[pos_source]

                theta_out = theta(batch_x_s, batch_y_s, r_source.detach(), p_t_source.detach())

                source_pred = F.softmax(theta_out, dim=1)
                nn_out = theta(batch_x_t, None, r_target.detach())

                pred_target = F.softmax(nn_out, dim=1)
                prob_grad_r = discriminator(batch_x_t, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                            sign_variable)
                loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape)))
                loss_theta = torch.sum(theta_out)

                # Backpropagate
                if (step + 1) % 1 == 0 and iter_num == 0:
                    optimizer_dis.zero_grad()
                    loss_dis.backward(retain_graph=True)
                    optimizer_dis.step()

                if (step + 1) % 5 == 0 and iter_num == 0:
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
                batch_count = np.zeros(CONFIG["n_classes"])
                batch_confidence_count = np.zeros(CONFIG["n_classes"])
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

                        # Class-specific accuracy
                        batch_predict = torch.argmax(prediction_t, dim=1)
                        batch_true = torch.argmax(label, dim=1)
                        for j in range(batch_true.shape[0]):
                            if (batch_true[j] == batch_predict[j]):
                                batch_count[batch_true[j]] += 1

                        # Class-specfic confidence
                        batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
                        for j in range(batch_confidence.shape[0]):
                            batch_confidence_count[batch_true[j]] += batch_confidence[j]

                    print (
                        "{} epochs: train_loss: {:.3f}, dis_loss: {:.3f}, test_loss:{:.3f}, train_acc: {:.4f}, test_acc: {:.4f}, dis_acc: {:.4f}, ent_dis: {: .3f}, ent_clas: {: .3f}, mis_ent_clas: {:.3f}, cor_ent: {:.3f}").format(
                        (epoch + 1), train_loss * 1e3, dis_loss * 1e3, test_loss * 1e3 / test_num, train_acc,
                                     test_acc / test_num, dis_acc, entropy_dis / test_num, entropy_clas / test_num,
                                     mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
                    )
                    #print("Class-specific accuracy:", batch_count/separate_count)
                    #print("Class-specific confidence:", batch_confidence_count/separate_count)
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
        # Get the converted data samples from test set
        confidence = torch.FloatTensor([1])
        conf_idx = torch.FloatTensor([1]).long()
        prediction_result = torch.FloatTensor([1, CONFIG["n_classes"]])
        discriminator.eval()
        theta.eval()
        with torch.no_grad():
            for data, label in test_loader:
                pred = F.softmax(discriminator(data, None, None, None, None).detach(), dim=1)
                entropy_dis += entropy(pred)
                r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                target_out = theta(data, None, r_target).detach()
                prediction_t = F.softmax(target_out, dim=1)
                prediction_result = torch.cat((prediction_result, prediction_t), dim=0)
                batch_confidence = torch.max(prediction_t, dim=1).values.cpu()
                confidence = torch.cat((confidence, batch_confidence), dim=0)
                max_idx = torch.argmax(prediction_t, dim=1).cpu()
                conf_idx = torch.cat([conf_idx, max_idx])
        confidence = confidence[1:].numpy()
        conf_idx = conf_idx[1:].numpy()
        prediction_result = prediction_result[1:].numpy()

        # vanilla self-training convert
        idx_conf = np.where(confidence > (0.9))[0]
        y_conv = torch.FloatTensor(np.zeros([conf_idx.shape[0], CONFIG["n_classes"]]))
        for k in range(idx_conf.shape[0]):
            y_conv[k, conf_idx[idx_conf[k]]] = 1
        x_s = torch.cat((x_s_clas, x_t[idx_conf]), dim=0)
        y_s = torch.cat((y_s_clas, y_conv.cuda()), dim=0)

        """
        # class-specific convert: not critically CBST
        p = 0.8  # The only parameter needed to tune for CBST
        class_specific_num = np.zeros(CONFIG["n_classes"])
        for j in range(conf_idx.shape[0]):
            class_specific_num[conf_idx[j]] += 1
        class_specific_convert_num = p*class_specific_num
        convert_all_idx = np.zeros(1)
        for j in range(CONFIG["n_classes"]):
            class_idx = np.where(conf_idx == j)[0]
            conf_class_value = confidence[class_idx]
            class_convert = heapq.nlargest(int(class_specific_convert_num[j]), range(len(conf_class_value)), conf_class_value.take)
            j_class_convert = class_idx[class_convert]
            convert_all_idx = np.concatenate([convert_all_idx, j_class_convert])
        convert_all_idx = convert_all_idx[1:]
        y_conv = torch.FloatTensor(np.zeros([convert_all_idx.shape[0], CONFIG["n_classes"]]))
        for k in range(convert_all_idx.shape[0]):
            y_conv[k, conf_idx[int(convert_all_idx[k])]] = 1
        x_s = torch.cat((x_s_clas, x_t[convert_all_idx]), dim=0)
        y_s = torch.cat((y_s_clas, y_conv.cuda()), dim=0)
        """

        # CBST
        p = 0.8  # The only parameter need to be tuned, the portion of data to be converted
        class_specific_num = np.zeros(CONFIG["n_classes"])
        lambda_k = np.zeros(CONFIG["n_classes"])
        for j in range(conf_idx.shape[0]):
            class_specific_num[conf_idx[j]] += 1
        class_specific_convert_num = p * class_specific_num
        # Get lambda_k and convert sample index
        convert_all_idx = np.zeros(1)
        for j in range(CONFIG["n_classes"]):
            class_idx = np.where(conf_idx == j)[0]
            conf_class_value = confidence[class_idx]
            class_convert = heapq.nlargest(int(class_specific_convert_num[j]), range(len(conf_class_value)),
                                           conf_class_value.take)
            j_class_convert = class_idx[class_convert]
            convert_all_idx = np.concatenate([convert_all_idx, j_class_convert])
            conf_class_tmp = np.sort(conf_class_value)
            lambda_k[j] = conf_class_tmp[-class_specific_convert_num[j]]
        # Get new pseudo labels
        new_prediction_result = prediction_result/lambda_k
        new_conf_idx = np.argmax(new_prediction_result, axis=1)
        # Convert samples from test set to train set
        convert_all_idx = convert_all_idx[1:]
        y_conv = torch.FloatTensor(np.zeros([convert_all_idx.shape[0], CONFIG["n_classes"]]))
        for k in range(convert_all_idx.shape[0]):
            y_conv[k, new_conf_idx[int(convert_all_idx[k])]] = 1
        x_s = torch.cat((x_s_clas, x_t[convert_all_idx]), dim=0)
        y_s = torch.cat((y_s_clas, y_conv.cuda()), dim=0)

        # AVH + CBST
        # replace softmax with AVH score
        discriminator.eval()
        theta.eval()
        for name, param in thetaNet.named_parameters():
            print (name, ":", param.size())
            #if name == "xxx":
            #    w = param.detach().numpy()
        w = None # temporary placeholder
        w = w.detach().numpy()
        x_feature = np.zeros([1, 1024])
        with torch.no_grad():
            for data, label in test_loader:
                feature_inter = theta.extractor(data).detach().numpy()
                x_feature = np.concatenate([x_feature, feature_inter], axis=0)
        x_feature = x_feature[1:]
        prediction_result = np.zeros([1, CONFIG["n_classes"]])
        for sample_id in range(x_feature.shape[0]):
            pred = avh_score(x_feature[sample_id], w)
            prediction_result = np.concatenate([prediction_result, pred])
        prediction_result = prediction_result[1:]
        avh_conf_idx = np.argmax(prediction_result, axis=1)
        avh_confidence = np.max(prediction_result, axis=1)

        p = 0.8  # The only parameter need to be tuned
        class_specific_num = np.zeros(CONFIG["n_classes"])
        lambda_k = np.zeros(CONFIG["n_classes"])
        for j in range(avh_conf_idx.shape[0]):
            class_specific_num[avh_conf_idx[j]] += 1
        class_specific_convert_num = p * class_specific_num
        # Get lambda_k and convert sample index
        convert_all_idx = np.zeros(1)
        for j in range(CONFIG["n_classes"]):
            class_idx = np.where(avh_conf_idx == j)[0]
            conf_class_value = avh_confidence[class_idx]
            class_convert = heapq.nlargest(int(class_specific_convert_num[j]), range(len(conf_class_value)),
                                           conf_class_value.take)
            j_class_convert = class_idx[class_convert]
            convert_all_idx = np.concatenate([convert_all_idx, j_class_convert])
            conf_class_tmp = np.sort(conf_class_value)
            lambda_k[j] = conf_class_tmp[-class_specific_convert_num[j]]
        # Get new pseudo labels
        new_prediction_result = prediction_result / lambda_k
        new_conf_idx = np.argmax(new_prediction_result, axis=1)
        # Convert samples from test set to train set
        convert_all_idx = convert_all_idx[1:]
        y_conv = torch.FloatTensor(np.zeros([convert_all_idx.shape[0], CONFIG["n_classes"]]))
        for k in range(convert_all_idx.shape[0]):
            y_conv[k, new_conf_idx[int(convert_all_idx[k])]] = 1
        x_s = torch.cat((x_s_clas, x_t[convert_all_idx]), dim=0)
        y_s = torch.cat((y_s_clas, y_conv.cuda()), dim=0)

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

from torchvision import datasets, transforms
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

# Alignment method
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

def test_model(source, target):
    import warnings
    from sklearn.metrics import brier_score_loss
    warnings.filterwarnings('ignore')
    task = source[0] + target[0]
    N_CLASSES = 65
    BATCH_SIZE = 64
    ce_func = nn.CrossEntropyLoss()

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
    b_score = 0
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
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
        print (("\nRelaxed aligned data with alternatively trained R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    # intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print(x_value)
    """
    # Aligned data with alternative training
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
    b_score = 0
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
            enc = OneHotEncoder(categories="auto")
            label = label.cpu().numpy().reshape(-1, 1)
            label = enc.fit_transform(label).toarray()
        print (("Aligned data with alternatively trained R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95,
                0.955, 0.96, 0.97, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print(x_value)
    
    # Aligned data with fixed R
    theta = thetaNet_align(2048, N_CLASSES)
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
        print (("Aligned data with Fixed R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    # intervals = [0, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 0.98, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print(x_value)
    
    test_loader = dataloader("OfficeHome/", target, 32, False)
    # IID
    theta = torch.load("models/sourceOnly_" + source + "_" + target + ".pkl")
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
    print (("IID: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
           (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    # intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print(x_value)

    # DeepCORAL
    theta = torch.load("models/deepcoral_" + source + "_" + target + ".pkl")
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
    print (("DeepCORAL: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
           (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    # intervals = [0, 0.27, 0.33, 0.4, 0.5, 0.6, 0.7, 0.72, 0.75, 0.77, 0.8, 0.83, 0.85, 0.87, 0.90, 0.92, 0.94, 0.95,
    #             0.96, 0.97, 1]  # aw
    intervals = [0.0, 0.6, 0.65, 0.67, 0.7, 0.72, 0.77, 0.8, 0.85, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                 0.96, 0.97, 0.98, 0.985, 0.99, 1]  # aw
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print(x_value)

    # relaxed DeepCORAL
    theta = torch.load("models/relaxed_deepcoral_" + source + "_" + target + ".pkl")
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
    print (("Relaxed DeepCORAL: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
           (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
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
    #print(x_value)

    # Temperature scaling
    from temperature_scaling import ModelWithTemperature, _ECELoss
    orig_model = torch.load("models/sourceOnly_" + source + "_" + target + ".pkl")
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
        print("Temperature scaling: test_loss: %.3f, test_acc: %.4f, mis_ent_clas: %.3f\n" % (
                test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
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
    #print(x_value)

    # IW + Fixed R
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_fixed_" + task + ".pkl"))
    discriminator = torch.load("models/dis_iw_fixed_" + task + ".pkl")
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
        print (("IW with fixed R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
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
    #print(x_value)

    # IW + Alternative Training
    theta = IWNet(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_iw_alter_" + task + ".pkl"))
    discriminator = torch.load("models/dis_iw_alter_" + task + ".pkl")
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
        print (("IW with alternatively trained R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
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
    #print(x_value)

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
        print (("End2end training with fixed R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num, mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print (x_value)

    # End2end with alternative r
    theta = thetaNet_e2e(N_CLASSES)
    theta.load_state_dict(torch.load("models/theta_rba_alter_" + task + ".pkl"))
    discriminator = torch.load("models/dis_rba_alter_" + task + ".pkl")
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
        print (("End2end training with alternative R: test_loss:%.3f, test_acc: %.4f, mis_ent_clas: %.3f\n") %
               (test_loss * 1e3 / test_num, test_acc / test_num,  mis_entropy_clas / mis_num))
    confidence = confidence[1:].numpy()
    accuracy = accuracy[1:].numpy()
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((confidence >= intervals[i]) == (confidence < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(confidence[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(confidence[int_idx]) - np.sum(accuracy[int_idx])) / np.sum(int_idx))
    #print(x_value)
    """

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
    source = "amazon"
    target = "webcam"
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
    print("Self-training with softlabels")
    self_training_softlabels(source_x, source_y, target_x, target_y, task_name)
    #print("\n\nTrain with soft labels")
    #softlabels(source_x, source_y, target_x, target_y, task_name)
    """
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
    #softlabels_relaxed(source_x, source_y, target_x, target_y, task_name)
    #test_model(source, target)
    """

    """
    Number of iterations for E2C2-DCORAL and E2C2-R-DCORAL
    learning rate from 1e-3, 1e-4 to 1e-4, 1e-5
    A-C: 20, 20
    A-P: 35, 60
    A-R: 20, 70
    C-A: 20, 75
    C-P: 20, 70
    C-R: 20, 70
    P-A: 20, 70
    P-C: 20, 75
    P-R: 20, 75
    R-A: 20, 75
    R-C: 20, 70
    R-P: 25, 75
    """

    """
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    for source in domains:
        for target in domains:
            if source != target:
                task_name = source[0]+target[0]
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
                early_fix(source_x, source_y, target_x, target_y, task_name)
                """
    #print "\n\nStart Training IID"
    #iid_baseline(source_x, source_y, target_x, target_y, task_name)
    #print "\n\nStart Training IW"
    #train_iw(source_x, source_y, target_x, target_y, task_name)
    #print "\n\nTraining Fixed Discrimiantor (After Few Epoches)"
    #early_fix(source_x, source_y, target_x, target_y, task_name)
    #print("\n\nTrain with soft labels")
    #softlabels(source_x, source_y, target_x, target_y, task_name)
    #print ("\n\nTrain soft labels with relaxed alignment")
    #softlabels_relaxed(source_x, source_y, target_x, target_y, task_name)
    #confidence_accuracy_plot(target_x, target_y, task_name)


"""
A-P convergence metrics
with softlabels
train_loss: [32.63186817988753, 6.970758674267147, 2.0915692934899455, 9.619338672075951, 12.197856425440737, 5.949250824217286, 3.721071489687477, 2.300235320414816, 1.4788291400431524, 1.150239790024768, 0.8572585849157934, 0.6805468975965465, 0.5900798649859748, 0.4726375895552337, 0.39538519490244134, 0.4093360702972859, 0.32761398104152506, 0.29421921353787184, 0.3187119638148163, 0.27022512230489937, 0.23690240158300313, 0.31858028278553063, 0.24964664863156422, 0.21063809915046608, 0.1696760656445154, 0.15194170443075045, 0.13305744422333582, 0.2321802784821817, 0.17290293305580107, 0.15949732324640664, 0.6625924350893391, 1.7542186687101744, 2.3948010098787824, 2.425479236823906, 2.013729515851342, 1.8740562608997735, 1.7241600925834581, 1.510725471556985, 1.2580470577813685, 1.1319620852425163]
test_loss: [30.222374647956464, 23.21066276853647, 22.41587304510386, 33.89270484487762, 26.653320275332696, 24.075497697940627, 22.90019991162071, 22.230244577132414, 22.02226721399887, 21.850033020162186, 21.87206216641551, 21.846547806738304, 21.897614687029964, 21.889127342785283, 22.039527357577956, 22.40599259838217, 22.60669890651028, 22.99050997173123, 22.96449112768618, 22.917254466842923, 23.164172730055927, 23.038042131429535, 22.764502357969306, 22.84906554474544, 23.20155436242317, 23.579127591428094, 23.6583307375374, 23.595837932847914, 23.471328046145164, 22.19552260901805, 24.70400245320182, 31.782293969592846, 35.23227437494789, 35.962849056487094, 36.7745020336384, 37.29101297795759, 36.827160340489826, 37.76941938050079, 38.32043165235698, 40.70722312910059]
test_acc: [0.5580085604865961, 0.6001351655778329, 0.6307726965532777, 0.625366073439964, 0.6213111061049786, 0.6213111061049786, 0.6219869339941428, 0.6282946609596756, 0.6328001802207704, 0.6411353908537959, 0.6512728091912593, 0.6517233611173687, 0.646316738004055, 0.6447398062626718, 0.6460914620410002, 0.6476683937823834, 0.6474431178193287, 0.6460914620410002, 0.6449650822257265, 0.645415634151836, 0.6393331831493579, 0.6490200495607119, 0.654201396710971, 0.646316738004055, 0.6427123225951791, 0.6391079071863032, 0.6406848389276865, 0.6492453255237666, 0.653300292858752, 0.6494706014868213, 0.6521739130434783, 0.6334760081099347, 0.61567920702861, 0.5769317413831944, 0.5636404595629646, 0.5440414507772021, 0.5467447623338589, 0.5316512728091912, 0.5330029285875197, 0.5325523766614102]

without softlabels
train_loss: [32.92271992723857, 6.9459486832576145, 2.137880667744737, 0.8793243389975812, 0.5075376743583807, 0.33991072443313897, 0.25274908303150107, 0.19542915813092676, 0.19277389527165464, 0.1562311962646033, 0.17732659886990276, 0.15712845205728498, 0.15351018435986977, 0.13597072434744664, 0.12640069238841534, 0.10437847813591361, 0.13928497309929558, 0.13024793904540796, 0.11887015209400227, 0.0994611910677382, 0.09795435371675662, 0.09684772663084525, 0.087474599547152, 0.08224549137854151, 0.08196362759917974, 0.08274276341710772, 0.09033673808776907, 0.08592831103929452, 0.10812826970193003, 1.1158038950192608, 3.0606718120231693, 3.7861924309150448, 3.7535239193987633, 3.2394758806497395, 2.1819302457983474, 1.7656576024767543, 1.3916169210071012, 1.4582000237091313, 1.503999672131613, 1.4174810735442278]
test_loss: [30.833475276623044, 23.13757730942907, 22.019303336447493, 22.31081238628696, 23.193905095201163, 23.966503172851034, 24.80013785692023, 25.03312454483589, 25.260122409407206, 25.412116375128463, 25.574350335571673, 25.609247223647174, 25.80565129563894, 25.667380403092004, 25.800818539224355, 25.817991216109125, 26.063870902962716, 26.09801590509795, 26.163858292096904, 26.742323531468568, 26.345518505984153, 26.330510112825184, 26.38725217706208, 26.787778860707895, 26.93743210543542, 26.74272125383357, 26.50659882987397, 25.556035099815855, 22.697938981693213, 39.862666347054216, 53.952685560452885, 68.32582151924713, 63.66626666886067, 67.1422885972977, 70.0426063819038, 69.2719423601718, 70.24372966742295, 75.70684149502151, 75.1318075711138, 84.34772762690451]
test_acc: [0.5465194863708043, 0.6041901329128182, 0.6289704888488399, 0.645415634151836, 0.6449650822257265, 0.6433881504843433, 0.6440639783735076, 0.6395584591124127, 0.6415859427799054, 0.6458661860779454, 0.6427123225951791, 0.6406848389276865, 0.6373056994818653, 0.6508222572651499, 0.6492453255237666, 0.6429375985582338, 0.6469925658932192, 0.645415634151836, 0.6368551475557558, 0.644514530299617, 0.6440639783735076, 0.647217841856274, 0.6440639783735076, 0.645415634151836, 0.6433881504843433, 0.647217841856274, 0.6427123225951791, 0.6505969813020951, 0.6438387024104528, 0.6111736877675152, 0.5785086731245777, 0.5305248929939176, 0.5393106555530525, 0.5379589997747241, 0.5334534805136292, 0.5314259968461366, 0.5280468574003154, 0.5264699256589322, 0.52399189006533, 0.5287226852894796]


C-P convergence metrics:
with softlabels
soft_train_loss = [35.20621783499207, 8.14895910942661, 2.6097696427521964, 10.63975038472563, 13.570522756448813, 6.812829585479839, 4.205688248122377, 2.5971930926399573, 1.794081081835819, 1.4613654369687927, 1.1803819806248481, 0.9767414235310363, 0.8956374221348338, 0.7816150100968245, 0.7195404414752764, 0.6775513048549847, 0.6114979696992252, 0.5794830161279865, 0.6035833010849143, 0.564954713751961, 0.5067042352831257, 0.5743104249372014, 0.5198222279016461, 0.48061612361509887, 0.4987000426210995, 0.46041734450097593, 0.4365036343889577, 0.4139910874489163, 0.4206441668793559, 0.3527549897054476, 0.4206477697672589, 0.37312796605484827, 0.34236789814063484, 0.45320828278948155, 0.42668943226869616, 0.3918179527058133, 0.3448771858321769, 0.5054844465173249, 1.4270401736056164, 2.9388303437735885]
soft_test_loss = [32.746622196643514, 24.865809231358732, 23.645403270223007, 36.16761591929148, 28.230447735864853, 25.561248995640653, 24.11665471187178, 23.437699586267165, 23.175006676536608, 23.072467808466715, 23.065014018698786, 23.25982752349039, 23.29037511206393, 23.412876890220737, 23.264705825963787, 23.659182497176214, 23.880020660117662, 23.90215364118019, 23.768455333928852, 23.87662275225817, 24.183914102286632, 23.99487792438525, 24.0172709960018, 24.044989519061257, 24.767777986369655, 24.566832735990182, 24.55401279604577, 24.707449156573613, 24.866385176848333, 25.15188010594522, 25.090318175687745, 25.369641492967805, 25.27380575062751, 25.344362033866336, 25.829300334738782, 25.234133763795608, 23.492884947443148, 27.68246712354852, 40.57877937612514, 42.918443948359446]
soft_test_acc = [0.5667943230457311, 0.595629646316738, 0.6138769993241721, 0.6055417886911466, 0.6134264473980626, 0.6041901329128182, 0.6118495156566794, 0.6129758954719532, 0.6163550349177743, 0.6170308628069385, 0.6204100022527597, 0.6242396936246902, 0.6217616580310881, 0.6210858301419239, 0.6217616580310881, 0.6206352782158143, 0.6204100022527597, 0.6197341743635955, 0.6195088984005407, 0.6179319666591575, 0.6177066906961027, 0.6201847262897049, 0.625366073439964, 0.6233385897724713, 0.6199594503266501, 0.6199594503266501, 0.6215363820680333, 0.6213111061049786, 0.6217616580310881, 0.6199594503266501, 0.6186077945483217, 0.6199594503266501, 0.6233385897724713, 0.6197341743635955, 0.6086956521739131, 0.6181572426222122, 0.6255913494030187, 0.6206352782158143, 0.6136517233611174, 0.6041901329128182]

without softlabels:
no_soft_train_loss = [35.16051894586001, 7.98972490842321, 2.4994539126886854, 1.2224012548436543, 0.7718347222544253, 0.6398112707704837, 0.5460388815429594, 0.5003681595969413, 0.4474692136448409, 0.4520417327460434, 0.4457637893834284, 0.3733741748146713, 0.4161463591403195, 0.3515838306131107, 0.39757901902443593, 0.39606742893478697, 0.3757812488558037, 0.3541618568955788, 0.38261631270870566, 0.3331890898490591, 0.3634078015706369, 0.34391689114272594, 0.3349987996209945, 0.36056007890562924, 0.3347541445067951, 0.3286022190669818, 0.2963378553145698, 0.30639346855293426, 0.8760418235656938, 2.979960644318323, 3.975116206066949, 4.7530653417509585, 5.62914250751159, 5.490497752824532, 5.776649785194812, 4.5370851909475665, 4.280487951057564, 4.781515892994191, 4.306387003910329, 4.133303333406469]
no_soft_test_loss = [32.21186519501633, 24.439743763424573, 23.358371537610953, 23.65915408459943, 24.553446397504228, 24.88546474583113, 25.00860763310473, 25.227178791993158, 25.70469241390413, 26.365851209249065, 26.207417471233214, 25.596825140342705, 25.89230433754299, 26.564928386414955, 26.69857211090418, 26.387083957420327, 26.32692921508081, 26.842417333490758, 27.1137347493147, 26.714837330906047, 26.583686850312112, 26.747797611839843, 27.445295555791922, 27.402748608272283, 27.673079076450293, 27.16481242531134, 26.749347587726383, 23.661141568514967, 40.272358436524314, 57.683992128056374, 63.86118596887556, 70.56364043120409, 68.91328296459427, 72.76245999320129, 74.2483291574415, 74.81978025004574, 76.36602340074347, 74.28867582212457, 75.92792853655969, 76.11743478416241]
no_soft_test_acc = [0.57197567019599, 0.5974318540211759, 0.614778103176391, 0.6204100022527597, 0.6210858301419239, 0.6217616580310881, 0.6231133138094165, 0.625366073439964, 0.6197341743635955, 0.6123000675827889, 0.6172561387699932, 0.6260419013291282, 0.6251407974769092, 0.6152286551025006, 0.6213111061049786, 0.6217616580310881, 0.6251407974769092, 0.6224374859202523, 0.6188330705113765, 0.6183825185852669, 0.6271682811444019, 0.6233385897724713, 0.6190583464744311, 0.6204100022527597, 0.6208605541788691, 0.6217616580310881, 0.6208605541788691, 0.6219869339941428, 0.6159044829916648, 0.605316512728092, 0.5753548096418112, 0.5712998423068258, 0.5566569047082677, 0.5609371480063077, 0.5555305248929939, 0.551926109484118, 0.5613876999324172, 0.5507997296688444, 0.5609371480063077, 0.5559810768191034]

R-A convergence metrics:
with softlabels:
soft_train_loss = [34.771711835502714, 9.083574300334938, 3.803165107830495, 11.750581207941623, 15.344005148263946, 8.775147640456757, 6.3960194169287234, 4.423991218209267, 3.2476642388391537, 2.7553823496471495, 2.2531465937693915, 1.8283347464234068, 1.56108682597245, 1.3950925941268602, 1.225176188028485, 1.3159833797861054, 1.1462667265205064, 1.019756528346435, 1.0979028884321451, 0.9543554212871022, 0.8948723373907631, 0.9352822482680389, 0.8609371813203114, 0.7868468025834232, 0.6538643697411686, 0.5922831864892573, 0.5557271300990512, 0.5740383596065035, 0.5330037183897651, 0.45368892775065656, 0.5076086455686153, 0.49000829863159556, 0.4371221639566879, 0.44177644751102163, 0.4175370130552978, 0.37731488093571813, 0.4155635378202019, 0.39707017063662625, 0.3368564280748799, 0.4055413781948711, 0.3990816969788917, 0.7857649293525711, 1.2879827653692253, 1.0652391035852118, 0.9238973568679522, 1.1386386503505965, 0.953247689007633, 0.8657781856026554, 0.857356658536077, 0.7338162226453964, 0.6169320153328928, 0.6983544272573098, 0.5893224334020329, 0.5000118694295161, 0.5041182154209177, 0.5465421204765637, 0.41627446661932743, 0.4381440898767956, 0.42472827869157, 0.38196807946550887]
soft_test_loss = [30.333785397750045, 22.909304365779718, 21.850549811240228, 35.40406847962834, 27.688667172014245, 24.6742932355399, 22.88248193180507, 22.227072843465972, 21.83485284085425, 22.040813068426043, 22.133779565051203, 22.04161703119958, 22.066544985447, 22.3877206475439, 22.368301174303706, 22.58890064154626, 22.849719564660273, 22.66580055513449, 22.57604411368415, 22.700488125290654, 22.889004993399425, 23.121257573918744, 23.169387084346372, 23.047762664689614, 23.156408252763217, 23.45887078244744, 23.49550541310729, 23.482347841266748, 24.017586131423993, 23.946139907758326, 23.948896997103635, 24.183161424716214, 24.08487835095456, 23.86558502851124, 23.76456785260968, 24.076692056989884, 24.44435499507211, 24.794941435736494, 24.8280114286664, 23.9450844119103, 23.509149449226243, 25.388994431073254, 26.75867571868079, 27.257853247866887, 27.829913922087076, 28.10162504759648, 28.005234341489764, 28.995003383936684, 28.73541078893648, 28.977473625981567, 30.264461978649866, 29.459559529418065, 29.263856531810408, 29.805841394840172, 30.183953073813584, 30.358602836586883, 30.499547855029835, 30.69559045421473, 30.523254294723554, 31.232180004017707]
soft_test_acc = [0.6271116604861969, 0.6279357231149567, 0.6402966625463535, 0.6407086938607334, 0.6431808817470128, 0.6468891635764318, 0.6510094767202307, 0.6440049443757726, 0.6477132262051916, 0.6407086938607334, 0.6374124433456942, 0.642356819118253, 0.6435929130613927, 0.6415327564894932, 0.6440049443757726, 0.6440049443757726, 0.6394725999175938, 0.6427688504326329, 0.6427688504326329, 0.6415327564894932, 0.6456530696332922, 0.6452410383189122, 0.638236505974454, 0.6427688504326329, 0.6444169756901524, 0.6407086938607334, 0.6411207251751133, 0.646477132262052, 0.6378244746600742, 0.6398846312319736, 0.6448290070045324, 0.6431808817470128, 0.6411207251751133, 0.6448290070045324, 0.6440049443757726, 0.6435929130613927, 0.6415327564894932, 0.6394725999175938, 0.6390605686032138, 0.6374124433456942, 0.6328800988875154, 0.630819942315616, 0.6234033786567779, 0.6102183765966214, 0.6081582200247219, 0.6048619695096827, 0.6098063452822414, 0.5945611866501854, 0.6015657189946436, 0.6023897816234034, 0.5990935311083643, 0.6007416563658838, 0.5982694684796045, 0.5990935311083643, 0.6019777503090235, 0.5966213432220849, 0.5974454058508447, 0.5953852492789452, 0.5995055624227441, 0.6044499381953028]

without softlabels:
no_soft_train_loss = [34.98384601072125, 8.98713592601859, 3.9048411714695934, 2.2626526771869133, 1.6717072201730765, 1.3313394444792166, 1.1355400895294936, 0.9839456349584287, 0.846616758465551, 0.7975312810310202, 0.7169902409035442, 0.7087146154726328, 0.6175781860002789, 0.5460477845770293, 0.49539631387839717, 0.5089806612122102, 0.4850494928415055, 0.43127405848624045, 0.4150400582727962, 0.3903906131028265, 0.3734465826815669, 0.35195921453228896, 0.335640682960334, 0.34161516822928534, 0.2974334591324779, 0.28231329237367364, 0.27132022491507773, 0.2599095210325027, 0.25481823730565933, 0.2410051466870135, 0.23861848276810368, 0.2269897917015613, 0.24869014556263236, 0.22656373119494622, 0.20135932777454887, 0.20640437504735545, 0.16072983718544678, 0.19444898175804512, 0.17481896135470143, 0.18316581575334936, 0.15717022512378037, 0.12931609053866588, 0.2613103126877568, 1.467857347038723, 1.9370631544270378, 1.742286076668002, 1.6237369363965546, 1.172503821072641, 1.0684997388296693, 0.897337374297659, 0.7515409279722666, 0.6209238594316918, 0.7303362860735775, 0.6673287809488998, 0.5560083910350458, 0.584624552478393, 0.5340595449577422, 0.5972323367250678, 0.6899644104439927, 0.6620614232657397]
no_soft_test_loss = [30.558909022076218, 22.420852283513934, 22.01483064980776, 22.622761359959412, 22.893083257639805, 23.538051635874215, 24.17230871316087, 24.5965936078817, 24.66690439621902, 25.014644794912066, 25.20612703707663, 25.39100846391575, 25.636925096122827, 25.792914847557466, 25.91586491184368, 26.35544015391648, 26.068377946697247, 26.211447709865123, 26.554166558449978, 26.68233290099783, 26.554120976982855, 26.331330455769027, 27.59714855006265, 27.481433389409855, 27.036938738027224, 27.7775768198297, 27.994110940855425, 27.48538640285157, 27.525721089457797, 28.13022393083985, 28.29173778279308, 28.53963372144475, 28.818643628494897, 28.614354369372165, 28.51616658445342, 29.085599791625423, 29.009084619019912, 29.01144994351419, 29.091332152295514, 29.09880392729229, 30.011049012465882, 29.988853293875877, 29.04372918640187, 38.11613988817401, 40.76091222523366, 40.6782076984558, 44.71999399446881, 43.47052616399161, 44.292147531890556, 44.27231379777353, 47.074567254895186, 45.23207820095444, 47.87615935007731, 48.77491801866642, 49.833344637144236, 49.21290336148357, 50.48434219588259, 49.532613098056515, 50.51188189943091, 51.019766716786165]
no_soft_test_acc = [0.6188710341985991, 0.6505974454058508, 0.6435929130613927, 0.6378244746600742, 0.6427688504326329, 0.6448290070045324, 0.6398846312319736, 0.6341161928306551, 0.6407086938607334, 0.6427688504326329, 0.638236505974454, 0.6357643180881747, 0.6328800988875154, 0.6374124433456942, 0.6374124433456942, 0.6295838483724763, 0.6374124433456942, 0.6435929130613927, 0.6345282241450351, 0.6291718170580964, 0.638648537288834, 0.6378244746600742, 0.630407911001236, 0.6324680675731356, 0.6411207251751133, 0.6316440049443758, 0.6320560362587556, 0.6378244746600742, 0.6419447878038731, 0.6341161928306551, 0.6353522867737948, 0.6349402554594149, 0.6328800988875154, 0.6316440049443758, 0.6345282241450351, 0.6345282241450351, 0.6349402554594149, 0.6320560362587556, 0.638648537288834, 0.6370004120313144, 0.6291718170580964, 0.630819942315616, 0.6324680675731356, 0.6073341573959621, 0.5904408735063865, 0.5867325916769675, 0.580552121961269, 0.5822002472187886, 0.5879686856201072, 0.5854964977338277, 0.5908529048207664, 0.5970333745364648, 0.5953852492789452, 0.5896168108776267, 0.6011536876802637, 0.5974454058508447, 0.5850844664194479, 0.5966213432220849, 0.595797280593325, 0.5949732179645653]


R-P convergence metrics:
with softlabels:
soft_train_loss = [33.16858852016074, 7.444751053782446, 2.5395173659282073, 10.469610918911972, 13.319425964727998, 6.890705096136246, 4.342557738002922, 2.832571037579328, 1.9464605980153595, 1.533451418259314, 1.1835422556448196, 0.9671012333793831, 0.9532050652030324, 0.7755258152194854, 0.6792787174760764, 0.6719008048198053, 0.6069789625637765, 0.52535589351984, 0.4979416967502662, 0.4832166765949556, 0.41456460520359023, 0.4316349853096264, 0.38642492378130555, 0.3739884096596922, 0.39406458514609505, 0.3516578075609037, 0.3195143667315798, 0.3110874810123018, 0.28397222116057363, 0.27464213016043815, 0.24683738326919927, 0.20176247926428914, 0.20677919566099132, 0.26939903053322006, 0.23572005531085388, 0.20432881844629136, 0.17320815074656692, 0.1609713301461722, 0.13465746073052287, 0.17630681728145906, 0.2226720083438392, 0.7485752842122956, 1.030786007842315, 0.5549345331798706, 0.4610391104194735, 0.45888366809646997, 0.35382861936730997, 0.29663512433346895, 0.30046572343313266, 0.33432325980226907, 0.23556509221504843, 0.2833041626893516, 0.21737556555308402, 0.21506931050680578, 0.20849490854223923, 0.2232686471792736, 0.22286849562078714, 0.25784254190512, 0.2402482637470322, 0.25230375002138317, 0.26673801076997605, 0.26915540669246446, 0.2748956243574087, 0.29979992325284655, 0.2550989830134703, 0.2994343956067626, 0.40078534461957005, 0.36475106491707265, 0.4188465227239898, 0.43914588800232324, 0.44446276417667313, 0.43988038336725105, 0.4681316523679665, 0.44404994530071107, 0.4359459298263703, 0.5216316220217517, 0.4686666569406433, 0.4435558848282588, 0.47527056819360175, 0.4142917536332139]
soft_test_loss = [22.802746491970364, 14.78326891316678, 13.591255293582092, 28.401967963003635, 19.431792110654516, 16.42615442260962, 14.782099109910904, 13.903190065649452, 13.61674587316786, 13.37196148712105, 13.547656080373157, 13.56353714649246, 13.412536801881526, 13.48459787749041, 13.618660902327099, 13.603460428869974, 13.6271877636859, 13.89247677862765, 13.938140207165183, 13.962148049039383, 14.14913313698731, 14.124777234333758, 14.12917951872607, 14.141062696282459, 14.283424093530478, 14.573342643307253, 14.621181426539188, 14.576579018416881, 14.626149404529624, 14.7886189570645, 14.66729596515689, 14.848694913584414, 14.82590617400941, 15.01696173283774, 14.921055308170322, 15.039727638959617, 15.106663077660148, 15.004617468573322, 15.104255481033771, 14.442402967651516, 14.484583015618064, 17.27171856209243, 16.28474048946215, 16.80504180042828, 17.61607207755673, 17.326819370340242, 17.248944928447898, 17.3369656143008, 18.00743842425715, 18.299823356227957, 18.571758611322014, 18.356266108201574, 18.292661063921937, 18.449354209349913, 18.81960697017239, 19.354862843475033, 19.383076129396855, 19.33567369271571, 19.328054409087255, 19.557422830543423, 20.056885449997502, 20.727166992019065, 21.01276503084049, 21.447885409752832, 21.760774077690247, 22.894688838504138, 24.150359963586563, 24.89179864233747, 25.858051781182784, 26.28629439638176, 27.42712230235506, 28.400238783785447, 28.402664460532165, 29.085290179539435, 29.19635388141657, 31.07889394390607, 30.813226562331565, 31.186223970014893, 31.09171651563926, 31.64351537007957]
soft_test_acc = [0.7384546068934444, 0.7555755800856049, 0.7654877224600135, 0.7650371705339041, 0.7609822031989186, 0.7625591349403019, 0.7666141022752873, 0.7668393782383419, 0.7715701734624916, 0.7713448974994368, 0.7708943455733274, 0.7713448974994368, 0.7751745888713675, 0.7742734850191485, 0.7731471052038748, 0.7697679657580536, 0.7735976571299842, 0.7756251407974769, 0.772020725388601, 0.7722460013516558, 0.7697679657580536, 0.7706690696102726, 0.77292182924082, 0.7731471052038748, 0.7688668619058346, 0.7699932417211084, 0.7706690696102726, 0.7681910340166704, 0.770218517684163, 0.7704437936472178, 0.7722460013516558, 0.7699932417211084, 0.7708943455733274, 0.77292182924082, 0.7708943455733274, 0.7715701734624916, 0.7753998648344221, 0.7740482090560937, 0.7726965532777652, 0.7708943455733274, 0.770218517684163, 0.7686415859427799, 0.7600810993466998, 0.7571525118269881, 0.7494931290831268, 0.7483667492678531, 0.7494931290831268, 0.7483667492678531, 0.7515206127506195, 0.7503942329353458, 0.7508447848614552, 0.75107006082451, 0.749267853120072, 0.7497184050461816, 0.7508447848614552, 0.7485920252309078, 0.7452128857850867, 0.7474656454156342, 0.7494931290831268, 0.7503942329353458, 0.7456634377111963, 0.7470150934895247, 0.7479161973417436, 0.7443117819328677, 0.7413831944131561, 0.738004054967335, 0.7332732597431854, 0.7350754674476233, 0.7235863933318315, 0.7294435683712548, 0.7314710520387475, 0.7278666366298716, 0.7285424645190358, 0.731921603964857, 0.7240369452579409, 0.7249380491101599, 0.7274160847037621, 0.7242622212209957, 0.7244874971840505, 0.7235863933318315]

without softlabels:
no_soft_train_loss = [34.95488563552499, 7.101503820158541, 2.6511119342675165, 1.362959036736616, 0.8306408859789371, 0.6140394901324596, 0.4627653490751982, 0.4365989050295736, 0.3469612782022783, 0.31016049407688634, 0.2746493492408522, 0.25767535536683034, 0.24625971414414902, 0.23034083091520838, 0.21932333109102078, 0.22121220594272017, 0.20546386916456477, 0.1990886531504137, 0.19185718880700212, 0.1667945968386318, 0.16914852096566133, 0.17930983532486217, 0.16659013128706388, 0.1274107942091567, 0.1415418314614466, 0.1296558603644371, 0.12299593504784363, 0.477185396344534, 2.052572637497048, 2.2948226573810513, 1.881202120899356, 1.615969353920913, 1.2599138898908029, 1.1029000009875745, 0.9433061230395522, 0.7204520023827041, 0.7126870011312089, 0.4869831627833524, 0.5196937684169306, 0.5663925331152443, 0.6242185054413443, 0.6167243672202208, 0.4430736636277288, 0.4491757007781416, 0.4167714375736458, 0.36630609671452213, 0.35042510401191457, 0.557366964972711, 0.3044217271131596, 0.4648716074214982, 0.3884214527040188, 0.3260362083424947, 0.30247993733999984, 0.3562893345952034, 0.3397129224945924, 0.4669701728770243, 0.4071069453909461, 0.19809188986463205, 0.2397124412735658, 0.38284310089823387, 0.31528654924061683, 0.28022632169138106, 0.40587081390965196, 0.4798519212220396, 0.3150103407512818, 0.44817988611092524, 0.5102775329058724, 0.3381305951292493, 0.42930435455803356, 0.4558095021639019, 0.39482626806212856, 0.49655680694351245, 0.6775362194249673, 0.4908041898826403, 0.4687381637216147, 0.6506216217530891, 0.595869490727117, 0.5858483354261677, 0.6610524275207094, 0.4508619028742292]
no_soft_test_loss = [20.210950861752746, 14.7323192414681, 13.898970375030958, 13.78724814857763, 14.008847865254848, 14.492384872879093, 14.584109546311618, 14.921559557557025, 15.183296564007213, 15.16136422327011, 15.410760537169221, 15.385626699237518, 15.504072595485885, 15.389211551378803, 15.636130868650069, 15.60143350641801, 15.800120581321604, 15.7387093142578, 15.993118702385118, 15.97977285450658, 16.22929137202635, 16.222269934129812, 16.02642950804337, 16.433560303079837, 16.241510915015027, 16.231588292535456, 15.115300433865054, 18.17265940453722, 28.405443830891862, 29.64767936544124, 32.346049335846686, 32.64172524583047, 32.40727833139652, 35.88952611872516, 32.12130733050857, 34.43484405430353, 32.049499282591995, 33.5372629184985, 34.978966987946734, 35.47647940492168, 35.120754567962905, 38.10215088987597, 36.53845879160947, 39.793541689771985, 40.20336914180662, 36.53068146637953, 39.9961706270208, 41.049647086857405, 39.63150753043583, 40.81289361097384, 40.55349705696751, 38.69194990075582, 43.331343275277725, 43.23181328029078, 42.99690080779338, 41.80691310161136, 43.6241449295172, 45.56037213357082, 47.59546771971163, 45.00434063769429, 45.5046308875809, 44.47069938023312, 44.96824885037709, 44.99763349714406, 47.407846336403416, 47.534075700960116, 49.26951169484261, 53.78914729354886, 55.4625005114049, 52.819045526161204, 54.67563381493696, 58.03928782902851, 55.01240651654241, 56.92409823192726, 56.41304812847051, 64.3145487528176, 59.505365283727166, 62.957618775789896, 61.058224523139984, 65.79799625513172]
no_soft_test_acc = [0.7391304347826086, 0.7530975444920027, 0.7594052714575354, 0.7627844109033566, 0.7684163099797252, 0.7699932417211084, 0.772020725388601, 0.7722460013516558, 0.7695426897949988, 0.7717954494255463, 0.7695426897949988, 0.7713448974994368, 0.7735976571299842, 0.7747240369452579, 0.7715701734624916, 0.7715701734624916, 0.7715701734624916, 0.7717954494255463, 0.7688668619058346, 0.7704437936472178, 0.7717954494255463, 0.771119621536382, 0.7715701734624916, 0.771119621536382, 0.772020725388601, 0.7731471052038748, 0.7706690696102726, 0.7661635503491777, 0.7443117819328677, 0.7265149808515431, 0.7150259067357513, 0.7120973192160397, 0.7199819779229556, 0.7190808740707366, 0.7258391529623789, 0.728317188555981, 0.7314710520387475, 0.738905158819554, 0.7355260193737329, 0.7375535030412255, 0.7355260193737329, 0.7368776751520613, 0.7400315386348276, 0.7368776751520613, 0.737102951115116, 0.738905158819554, 0.7386798828564992, 0.7404820905609372, 0.739806262671773, 0.738905158819554, 0.7373282270781708, 0.7362018472628971, 0.7407073665239919, 0.7418337463392656, 0.7355260193737329, 0.7328227078170759, 0.7337238116692949, 0.7366523991890065, 0.7350754674476233, 0.7323721558909664, 0.7289930164451452, 0.7330479837801307, 0.7312457760756927, 0.731921603964857, 0.7303446722234738, 0.7280919125929264, 0.7256138769993242, 0.7256138769993242, 0.7269655327776526, 0.7280919125929264, 0.723135841405722, 0.7229105654426673, 0.7258391529623789, 0.7269655327776526, 0.7251633250732147, 0.7215589096643388, 0.7244874971840505, 0.7220094615904483, 0.7251633250732147, 0.7217841856273935]


"""