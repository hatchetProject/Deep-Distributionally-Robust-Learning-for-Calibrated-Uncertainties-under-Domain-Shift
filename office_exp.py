"""
Train on office31 dataset using the ResNet50 backbone
"""
import numpy as np
import torch
import math
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model_layers import ClassifierLayerAVH, GradLayer
import os
import argparse
import heapq
from torchvision import transforms, datasets
from sklearn.metrics import brier_score_loss
import random
import time
from torch.optim import lr_scheduler
from operator import itemgetter
import shutil
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    #"num_classes": 31,
    "batch_size": 16,
    "lr": 1e-5,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "print_freq": 10,
    "epochs": 10,
    "rand_seed": 0,
    "src_portion": 1,
    "src_portion_step": 0,
    "src_portion_max": 1,
}

parser = argparse.ArgumentParser(description='Choose DA task')
parser.add_argument('--num_classes', type=int,
                    help='number of classes for the domain adaptation task')
parser.add_argument('--src', type=str,
                    help='the domain adaptation task source domain')
parser.add_argument('--tgt', type=str,
                    help='the domain adaptation task target domain')
args = parser.parse_args()


class pseudo_dataset(Dataset):
    def __init__(self, input, label, transform=transforms.ToTensor()):
        self.input = input
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx]
        label = self.label[idx]
        #if self.transform:
        #    input = self.transform(input)

        return input, label

path_dic = {"amazon": "office/amazon", "webcam": "office/webcam", "dslr": "office/dslr"}

path_dic_home = {"Art": "OfficeHome/Art", "Clipart": "OfficeHome/Clipart", "Product": "OfficeHome/Product", "RealWorld": "OfficeHome/RealWorld"}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)

def dataloader_office(source_root, target_root):
    color_aug1 = transforms.ColorJitter(brightness=0.5)
    color_aug2 = transforms.ColorJitter(contrast=0.5)
    color_aug3 = transforms.ColorJitter(saturation=0.5)
    color_aug4 = transforms.ColorJitter(hue=0.5)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([color_aug1, color_aug2, color_aug3, color_aug4]),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(45, resample=False, expand=False, center=None),
            #transforms.RandomApply([]),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    kwargs = {'num_workers': 0, 'pin_memory': False}
    source_data = datasets.ImageFolder(root=source_root, transform=data_transforms["train"])
    source_data_loader = torch.utils.data.DataLoader(source_data, batch_size=CONFIG["batch_size"], shuffle=True)

    target_data = datasets.ImageFolder(root=target_root, transform=data_transforms["val"])
    target_data_loader = torch.utils.data.DataLoader(target_data, batch_size=CONFIG["batch_size"], shuffle=True)

    return source_data_loader, target_data_loader

def accuracy(output, label, num_class, topk=(1,)):
    """Computes the precision@k for the specified values of k, currently only k=1 is supported"""
    label = label.reshape([label.shape[0], 1])
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    #_, gt = label.topk(maxk, 1, True, True)
    gt = label
    pred = pred.t()
    pred_class_idx_list = [pred == class_idx for class_idx in range(num_class)]
    gt = gt.t()
    gt_class_number_list = [(gt == class_idx).sum() for class_idx in range(num_class)]
    correct = pred.eq(gt)

    res = []
    gt_num = []
    for k in topk:
        correct_k = correct[:k].float()
        per_class_correct_list = [correct_k[pred_class_idx].sum(0) for pred_class_idx in pred_class_idx_list]
        per_class_correct_array = torch.tensor(per_class_correct_list)
        gt_class_number_tensor = torch.tensor(gt_class_number_list).float()
        gt_class_zeronumber_tensor = gt_class_number_tensor == 0
        gt_class_number_matrix = torch.tensor(gt_class_number_list).float()
        gt_class_acc = per_class_correct_array.mul_(100.0 / gt_class_number_matrix)
        gt_class_acc[gt_class_zeronumber_tensor] = 0
        res.append(gt_class_acc)
        gt_num.append(gt_class_number_matrix)
    return res, gt_num

def accuracy_new(output, label, num_class, topk=(1,)):
    """Computes the precision@k for the specified values of k, currently only k=1 is supported"""
    label = label.reshape([label.shape[0], 1])
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    #_, gt = label.topk(maxk, 1, True, True)
    gt = label
    pred = pred.t()
    pred_class_idx_list = [pred == class_idx for class_idx in range(num_class)]
    gt = gt.t()
    gt_class_number_list = [(gt == class_idx).sum() for class_idx in range(num_class)]
    correct = pred.eq(gt)

    res = []
    gt_num = []
    for k in topk:
        correct_k = correct[:k].float()
        per_class_correct_list = [correct_k[0][pred_class_idx[0]].sum() for pred_class_idx in pred_class_idx_list]
        per_class_correct_array = torch.tensor(per_class_correct_list)
        gt_class_number_tensor = torch.tensor(gt_class_number_list).float()
        gt_class_zeronumber_tensor = gt_class_number_tensor == 0
        gt_class_number_matrix = torch.tensor(gt_class_number_list).float()
        gt_class_acc = per_class_correct_array.mul_(100.0 / gt_class_number_matrix)
        gt_class_acc[gt_class_zeronumber_tensor] = 0
        res.append(gt_class_acc)
        gt_num.append(gt_class_number_matrix)
    return res,gt_num

def entropy(p):
    p[p<1e-20] = 1e-20
    return -torch.sum(p.mul(torch.log2(p)))

class source_office(nn.Module):
    def __init__(self, n_output):
        super(source_office, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_output)
        )
        self.model.fc = extractor

    def forward(self, x_s):
        x = self.model(x_s)
        return x

class alpha_office(nn.Module):
    def __init__(self, n_output):
        super(alpha_office, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
        )
        self.model.fc = extractor
        self.final_layer = ClassifierLayerAVH(512, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x1 = self.model(x_s)
        x = self.final_layer(x1, y_s, r)
        return x

class beta_office(nn.Module):
    def __init__(self):
        super(beta_office, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 2)
        )
        self.model.fc = extractor
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p

class source_office_new(nn.Module):
    def __init__(self, n_output):
        super(source_office_new, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_output)

    def forward(self, x_s):
        x = self.model(x_s)
        return x

class alpha_office_new(nn.Module):
    def __init__(self, n_output):
        super(alpha_office_new, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential()
        self.final_layer = ClassifierLayerAVH(num_ftrs, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x1 = self.model(x_s)
        x = self.final_layer(x1, y_s, r)
        return x

class beta_office_new(nn.Module):
    def __init__(self):
        super(beta_office_new, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p

def train_and_val_iid(source_root, target_root, save_path_numpy, save_path_model):
    model = source_office(args.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), 1e-3,
                                momentum=CONFIG["momentum"], nesterov=True,
                                weight_decay=CONFIG["weight_decay"])
    train_loader, val_loader = dataloader_office(source_root, target_root)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    end = time.time()
    loss_func = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    model.train()
    best_prec, best_epoch = 0, 0
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(20):
        brier_score, test_num, mis_ent, mis_num = 0, 0, 0, 0
        batch_time.reset()
        losses.reset()
        top1.reset()
        # Training process
        for i, (input, label) in enumerate(train_loader):
            label = label.reshape((label.shape[0],))

            label = label.to(DEVICE)
            input = input.to(DEVICE)

            # compute output
            output = model(input)
            loss = loss_func(output, label.long())

            # measure accuracy and record loss
            prec1 = accuracy_new(output.data, label.long(), CONFIG["num_class"], topk=(1,))[0]
            losses.update(loss, input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % CONFIG["print_freq"] == 0:
                print('Training process: \n Epoch: [{0}][{1}/{2}]\n'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
                      'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
        scheduler.step()

        # validation process
        batch_time.reset()
        losses.reset()
        top1.reset()
        model.eval()
        end = time.time()
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                label = label.reshape((label.shape[0],))
                label = label.to(DEVICE)
                input = input.to(DEVICE)
                test_num += input.shape[0]

                # compute output
                output = model(input)
                loss = loss_func(output, label.long())
                prediction_t = F.softmax(output, dim=1)

                # measure accuracy and record loss

                prec1, gt_num = accuracy_new(output.data, label.long(), CONFIG["num_class"], topk=(1,))
                losses.update(loss, input.size(0))
                top1.update(prec1[0], gt_num[0])

                #mis_idx = (torch.argmax(prediction_t, dim=1) != label.long()).nonzero().reshape(-1, )
                #mis_pred = prediction_t[mis_idx]
                #mis_ent += entropy(mis_pred) / math.log(CONFIG["num_class"], 2)
                #mis_num += mis_idx.shape[0]
                label_onehot = torch.zeros(output.shape)
                label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
                for j in range(input.shape[0]):
                    brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j].cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % CONFIG["print_freq"] == 0:
                    print('Test: [{0}/{1}]\n'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                          'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
                          'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
                          'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))
        epoch_prec = top1.vec2sca_avg
        is_best = epoch_prec > best_prec
        best_prec = max(epoch_prec, best_prec)
        if is_best:
            best_epoch = epoch
            if not os.path.exists("runs/office_iid/"):
                os.mkdir("runs/office_iid/")
            if not os.path.exists("runs/OfficeHome_iid/"):
                os.mkdir("runs/OfficeHome_iid/")
            torch.save(model.state_dict(), save_path_model + "_best.pth.tar")
        torch.save(model.state_dict(), save_path_model + "_" + str(epoch) + ".pth.tar")

        list_metrics["acc"].append(epoch_prec)
        list_metrics["brier"].append(brier_score / test_num)
        #list_metrics["misent"].append(mis_ent / mis_num)
        list_metrics["loss"].append(losses.vec2sca_avg)

        print("Current precision: ", epoch_prec)
        print("")
    np.save(save_path_numpy, list_metrics)
    print("\n")
    print("Best accuracy: ", best_prec)
    print("Best epoch:", best_epoch)

def train_and_val_iid_new(source_root, target_root, save_path_numpy, save_path_model):
    model = source_office_new(CONFIG["num_class"])
    optimizer = torch.optim.SGD(model.parameters(), 1e-3,
                                momentum=CONFIG["momentum"], nesterov=True,
                                weight_decay=CONFIG["weight_decay"])
    train_loader, val_loader = dataloader_office(source_root, target_root)
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    end = time.time()
    loss_func = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    model.train()
    best_prec, best_epoch = 0, 0
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(20):
        brier_score, test_num, mis_ent, mis_num = 0, 0, 0, 0
        batch_time.reset()
        losses.reset()
        top1.reset()
        # Training process
        for i, (input, label) in enumerate(train_loader):
            label = label.reshape((label.shape[0],))
            label = label.to(DEVICE)
            input = input.to(DEVICE)

            # compute output
            output = model(input)
            loss = loss_func(output, label.long())

            # measure accuracy and record loss
            prec1 = accuracy_new(output.data, label.long(), CONFIG["num_class"], topk=(1,))[0]
            losses.update(loss, input.size(0))
            top1.update(prec1[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        scheduler.step()

        # validation process
        batch_time.reset()
        losses.reset()
        top1.reset()
        model.eval()
        end = time.time()
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                test_num += input.shape[0]
                label = label.to(DEVICE)
                input = input.to(DEVICE)

                # compute output
                output = model(input)
                label = label.reshape(-1, )
                loss = loss_func(output, label.long())

                # measure accuracy and record loss
                prec1, gt_num = accuracy_new(output.data, label.long(), CONFIG["num_class"], topk=(1,))
                losses.update(loss, input.size(0))
                top1.update(prec1[0], gt_num[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        epoch_prec = top1.vec2sca_avg
        is_best = epoch_prec > best_prec
        best_prec = max(epoch_prec, best_prec)
        if is_best:
            best_epoch = epoch
            if not os.path.exists("runs/office_iid/"):
                os.mkdir("runs/office_iid/")
            if not os.path.exists("runs/OfficeHome_iid/"):
                os.mkdir("runs/OfficeHome_iid/")
            torch.save(model.state_dict(), save_path_model + "_best_new.pth.tar")
        torch.save(model.state_dict(), save_path_model + "_" + str(epoch) + "_new.pth.tar")

        list_metrics["acc"].append(epoch_prec)
        list_metrics["brier"].append(brier_score / test_num)
        list_metrics["loss"].append(losses.vec2sca_avg)

        print("Current precision: ", epoch_prec)
        print("")
    np.save(save_path_numpy, list_metrics)
    print("\n")
    print("Best accuracy: ", best_prec)
    print("Best epoch:", best_epoch)

def train_one_epoch(train_loader, test_loader, model_alpha, model_beta, optimizer_alpha, optimizer_beta, schedular_alpha, schedular_beta, epoch):
    ## train loader sample number must be smaller than test loader
    model_alpha.train()
    model_beta.train()
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    iter_train = iter(train_loader)
    iter_test = iter(test_loader)
    min_len = min(len(train_loader), len(test_loader))
    bce_loss = nn.BCEWithLogitsLoss()
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    ce_loss = nn.CrossEntropyLoss()
    train_loss, train_acc = 0, 0
    for i in range(min_len):
        input, label = next(iter_train)
        input_test, _ = next(iter_test)
        label_train = label.reshape((-1,))
        label_train = label_train.to(DEVICE)
        input_train = input.to(DEVICE)
        input_test = input_test.to(DEVICE)
        BATCH_SIZE = input.shape[0]
        input_concat = torch.cat([input_train, input_test], dim=0)
        # this parameter used for softlabling
        label_concat = torch.cat(
            (torch.FloatTensor([1, 0]).repeat(input_train.shape[0], 1), torch.FloatTensor([0, 1]).repeat(input_test.shape[0], 1)), dim=0)
        label_concat = label_concat.to(DEVICE)

        prob = model_beta(input_concat, None, None, None, None)
        assert(F.softmax(prob.detach(), dim=1).cpu().numpy().all()>=0 and F.softmax(prob.detach(), dim=1).cpu().numpy().all()<=1)
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
        label_train_onehot = torch.zeros([BATCH_SIZE, CONFIG["num_class"]])
        for j in range(BATCH_SIZE):
            label_train_onehot[j][label_train[j].long()] = 1

        theta_out = model_alpha(input_train, label_train_onehot.cuda(), r_source.detach().cuda())
        source_pred = F.softmax(theta_out, dim=1)
        nn_out = model_alpha(input_test, torch.ones((input_test.shape[0], CONFIG["num_class"])).cuda(), r_target.detach().cuda())

        pred_target = F.softmax(nn_out, dim=1)
        prob_grad_r = model_beta(input_test, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                    sign_variable)
        loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape).cuda()))
        loss_theta = torch.sum(theta_out)

        # Backpropagate
        #if i < 5 and epoch==0:
        if i % 5 == 0:
            optimizer_beta.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_beta.step()

        #if i < 5 and epoch==0:
        if i % 5 == 0:
            optimizer_beta.zero_grad()
            loss_r.backward(retain_graph=True)
            optimizer_beta.step()

        if (i + 1) % 1 == 0:
            optimizer_alpha.zero_grad()
            loss_theta.backward()
            optimizer_alpha.step()

        train_loss += float(ce_loss(theta_out.detach(), label_train.long()))
        train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == label_train.long()).float() / BATCH_SIZE
        if i % CONFIG["print_freq"] == 0:
            train_loss = train_loss/(CONFIG["print_freq"]*BATCH_SIZE)
            train_acc = train_acc/(CONFIG["print_freq"])
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss: {3:.4f} \t Train Acc: {4:.4f}'.format(
                epoch, i, min_len, train_loss, train_acc*100.0))
            train_loss, train_acc = 0, 0
    schedular_alpha.step()
    schedular_beta.step()
    return model_alpha, model_beta, schedular_alpha, schedular_beta

def validate(test_loader, model_alpha, model_beta):
    # validate model and select samples for self-training
    model_alpha.eval()
    model_beta.eval()
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    top1_acc = AverageMeter()
    losses = AverageMeter()
    ce_loss = nn.CrossEntropyLoss()
    mis_ent, mis_num, brier_score, test_num = 0, 0, 0, 0
    pred_logit_list = []
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1, ))
            BATCH_SIZE = input.shape[0]
            test_num += BATCH_SIZE
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(), r_target.cuda()).detach()
            prediction_t = F.softmax(target_out, dim=1)
            test_loss = float(ce_loss(target_out, label.long()))
            losses.update(test_loss, BATCH_SIZE)
            prec1, gt_num = accuracy_new(prediction_t, label.long(), CONFIG["num_class"], topk=(1,))
            top1_acc.update(prec1[0], gt_num[0])
            mis_idx = (torch.argmax(prediction_t, dim=1) != label.long()).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            mis_ent += entropy(mis_pred) / math.log(CONFIG["num_class"], 2)
            mis_num += mis_idx.shape[0]

            #one-hot encoding
            label_onehot = torch.zeros(prediction_t.shape)
            label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)

            for j in range(input.shape[0]):
                brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j].cpu().numpy())
            if i % CONFIG["print_freq"] == 0:
                print("100 test samples processed")
            #    print('Test: [{0}/{1}]\n'
            #          'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
            #          'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
            #          'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
            #        i, len(test_loader), loss=losses, top1=top1_acc))
            for idx in range(input.shape[0]):
                pred_logit_list.append(prediction_t[idx, :])
    brier_score = brier_score/test_num
    misent = mis_ent/mis_num
    print("Mis entropy: {}, Brier score: {}".format(misent, brier_score))

    return top1_acc.vec2sca_avg, top1_acc.avg, losses.avg, misent, brier_score, pred_logit_list

def pseudo_labeling_cbst(pred_logit_list, epoch):
    # Select images which would be transferred
    pred_logit = torch.stack(pred_logit_list)
    pred_max = torch.max(pred_logit, dim=1)
    confidence = pred_max[0].cpu().numpy()
    conf_idx = pred_max[1].cpu().numpy()

    # CBST
    #p = min(0.2 + 0.05*epoch, 0.8)  # The only parameter need to be tuned, the portion of data to be converted
    p = 0.8
    #p = 0
    class_specific_num = np.zeros(CONFIG["num_class"])
    lambda_k = np.zeros(CONFIG["num_class"])
    for j in range(conf_idx.shape[0]):
        class_specific_num[conf_idx[j]] += 1
    class_specific_convert_num = p * class_specific_num
    ## Get lambda_k and convert sample index
    convert_all_idx = np.zeros(1)
    for j in range(CONFIG["num_class"]):
        class_idx = np.where(conf_idx == j)[0]
        conf_class_value = confidence[class_idx]
        class_convert = heapq.nlargest(int(class_specific_convert_num[j]), range(len(conf_class_value)),
                                       conf_class_value.take)
        j_class_convert = class_idx[class_convert]
        convert_all_idx = np.concatenate([convert_all_idx, j_class_convert])
        conf_class_tmp = np.sort(conf_class_value)
        if conf_class_tmp.shape[0] == 0:
            lambda_k[j] = 1e12
        else:
            lambda_k[j] = conf_class_tmp[-int(class_specific_convert_num[j])]
    convert_all_idx = convert_all_idx[1:]
    ## Get new pseudo labels
    new_prediction_result = pred_logit.cpu().numpy() / lambda_k
    new_conf_idx = np.argmax(new_prediction_result, axis=1)
    new_conf_idx = new_conf_idx.reshape(-1, )
    ## Convert samples from test set to train set
    returned_label = np.zeros_like(convert_all_idx)
    for j in range(convert_all_idx.shape[0]):
        returned_label[j] = new_conf_idx[int(convert_all_idx[j])]
    return convert_all_idx, returned_label

def train_and_val_rescue(source, target, path_name):
    # Do self-training with new parametric form, use different label selection criterions
    color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),
            transforms.RandomApply([color_aug], p=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    kwargs = {'num_workers': 0, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    model_alpha = alpha_office(CONFIG["num_class"])
    model_beta = beta_office()
    #optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    #optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_alpha = torch.optim.Adam(model_alpha.parameters(), lr=1e-5)
    optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=1e-5)
    scheduler_alpha = lr_scheduler.StepLR(optimizer_alpha, step_size=7, gamma=0.1)
    scheduler_beta = lr_scheduler.StepLR(optimizer_beta, step_size=7, gamma=0.1)
    random_seed = CONFIG["rand_seed"]
    prec1_best, prec_cls_best, loss_best, mis_ent_best, brier_best, best_epoch = 0, 0, 0, 0, 0, 0

    ## load checkpoint
    resume_path = "runs/office_iid/"+path_name+"_best.pth.tar"
    checkpoint = torch.load(resume_path)
    state = model_alpha.state_dict()
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        elif key == "final_layer.weight":
            state[key] = checkpoint["model.fc.2.weight"]
        elif key == "final_layer.bias":
            state[key] = checkpoint["model.fc.2.bias"]
        else:
            print("Param key {} not loaded correctly")
            raise ValueError("Parameter load error")
    model_alpha.load_state_dict(state, strict=True)
    print("=> loaded checkpoint from '{}'".format(resume_path))
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(CONFIG["epochs"]):
        val_data = datasets.ImageFolder(root=path_dic[target], transform=data_transforms["val"])
        # do not shuffle
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, **kwargs)

        prec1, prec_pcls, loss_val, mis_ent, brier_score, pred_logit_list = validate(val_loader, model_alpha, model_beta)

        list_metrics["acc"].append(prec1)
        list_metrics["brier"].append(brier_score)
        list_metrics["misent"].append(mis_ent)
        list_metrics["loss"].append(loss_val)
        directory = "runs/office_best/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        if (prec1 > prec1_best):
            prec1_best = prec1
            prec_cls_best = prec_pcls
            loss_best = loss_val
            mis_ent_best = mis_ent
            brier_best = brier_score
            best_epoch = epoch
            torch.save(model_alpha.state_dict(), directory + "drst_" + path_name + "_alpha.pth.tar")
            torch.save(model_beta.state_dict(), directory + "drst_" + path_name + "_beta.pth.tar")
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))

        convert_idx, convert_label = pseudo_labeling_cbst(pred_logit_list, epoch)

        random_seed += 1
        office_train_set = datasets.ImageFolder(root=path_dic[source], transform=data_transforms["train"])
        pseudo_array = torch.zeros((1, 3, 224, 224))
        for idx in convert_idx:
            input = val_data[int(idx)][0].reshape((1, 3, 224, 224))
            pseudo_array = torch.cat((pseudo_array, input))
        pseudo_array = pseudo_array[1:]
        office_valset_pseudo = pseudo_dataset(pseudo_array, convert_label.astype(np.int))
        mix_trainset = torch.utils.data.ConcatDataset([office_train_set, office_valset_pseudo])
        mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)
        model_alpha, model_beta, scheduler_alpha, scheduler_beta = train_one_epoch(mix_train_loader, val_loader,
                                                                                   model_alpha, model_beta,
                                                                                   optimizer_alpha, optimizer_beta,
                                                                                   scheduler_alpha, scheduler_beta,
                                                                                   epoch)

        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print("Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(prec1, prec1_best, loss_best, mis_ent_best, brier_best))
        print("Class specific acc: ", prec_cls_best)
    np.save("log/office_drst" + path_name + ".npy", list_metrics)

def train_and_val_rescue_new(source, target, path_name):
    # Do self-training with new parametric form, use different label selection criterions
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    kwargs = {'num_workers': 0, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    model_alpha = alpha_office_new(CONFIG["num_class"])
    model_beta = beta_office_new()
    optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    #optimizer_alpha = torch.optim.Adam(model_alpha.parameters(), lr=1e-5)
    #optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=1e-5)
    scheduler_alpha = lr_scheduler.StepLR(optimizer_alpha, step_size=7, gamma=0.1)
    scheduler_beta = lr_scheduler.StepLR(optimizer_beta, step_size=7, gamma=0.1)
    random_seed = CONFIG["rand_seed"]
    prec1_best, prec_cls_best, loss_best, mis_ent_best, brier_best, best_epoch = 0, 0, 0, 0, 0, 0

    ## load checkpoint
    resume_path = "crst-office/code/base_model/"+source[0]+"2"+target[0]+"/epoch_200_checkpoint.pth.tar"
    checkpoint = torch.load(resume_path, map_location='cuda:0')
    state = model_alpha.state_dict()
    checkpoint = checkpoint["state_dict"]
    for key in list(checkpoint.keys()):
        checkpoint["model." + key] = checkpoint.pop(key)
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        elif key == "final_layer.weight":
            state[key] = checkpoint["model.fc.0.weight"]
        elif key == "final_layer.bias":
            state[key] = checkpoint["model.fc.0.bias"]
        else:
            print("Param key {} not loaded correctly")
            raise ValueError("Parameter load error")
    model_alpha.load_state_dict(state, strict=True)
    print("=> loaded checkpoint from '{}'".format(resume_path))
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(CONFIG["epochs"]):
        val_data = datasets.ImageFolder(root=path_dic[target], transform=data_transforms["val"])
        # do not shuffle
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, **kwargs)

        prec1, prec_pcls, loss_val, mis_ent, brier_score, pred_logit_list = validate(val_loader, model_alpha, model_beta)

        list_metrics["acc"].append(prec1)
        list_metrics["brier"].append(brier_score)
        list_metrics["misent"].append(mis_ent)
        list_metrics["loss"].append(loss_val)
        directory = "runs/office_best/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        if (prec1 > prec1_best):
            prec1_best = prec1
            prec_cls_best = prec_pcls
            loss_best = loss_val
            mis_ent_best = mis_ent
            brier_best = brier_score
            best_epoch = epoch
            torch.save(model_alpha.state_dict(), directory + "aw_alpha_new.pth.tar")
            torch.save(model_beta.state_dict(), directory + "aw_beta_new.pth.tar")
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))

        convert_idx, convert_label = pseudo_labeling_cbst(pred_logit_list, epoch)

        random_seed += 1
        office_train_set = datasets.ImageFolder(root=path_dic[source], transform=data_transforms["train"])
        pseudo_array = torch.zeros((1, 3, 224, 224))
        for idx in convert_idx:
            input = val_data[int(idx)][0].reshape((1, 3, 224, 224))
            pseudo_array = torch.cat((pseudo_array, input))
        pseudo_array = pseudo_array[1:]
        office_valset_pseudo = pseudo_dataset(pseudo_array, convert_label.astype(np.int))
        mix_trainset = torch.utils.data.ConcatDataset([office_train_set, office_valset_pseudo])
        mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)
        model_alpha, model_beta, scheduler_alpha, scheduler_beta = train_one_epoch(mix_train_loader, val_loader,
                                                                                   model_alpha, model_beta,
                                                                                   optimizer_alpha, optimizer_beta,
                                                                                   scheduler_alpha, scheduler_beta,
                                                                                   epoch)

        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print("Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(prec1, prec1_best, loss_best, mis_ent_best, brier_best))
        print("Class specific acc: ", prec_cls_best)
    np.save("log/office_" + path_name + "_new.npy", list_metrics)

def drl_boost(source, target, resume_path, save_path):
    color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),
            transforms.RandomApply([color_aug], p=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    kwargs = {'num_workers': 0, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    model_alpha = alpha_office(args.num_classes)
    model_beta = beta_office()
    #model_alpha = alpha_office_new(args.num_classes)
    #model_beta = beta_office_new()
    # office31
    optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    #office-home
    #optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)
    #optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler_alpha = lr_scheduler.StepLR(optimizer_alpha, step_size=7, gamma=0.1)
    scheduler_beta = lr_scheduler.StepLR(optimizer_beta, step_size=7, gamma=0.1)
    random_seed = CONFIG["rand_seed"]
    prec1_best, prec_cls_best, loss_best, mis_ent_best, brier_best, best_epoch = 0, 0, 0, 0, 0, 0

    ## load checkpoint
    """
    checkpoint = torch.load(resume_path, map_location='cuda:0')
    state = model_alpha.state_dict()
    checkpoint = checkpoint["state_dict"]
    for key in list(checkpoint.keys()):
        checkpoint["model." + key] = checkpoint.pop(key)
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        elif key == "final_layer.weight":
            state[key] = checkpoint["model.fc.0.weight"]
        elif key == "final_layer.bias":
            state[key] = checkpoint["model.fc.0.bias"]
        else:
            print("Param key {} not loaded correctly")
            raise ValueError("Parameter load error")
    """
    checkpoint = torch.load(resume_path)
    state = model_alpha.state_dict()
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        elif key == "final_layer.weight":
            state[key] = checkpoint["model.fc.2.weight"]
        elif key == "final_layer.bias":
            state[key] = checkpoint["model.fc.2.bias"]
        else:
            print("Param key {} not loaded correctly")
            raise ValueError("Parameter load error")

    model_alpha.load_state_dict(state, strict=True)
    print("=> loaded checkpoint from '{}'".format(resume_path))
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(20):
        # office31
        train_loader, val_loader = dataloader_office(path_dic[source], path_dic[target])
        # officehome
        #train_loader, val_loader = dataloader_office(path_dic_home[source], path_dic_home[target])
        prec1, prec_pcls, loss_val, mis_ent, brier_score, pred_logit_list = validate(val_loader, model_alpha,
                                                                                     model_beta)

        list_metrics["acc"].append(prec1)
        list_metrics["brier"].append(brier_score)
        list_metrics["misent"].append(mis_ent)
        list_metrics["loss"].append(loss_val)
        directory = "runs/office_best/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        if (prec1 > prec1_best):
            prec1_best = prec1
            prec_cls_best = prec_pcls
            loss_best = loss_val
            mis_ent_best = mis_ent
            brier_best = brier_score
            best_epoch = epoch
            torch.save(model_alpha.state_dict(), save_path + "_drl_alpha.pth.tar")
            torch.save(model_beta.state_dict(), save_path + "_drl_beta.pth.tar")
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))

        model_alpha, model_beta, scheduler_alpha, scheduler_beta = train_one_epoch(train_loader, val_loader,
                                                                                   model_alpha, model_beta,
                                                                                   optimizer_alpha, optimizer_beta,
                                                                                   scheduler_alpha, scheduler_beta,
                                                                                   epoch)

        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print("Class specific acc: ", prec_cls_best)
    return prec1_best, brier_best, mis_ent_best, list_metrics

def drst_boost(source, target, resume_path, save_path):
    # Do self-training with new parametric form, use different label selection criterions
    color_aug1 = transforms.ColorJitter(brightness=0.5)
    color_aug2 = transforms.ColorJitter(contrast=0.5)
    color_aug3 = transforms.ColorJitter(saturation=0.5)
    color_aug4 = transforms.ColorJitter(hue=0.5)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),

            #transforms.RandomVerticalFlip(),
            #transforms.RandomChoice([color_aug1, color_aug2, color_aug3, color_aug4]),
            #transforms.RandomGrayscale(p=0.1),
            #transforms.RandomRotation(45, resample=False, expand=False, center=None),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),

            #transforms.RandomVerticalFlip(),
            #transforms.RandomChoice([color_aug1, color_aug2, color_aug3, color_aug4]),
            #transforms.RandomGrayscale(p=0.1),
            #transforms.RandomRotation(45, resample=False, expand=False, center=None),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    kwargs = {'num_workers': 0, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    model_alpha = alpha_office_new(CONFIG["num_class"])
    model_beta = beta_office_new()

    optimizer_alpha = torch.optim.SGD([
        {'params': model_alpha.model.parameters()},
        {'params': model_alpha.final_layer.parameters(), 'lr': 1e-5}], lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # optimizer_alpha = torch.optim.Adam(model_alpha.parameters(), lr=1e-5)
    # optimizer_beta = torch.optim.Adam(model_beta.parameters(), lr=1e-5)
    scheduler_alpha = lr_scheduler.StepLR(optimizer_alpha, step_size=7, gamma=0.1)
    scheduler_beta = lr_scheduler.StepLR(optimizer_beta, step_size=7, gamma=0.1)
    random_seed = CONFIG["rand_seed"]
    prec1_best, prec_cls_best, loss_best, mis_ent_best, brier_best, best_epoch = 0, 0, 0, 0, 0, 0

    ## load checkpoint

    checkpoint = torch.load(resume_path, map_location='cuda:0')
    checkpoint = checkpoint["state_dict"]
    state = model_alpha.state_dict()
    for key in list(checkpoint.keys()):
        checkpoint["model." + key] = checkpoint.pop(key)
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        elif key == "final_layer.weight":
            state[key] = checkpoint["model.fc.0.weight"]
        elif key == "final_layer.bias":
            state[key] = checkpoint["model.fc.0.bias"]
        else:
            print("Param key {} not loaded correctly")
            raise ValueError("Parameter load error")
    """
    checkpoint = torch.load(resume_path)
    state = model_alpha.state_dict()
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        elif key == "final_layer.weight":
            state[key] = checkpoint["model.fc.weight"]
        elif key == "final_layer.bias":
            state[key] = checkpoint["model.fc.bias"]
        else:
            print("Param key {} not loaded correctly")
            raise ValueError("Parameter load error")
    """
    model_alpha.load_state_dict(state, strict=True)
    print("=> loaded checkpoint from '{}'".format(resume_path))
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(25):
        val_data = datasets.ImageFolder(root=path_dic[target], transform=data_transforms["val"])
        # do not shuffle
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, **kwargs)

        prec1, prec_pcls, loss_val, mis_ent, brier_score, pred_logit_list = validate(val_loader, model_alpha,
                                                                                     model_beta)

        list_metrics["acc"].append(prec1)
        list_metrics["brier"].append(brier_score)
        list_metrics["misent"].append(mis_ent)
        list_metrics["loss"].append(loss_val)
        directory = "runs/office_best/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        if (prec1 > prec1_best):
            prec1_best = prec1
            prec_cls_best = prec_pcls
            loss_best = loss_val
            mis_ent_best = mis_ent
            brier_best = brier_score
            best_epoch = epoch
            torch.save(model_alpha.state_dict(), save_path+"_alpha_drst.pth.tar")
            torch.save(model_beta.state_dict(), save_path+"_beta_drst.pth.tar")
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))

        convert_idx, convert_label = pseudo_labeling_cbst(pred_logit_list, epoch)

        random_seed += 1
        office_train_set = datasets.ImageFolder(root=path_dic[source], transform=data_transforms["train"])
        pseudo_array = torch.zeros((1, 3, 224, 224))
        for idx in convert_idx:
            input = val_data[int(idx)][0].reshape((1, 3, 224, 224))
            pseudo_array = torch.cat((pseudo_array, input))
        pseudo_array = pseudo_array[1:]
        office_valset_pseudo = pseudo_dataset(pseudo_array, convert_label.astype(np.int))
        mix_trainset = torch.utils.data.ConcatDataset([office_train_set, office_valset_pseudo])
        mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=CONFIG["batch_size"], shuffle=True,
                                                       **kwargs)
        model_alpha, model_beta, scheduler_alpha, scheduler_beta = train_one_epoch(mix_train_loader, val_loader,
                                                                                   model_alpha, model_beta,
                                                                                   optimizer_alpha, optimizer_beta,
                                                                                   scheduler_alpha, scheduler_beta,
                                                                                   epoch)

        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print("Class specific acc: ", prec_cls_best)
    np.save("log/office_" + source[0]+target[0] + "drst.npy", list_metrics)

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__=="__main__":
    source = args.src # e.g. "amazon"
    target = args.tgt # e.g. "webcam"

    print ('Using device:', DEVICE)
    if not os.path.exists("log/rebuttal/"):
        os.mkdir("log/rebuttal/")

    path_name = source[0] + target[0]
    seed_torch(CONFIG["rand_seed"])

    path_numpy = "log/office_" + path_name + "_new.npy"
    path_model = "runs/office_iid/" + path_name
    train_and_val_iid_new(path_dic[source], path_dic[target], save_path_numpy=path_numpy, save_path_model=path_model)

    train_and_val_rescue(source, target, path_name)
    #train_and_val_rescue_new(source, target, path_name)
    drl_boost(source, target, "crst-office/code/runs/ResNet50-aug_" + source[0] + "2" + target[0] + "/model_best.pth.tar",
              "runs/office_best/" + source[0] + "2" + target[0])
    drst_boost(source, target,
               "crst-office/code/runs/ResNet50-aug_" + source[0] + "2" + target[0] + "/model_best.pth.tar",
               "runs/office_best/" + source[0] + "2" + target[0])

    path_numpy = "log/office_" + path_name + ".npy"
    path_model = "runs/office_iid/" + path_name
    train_and_val_iid(path_dic[source], path_dic[target], save_path_numpy=path_numpy, save_path_model=path_model)
    #train_and_val_iid(path_dic_home[source], path_dic_home[target], save_path_numpy=path_numpy, save_path_model=path_model)
    save_path = "runs/office_best/" + path_name

    list_of_list = []
    prec_list, brier_list, misent_list = [], [], []
    best_prec = 0
    for i in range(20):
        print("Training for checkpoint epoch of {}".format(i))
        resume_path = path_model + "_" + str(i) + ".pth.tar"
        prec1, brier, misent, list_metrics = drl_boost(source, target, resume_path, save_path)
        prec_list.append(prec1)
        brier_list.append(brier)
        misent_list.append(misent)
        list_of_list.append(list_metrics)
        np.save("log/rebuttal/drl_prec_"+ path_name + ".npy", prec_list)
        np.save("log/rebuttal/drl_brier" + path_name + ".npy", brier_list)
        np.save("log/rebuttal/drl_misent" + path_name + ".npy", misent_list)
        np.save("log/rebuttal/drl_list" + path_name + ".npy", list_of_list)
        if prec1 > best_prec:
            best_prec = prec1
    print("Best precision for task {} is {}".format(path_name, best_prec))