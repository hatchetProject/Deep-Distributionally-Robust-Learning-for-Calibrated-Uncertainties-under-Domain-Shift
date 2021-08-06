"""
Evaluate and compare on samples that have lower density ratio
"""
import numpy as np
import torch
import math
from PIL import Image
import torch.nn as nn
import torchvision
from sklearn.preprocessing import OneHotEncoder
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
from torchvision import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    "n_classes": 12,
    "batch_size": 16,
    "src_gt_list": "visda/train/image_list.txt",
    "src_train_list": "visda/train/image_list_train.txt",
    "src_root": "visda/train/",
    "tgt_gt_list": "visda/validation/image_list.txt",
    "tgt_train_list": "visda/validation/image_list_train.txt",
    "tgt_root": "visda/validation/",
    "name": "res101_visda_used",
    "lr": 1e-5,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_class": 12,
    "print_freq": 100,
    "epochs": 20,
    "rand_seed": 0,
    "src_portion": 0.065, # orig 0.065
    "src_portion_step": 0.0085,
    "src_portion_max": 0.165,
}

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def test_hard_samples():
    from visda_exp import CONFIG, ImageClassdata, avh_score, accuracy_new, entropy
    from visda_exp import source_vis, beta_vis, alpha_vis, validate, AverageMeter
    import math
    from sklearn.metrics import brier_score_loss
    import torch.nn as nn
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
    kwargs = {'num_workers': 1, 'pin_memory': False}
    visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                    transform=data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True,
                                             **kwargs)

    # DRST
    #resume_path = "runs/best_model_gradcam/"
    resume_path = "runs/best_model_comp13/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    #checkpoint_alpha = torch.load(resume_path + "alpha_epoch_12.pth.tar")
    #checkpoint_beta = torch.load(resume_path + "beta_epoch_12.pth.tar")
    checkpoint_alpha = torch.load(resume_path + "alpha_best.pth.tar")
    checkpoint_beta = torch.load(resume_path + "beta_best.pth.tar")
    model_alpha = alpha_vis(12)
    model_beta = beta_vis()
    model_alpha.load_state_dict(checkpoint_alpha)
    model_beta.load_state_dict(checkpoint_beta)

    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    model_alpha.eval()
    model_beta.eval()

    # CBST + ASG model
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/cbst_model_gradcam/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path + "alpha_epoch_12.pth.tar")
    state = model.state_dict()
    for key in state.keys():
        if "model." + key in checkpoint.keys():
            state[key] = checkpoint["model." + key]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()

    # CRST
    model_crst = models.resnet101(pretrained=False)
    num_ftrs = model_crst.fc.in_features
    model_crst.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/crst/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path + "model_best.pth.tar")
    model_crst.load_state_dict(checkpoint["state_dict"])
    model_crst = model_crst.to(DEVICE)
    model_crst.eval()

    r_list = np.zeros((1, 1))
    drst_list = np.zeros((1, CONFIG["n_classes"]))
    cbst_list = np.zeros((1, CONFIG["n_classes"]))
    crst_list = np.zeros((1, CONFIG["n_classes"]))
    label_list = np.zeros((1, 1))
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            BATCH_SIZE = input.shape[0]
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).to(DEVICE)
            pred2 = F.softmax(model_beta(fliplr(input), None, None, None, None).detach(), dim=1).to(DEVICE)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            r_target2 = (pred2[:, 0] / pred2[:, 1]).reshape(-1, 1)
            # Add flipping
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(),
                                     r_target.cuda()).detach()
            target_out2 = model_alpha(fliplr(input), torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(),
                                      r_target2.cuda()).detach()
            drst_predict = (F.softmax(target_out, dim=1) + F.softmax(target_out2, dim=1)) / 2
            r_list = np.concatenate((r_list, (r_target + r_target2) / 2), axis=0)
            cbst_predict = (F.softmax(model(input), dim=1) + F.softmax(model(fliplr(input)), dim=1)) / 2
            crst_predict = (F.softmax(model_crst(input), dim=1) + F.softmax(model_crst(fliplr(input)), dim=1)) / 2
            drst_list = np.concatenate((drst_list, drst_predict.detach().cpu().numpy()), axis=0)
            cbst_list = np.concatenate((cbst_list, cbst_predict.detach().cpu().numpy()), axis=0)
            crst_list = np.concatenate((crst_list, crst_predict.detach().cpu().numpy()), axis=0)
            label_list = np.concatenate((label_list, label.detach().cpu().numpy()), axis=0)
            if i % 1000 == 0:
                print("1000 samples processed")
    print("Validation finished")
    r_list = r_list[1:]
    drst_list = drst_list[1:]
    cbst_list = cbst_list[1:]
    crst_list = crst_list[1:]
    label_list = label_list[1:]
    r_list = r_list.reshape(-1, )
    label_list = label_list.reshape(-1, )
    label_list = label_list.astype(np.int)
    drst_list = np.argmax(drst_list, axis=1)
    cbst_list = np.argmax(cbst_list, axis=1)
    crst_list = np.argmax(crst_list, axis=1)

    density_list = []
    drst_acc_list = []
    cbst_acc_list = []
    crst_acc_list = []
    percentage = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for num in percentage:
        threshold = np.sort(r_list)[int(r_list.shape[0]*num - 1)]
        density_list.append(threshold)
        idx = np.where(r_list <= threshold)[0]
        num_hard = idx.shape[0]
        label_list_p = label_list[idx]
        drst_list_p = drst_list[idx]
        cbst_list_p = cbst_list[idx]
        crst_list_p = crst_list[idx]

        drst_acc = float(np.sum(label_list_p == drst_list_p)) / num_hard
        cbst_acc = float(np.sum(label_list_p == cbst_list_p)) / num_hard
        crst_acc = float(np.sum(label_list_p == crst_list_p)) / num_hard
        drst_acc_list.append(drst_acc)
        cbst_acc_list.append(cbst_acc)
        crst_acc_list.append(crst_acc)
        print("Validated on {} samples, DRST acc: {}, CBST acc: {}, CRST acc: {}, density rato: {}".format(
            num_hard, drst_acc, cbst_acc, crst_acc, threshold))

    print(density_list)
    print(drst_acc_list)
    print(cbst_acc_list)
    print(crst_acc_list)

def test_hard_samples_torch():
    from visda_exp import CONFIG, ImageClassdata, avh_score, accuracy_new, entropy
    from visda_exp import source_vis, beta_vis, alpha_vis, validate, AverageMeter
    import math
    from sklearn.metrics import brier_score_loss
    import torch.nn as nn
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
    kwargs = {'num_workers': 1, 'pin_memory': False}
    visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                    transform=data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True,
                                             **kwargs)

    # DRST
    #resume_path = "runs/best_model_gradcam/"
    resume_path = "runs/best_model_comp13/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    #checkpoint_alpha = torch.load(resume_path + "alpha_epoch_12.pth.tar")
    #checkpoint_beta = torch.load(resume_path + "beta_epoch_12.pth.tar")
    checkpoint_alpha = torch.load(resume_path + "alpha_best.pth.tar")
    checkpoint_beta = torch.load(resume_path + "beta_best.pth.tar")
    model_alpha = alpha_vis(12)
    model_beta = beta_vis()
    model_alpha.load_state_dict(checkpoint_alpha)
    model_beta.load_state_dict(checkpoint_beta)

    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    model_alpha.eval()
    model_beta.eval()

    # CBST + ASG model
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/cbst_model_gradcam/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path + "alpha_epoch_12.pth.tar")
    state = model.state_dict()
    for key in state.keys():
        if "model." + key in checkpoint.keys():
            state[key] = checkpoint["model." + key]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()

    # CRST
    model_crst = models.resnet101(pretrained=False)
    num_ftrs = model_crst.fc.in_features
    model_crst.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/crst/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path + "model_best.pth.tar")
    model_crst.load_state_dict(checkpoint["state_dict"])
    model_crst = model_crst.to(DEVICE)
    model_crst.eval()

    r_list = torch.zeros((1, 1))
    drst_list = torch.zeros((1, CONFIG["n_classes"]))
    cbst_list = torch.zeros((1, CONFIG["n_classes"]))
    crst_list = torch.zeros((1, CONFIG["n_classes"]))
    label_list = torch.zeros((1, 1))
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            BATCH_SIZE = input.shape[0]
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).to(DEVICE)
            pred2 = F.softmax(model_beta(fliplr(input), None, None, None, None).detach(), dim=1).to(DEVICE)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            r_target2 = (pred2[:, 0] / pred2[:, 1]).reshape(-1, 1)
            # Add flipping
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(),
                                     r_target.cuda()).detach()
            target_out2 = model_alpha(fliplr(input), torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(),
                                      r_target2.cuda()).detach()
            drst_predict = (F.softmax(target_out, dim=1) + F.softmax(target_out2, dim=1)) / 2
            r_list = np.concatenate((r_list, (r_target + r_target2) / 2), axis=0)
            cbst_predict = (F.softmax(model(input), dim=1) + F.softmax(model(fliplr(input)), dim=1)) / 2
            crst_predict = (F.softmax(model_crst(input), dim=1) + F.softmax(model_crst(fliplr(input)), dim=1)) / 2
            drst_list = torch.cat((drst_list, drst_predict.detach().cpu()), dim=0)
            cbst_list = torch.cat((cbst_list, cbst_predict.detach().cpu()), dim=0)
            crst_list = torch.cat((crst_list, crst_predict.detach().cpu()), dim=0)
            label_list = torch.cat((label_list, label.detach().cpu()), dim=0)
            if i % 1000 == 0:
                print("1000 samples processed")

    print("Validation finished")
    r_list = r_list[1:]
    drst_list = drst_list[1:]
    cbst_list = cbst_list[1:]
    crst_list = crst_list[1:]
    label_list = label_list[1:]
    r_list = r_list.reshape(-1, )
    label_list = label_list.reshape(-1, )
    #drst_list = torch.argmax(drst_list, dim=1)
    #cbst_list = torch.argmax(cbst_list, dim=1)
    #crst_list = torch.argmax(crst_list, dim=1)

    density_list = []
    drst_acc_list = []
    cbst_acc_list = []
    crst_acc_list = []
    acc_drst = AverageMeter()
    acc_cbst = AverageMeter()
    acc_crst = AverageMeter()
    percentage = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for num in percentage:
        acc_drst.reset()
        acc_cbst.reset()
        acc_crst.reset()
        threshold = np.sort(r_list)[int(r_list.shape[0]*num - 1)]
        density_list.append(threshold)
        idx = np.where(r_list <= threshold)[0]
        num_hard = idx.shape[0]
        label_list_p = label_list[idx]
        drst_list_p = drst_list[idx]
        cbst_list_p = cbst_list[idx]
        crst_list_p = crst_list[idx]

        prec1, gt_num = accuracy_new(drst_list_p, label_list_p.long(), CONFIG["num_class"], topk=(1,))
        acc_drst.update(prec1[0], gt_num[0])
        prec1, gt_num = accuracy_new(cbst_list_p, label_list_p.long(), CONFIG["num_class"], topk=(1,))
        acc_cbst.update(prec1[0], gt_num[0])
        prec1, gt_num = accuracy_new(crst_list_p, label_list_p.long(), CONFIG["num_class"], topk=(1,))
        acc_crst.update(prec1[0], gt_num[0])

        drst_acc_list.append(acc_drst.vec2sca_avg.numpy().reshape(-1, )[0])
        cbst_acc_list.append(acc_cbst.vec2sca_avg.numpy().reshape(-1, )[0])
        crst_acc_list.append(acc_crst.vec2sca_avg.numpy().reshape(-1, )[0])
        print("Validated on {} samples, DRST acc: {}, CBST acc: {}, CRST acc: {}, density rato: {}".format(
            num_hard, acc_drst.vec2sca_avg.numpy(), acc_cbst.vec2sca_avg.numpy(), acc_crst.vec2sca_avg.numpy(), threshold))

    print(density_list)
    print(drst_acc_list)
    print(cbst_acc_list)
    print(crst_acc_list)

if __name__=="__main__":
    test_hard_samples()
    test_hard_samples_torch()