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

def plot_graph():
    import matplotlib.pyplot as plt
    #dr = [0.9547328948974609, 0.984018087387085, 1.0321789979934692, 1.0761606693267822, 1.104475975036621,
    #      1.1275405883789062, 1.1946954727172852, 1.2470722198486328, 1.2937593460083008, 1.3402671813964844,
    #      1.388777256011963, 1.4432116746902466, 1.5119720697402954, 1.6150619983673096, 2.6912431716918945]
    dr = [0.8906536, 0.9547329, 0.9840181, 1.032179, 1.0761607, 1.104476, 1.1275406, 1.1946955, 1.2470722,
          1.2937593, 1.3402672, 1.3887773, 1.4432117, 1.5119721, 1.615062, 2.6912432]
    dr = dr[2:14]
    #dr = [str(x)[:5] for x in dr]
    dr = [round(x, 5) for x in dr]
    #drst_acc = [0.8985507246376812, 0.8878842676311031, 0.8598265895953757, 0.838569880823402, 0.8341357727491574, 0.8324304803178043, 0.8258553760043333, 0.8272147327876745, 0.8305122997066126, 0.8332851881273922, 0.8339552238805971, 0.8351087152768822, 0.8362220717670955, 0.8372886116070533, 0.8360475193182639]
    #cbst_acc = [0.8478260869565217, 0.8481012658227848, 0.8352601156069365, 0.819068255687974, 0.8151179585941262, 0.8129288551823763, 0.8090638259456532, 0.8083172845450168, 0.8097043556759197, 0.8115115187405214, 0.8127106403466539, 0.8138299244280519, 0.8153464229293613, 0.8160043330859195, 0.8144543944536723]
    #crst_acc = [0.8768115942028986, 0.8625678119349005, 0.8424855491329479, 0.8277356446370531, 0.8269138180067405, 0.8264716504153123, 0.823327615780446, 0.8241454020221474, 0.8266756939742722, 0.8302520401531017, 0.8312770823302841, 0.8332516571664388, 0.8353193410065448, 0.8371080663604085, 0.8366433162417852]
    drst_acc = np.array([62.61574, 84.96778, 84.06623, 80.29194, 81.23406, 80.68415, 81.655365, 82.800934, 83.7922, 84.61667, 85.16906, 85.38248, 85.59379, 85.688484, 85.80006, 85.5503])
    cbst_acc = np.array([54.28241, 77.12081, 78.180916, 75.85826, 77.36738, 76.980675, 78.1996, 80.04111, 80.77235, 81.5319, 82.0073, 82.34723, 82.61672, 82.831924, 82.9345, 82.6907])
    crst_acc = np.array([61.574078, 81.151764, 78.36051, 74.81897, 76.52979, 77.13705, 78.52777, 80.53998, 81.66393, 82.58494, 83.287445, 83.660706, 84.05231, 84.3411, 84.6109, 84.48637])
    drst_acc = drst_acc[2:14]
    cbst_acc = cbst_acc[2:14]
    crst_acc = crst_acc[2:14]
    plt.figure(figsize=(15, 10))
    plt.plot(dr[1:], drst_acc[1:], c='red', linestyle='-', linewidth=6, label='DRST')
    plt.plot(dr[1:], cbst_acc[1:], c='blue', linestyle='-', linewidth=6, label='CBST')
    plt.plot(dr[1:], crst_acc[1:], c='green', linestyle='-', linewidth=6, label="CRST")
    #dr_2 = [2.69124, 1.61506, 1.51197, 1.44321, 1.38878, 1.34027, 1.29376, 1.24707, 1.1947, 1.12754, 1.10448, 1.07616, 1.03218, 0.98402]
    #cb_imp = [2.8596, 2.86556, 2.85656, 2.97707, 3.03525, 3.16176, 3.08477, 3.01985, 2.759824, 3.455765, 3.703475, 3.86668, 4.43368, 5.885314]
    #cr_imp = [1.06393, 1.18916, 1.347384, 1.54148, 1.721774, 1.881615, 2.03173, 2.12827, 2.260954, 3.127595, 3.5471, 4.70427, 5.47297, 5.70572]
    #dr_2 = [round(x, 5) for x in dr_2]
    #plt.plot(dr_2, cb_imp, c='red', linewidth=7, label='Margin over CBST')
    #plt.plot(dr_2, cr_imp, c='blue', linewidth=7, label='Margin over CRST')
    #plt.gca().invert_xaxis()
    plt.tick_params(labelsize=28)
    plt.xlabel("Density ratio threshold", fontdict={"weight": "normal", "size": 36})
    plt.ylabel("Accuracy", fontdict={"weight": "normal", "size": 36})
    plt.legend(fontsize=36)
    plt.title("Accuracy value with different density ratio", fontdict={"weight": "normal", "size": 38})
    #plt.title("DRST outperforms baselines", fontdict={"weight": "normal", "size": 40})
    plt.savefig("log/finals/dr_acc_correct_6.jpg")

if __name__=="__main__":
    #test_hard_samples()
    plot_graph()
    #test_hard_samples_torch()