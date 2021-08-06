"""
Train on Visda 2017 dataset using ResNet101 backbone
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
import argparse

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
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
    "rand_seed": 5,
    "src_portion": 0.065, # orig 0.065
    "src_portion_step": 0.0085,
    "src_portion_max": 0.165,
}

class source_vis(nn.Module):
    def __init__(self, n_output):
        super(source_vis, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
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

class alpha_vis(nn.Module):
    def __init__(self, n_output):
        super(alpha_vis, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
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

class beta_vis(nn.Module):
    def __init__(self):
        super(beta_vis, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
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

class ImageClassdata(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, txt_file, root_dir, transform=transforms.ToTensor()):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_frame = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        ImgName_Label = str.split(self.images_frame[idx])
        img_name = os.path.join(self.root_dir, ImgName_Label[0])
        img = Image.open(img_name)
        image = img.convert('RGB')
        lbl = np.asarray(ImgName_Label[1:],dtype=np.float32)
        label = torch.from_numpy(lbl)

        if self.transform:
            image = self.transform(image)

        return image, label, ImgName_Label[0]

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

def entropy(p):
    p[p<1e-20] = 1e-20
    return -torch.sum(p.mul(torch.log2(p)))

def avh_score(x, w):
    """
    Actually computes the AVC score for a single sample;
    AVH score is used to replace the prediction probability
    x with shape (1, num_features), w with shape (num_features, n_classes)
    :return: avh score of a single sample, with type float
    """
    avc_score = np.pi - np.arccos(np.dot(x, w.transpose())/(np.linalg.norm(x)*np.linalg.norm(w)))
    avc_score = avc_score / np.sum(avc_score)
    return avc_score

def dataloader_visda():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
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

    kwargs = {'num_workers': 1, 'pin_memory': True}
    visDA17_trainset = ImageClassdata(txt_file=CONFIG["src_gt_list"], root_dir=CONFIG["src_root"],
                                      transform=data_transforms['train'])
    # For UDA, validation set is test set
    visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                    transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(visDA17_trainset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)

    return train_loader, val_loader

def selftrain_dataloader():
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
    kwargs = {'num_workers': 1, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    # Currently we use the whole validation set, with randomly sampled train set
    # We guarantee that the train set smaller than the validation set: Train set: 152397; Val set: 55388
    # Here: src_train_list is used instead of src_gt_list
    visDA17_trainset = ImageClassdata(txt_file=CONFIG["src_train_list"], root_dir=CONFIG["src_root"],
                                      transform=data_transforms['train'])
    # For UDA, validation set is test set
    visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                    transform=data_transforms['val'])

    train_loader = torch.utils.data.DataLoader(visDA17_trainset, batch_size=CONFIG["batch_size"], shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)
    return train_loader, val_loader

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

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def saveSRCtxt(src_portion, randseed):
    src_gt_txt = CONFIG["src_gt_list"]
    src_train_txt = CONFIG["src_train_list"]
    item_list = []
    count0 = 0
    with open(src_gt_txt,"rt") as f:
        for item in f.readlines():
            fields = item.strip()
            item_list.append(fields)
            count0 = count0 + 1

    num_source = count0
    num_sel_source = int(np.floor(num_source*src_portion))
    np.random.seed(randseed)
    print("Number of total source samples:", num_source)
    print("Number of used source samples in each epoch:", num_sel_source)
    sel_idx = list(np.random.choice(num_source, num_sel_source, replace=False))
    item_list = list(itemgetter(*sel_idx)(item_list))

    with open(src_train_txt, 'w') as f:
        for item in item_list:
            f.write("%s\n" % item)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(CONFIG["name"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'epoch_'+str(state['epoch']) + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(CONFIG["name"]) + 'model_best.pth.tar')

def train_and_val_rescue_var3():
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
    kwargs = {'num_workers': 1, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    model_alpha = alpha_vis(12)
    model_beta = beta_vis()
    optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer_beta = torch.optim.SGD(model_beta.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=5e-4)

    scheduler_alpha = lr_scheduler.StepLR(optimizer_alpha, step_size=7, gamma=0.1)
    scheduler_beta = lr_scheduler.StepLR(optimizer_beta, step_size=7, gamma=0.1)
    random_seed = CONFIG["rand_seed"]
    prec1_best, prec_cls_best, loss_best, mis_ent_best, brier_best, best_epoch = 0, 0, 0, 0, 0, 0
    train_portion = CONFIG["src_portion"]
    """
    resume_path = "runs/res101_asg/res101_vista17_best.pth.tar"
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    print(
        "=> loaded checkpoint '{}' (epoch {}), with starting precision {:.3f}".format(resume_path, checkpoint['epoch'],
                                                                                      best_prec1))
    state_pre = checkpoint["state_dict"]
    for key in list(state_pre.keys()):
        state_pre["model." + key] = state_pre.pop(key)
    state = model_alpha.state_dict()
    for key in state_pre.keys():
        if key in state.keys():
            state[key] = state_pre[key]
        elif key == "model.fc_new.0.weight":
            state["model.fc.0.weight"] = state_pre[key]
        elif key == "model.fc_new.0.bias":
            state["model.fc.0.bias"] = state_pre[key]
        elif key == "model.fc_new.2.weight":
            state["final_layer.weight"] = state_pre[key]
        elif key == "model.fc_new.2.bias":
            state["final_layer.bias"] = state_pre[key]
    model_alpha.load_state_dict(state, strict=True)
    """
    resume_path = "runs/res101_visda_used/model_best.pth.tar"
    checkpoint = torch.load(resume_path)
    print(
        "=> loaded checkpoint '{}' (epoch {}), with starting precision {:.3f}".format(resume_path, checkpoint['epoch'],
                                                                                      checkpoint['best_prec1']))

    state_pre = checkpoint["state_dict"]
    state = model_alpha.state_dict()
    for key in state_pre.keys():
        if key in state.keys():
            state[key] = state_pre[key]
        elif key == "model.fc.2.weight":
            state["final_layer.weight"] = state_pre[key]
        elif key == "model.fc.2.bias":
            state["final_layer.bias"] = state_pre[key]
    model_alpha.load_state_dict(state, strict=True)

    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}
    for epoch in range(CONFIG["epochs"]):
        visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"], transform=data_transforms['val'])
        val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)

        prec1, prec_pcls, loss_val, mis_ent, brier_score, image_name_list, pred_logit_list, label_list = validate(val_loader, model_alpha, model_beta)

        # Extract softmax features
        #extract_feat_drst(model_alpha.state_dict(), epoch)

        list_metrics["acc"].append(prec1)
        list_metrics["brier"].append(brier_score)
        list_metrics["misent"].append(mis_ent)
        list_metrics["loss"].append(loss_val)
        directory = "runs/best_model_comp_new/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if (prec1 > prec1_best):
            prec1_best = prec1
            prec_cls_best = prec_pcls
            loss_best = loss_val
            mis_ent_best = mis_ent
            brier_best = brier_score
            best_epoch = epoch
            #torch.save(model_alpha.state_dict(), directory + "alpha_best.pth.tar")
            #torch.save(model_beta.state_dict(), directory + "beta_best.pth.tar")
            # shutil.copyfile(filename_alpha, 'runs/best_model_0.4_avh/' + 'alpha_best.pth.tar')
            # shutil.copyfile(filename_beta, 'runs/best_model_0.4_avh/' + 'beta_best.pth.tar')
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))

        pseudo_labeling_cbst(pred_logit_list, image_name_list, epoch)

        saveSRCtxt(train_portion, random_seed)
        random_seed += 1
        train_portion = min(train_portion+CONFIG["src_portion_step"], CONFIG["src_portion_max"])

        visDA17_trainset = ImageClassdata(txt_file=CONFIG["src_train_list"], root_dir=CONFIG["src_root"], transform=data_transforms['train'])
        visDA17_valset_pseudo = ImageClassdata(txt_file=CONFIG["tgt_train_list"], root_dir=CONFIG["tgt_root"], transform=data_transforms['val4mix'])
        mix_trainset = torch.utils.data.ConcatDataset([visDA17_trainset, visDA17_valset_pseudo])
        mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)
        model_alpha, model_beta, scheduler_alpha, scheduler_beta = train_one_epoch(mix_train_loader, val_loader,
                                                                                   model_alpha, model_beta,
                                                                                   optimizer_alpha, optimizer_beta,
                                                                                   scheduler_alpha, scheduler_beta,
                                                                                   epoch)

        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))
        print("Class specific acc: ", prec_cls_best)
    #np.save("log/var3_seed4.npy", list_metrics)

def train_and_val_cbst():
    # Do self-training (CBST) with IID model
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
    kwargs = {'num_workers': 1, 'pin_memory': False}  # num_workers = 1 is necessary for reproducible results
    model = source_vis(12)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4,
                                momentum=CONFIG["momentum"], nesterov=True,
                                weight_decay=CONFIG["weight_decay"])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    random_seed = CONFIG["rand_seed"]
    prec1_best, prec_cls_best, loss_best, mis_ent_best, brier_best, best_epoch = 0, 0, 0, 0, 0, 0
    train_portion = CONFIG["src_portion"]
    ## load checkpoint
    resume_path = "runs/res101_asg/res101_vista17_best.pth.tar"
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    state_pre = checkpoint["state_dict"]
    for key in list(state_pre.keys()):
        state_pre["model." + key] = state_pre.pop(key)
    state = model.state_dict()
    for key in state_pre.keys():
        if key in state.keys():
            state[key] = state_pre[key]
        elif key == "model.fc_new.0.weight":
            state["model.fc.0.weight"] = state_pre[key]
        elif key == "model.fc_new.0.bias":
            state["model.fc.0.bias"] = state_pre[key]
        elif key == "model.fc_new.2.weight":
            state["model.fc.2.weight"] = state_pre[key]
        elif key == "model.fc_new.2.bias":
            state["model.fc.2.bias"] = state_pre[key]
    model.load_state_dict(state, strict=True)

    print("=> loaded checkpoint '{}' (epoch {}), with starting precision {:.3f}"
          .format(resume_path, checkpoint['epoch'], best_prec1))
    model = model.to(DEVICE)
    list_metrics = {"acc": [], "misent": [], "brier": [], "loss": []}

    for epoch in range(CONFIG["epochs"]):
        visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                        transform=data_transforms['val'])
        val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True,
                                                 **kwargs)

        prec1, prec_pcls, loss_val, mis_ent, brier_score, image_name_list, pred_logit_list, label_list = validate_cbst(val_loader, model)

        list_metrics["acc"].append(prec1)
        list_metrics["brier"].append(brier_score)
        list_metrics["misent"].append(mis_ent)
        list_metrics["loss"].append(loss_val)
        #directory = "runs/cbst_model/"
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #filename_alpha = directory + "cbst_best.pth.tar"
        # torch.save(model.state_dict(), filename_alpha)
        if (prec1 > prec1_best):
            prec1_best = prec1
            prec_cls_best = prec_pcls
            loss_best = loss_val
            mis_ent_best = mis_ent
            brier_best = brier_score
            best_epoch = epoch
            #torch.save(model.state_dict(), filename_alpha)
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))

        pseudo_labeling_cbst(pred_logit_list, image_name_list, epoch)

        saveSRCtxt(train_portion, random_seed)
        random_seed += 1
        train_portion = min(train_portion + CONFIG["src_portion_step"], CONFIG["src_portion_max"])

        visDA17_trainset = ImageClassdata(txt_file=CONFIG["src_train_list"], root_dir=CONFIG["src_root"],
                                          transform=data_transforms['train'])
        visDA17_valset_pseudo = ImageClassdata(txt_file=CONFIG["tgt_train_list"], root_dir=CONFIG["tgt_root"],
                                               transform=data_transforms['val4mix'])
        mix_trainset = torch.utils.data.ConcatDataset([visDA17_trainset, visDA17_valset_pseudo])
        mix_train_loader = torch.utils.data.DataLoader(mix_trainset, batch_size=CONFIG["batch_size"], shuffle=True, **kwargs)
        model, scheduler = train_cbst(mix_train_loader, model, optimizer, scheduler, epoch)

        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print(
            "Current precision: {:.3f}, best precision achieved: {:.3f}, corresponding loss: {:.3f}, mis ent: {:.3f}, brier: {:.3f}".format(
                prec1, prec1_best, loss_best, mis_ent_best, brier_best))
        print("Class specific acc: ", prec_cls_best)
    np.save("log/visda_cbst_asg4.npy", list_metrics)

def train_one_epoch(train_loader, test_loader, model_alpha, model_beta, optimizer_alpha, optimizer_beta, schedular_alpha, schedular_beta, epoch):
    ## train loader sample number must be smaller than test loader
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    model_alpha.train()
    model_beta.train()
    iter_test = iter(test_loader)
    iter_train = iter(train_loader)
    bce_loss = nn.BCEWithLogitsLoss()
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    ce_loss = nn.CrossEntropyLoss()
    train_loss, train_acc = 0, 0
    for i, (input, label, _) in enumerate(train_loader):
        input_test, _, _ = iter_test.next()
        label_train = label.reshape((-1,))
        label_train = label_train.to(DEVICE)
        input_train = input.to(DEVICE)
        # Adding flipping
        #if np.random.rand() > 0.5:
        #    input_train = fliplr(input_train)
        input_test = input_test.to(DEVICE)
        BATCH_SIZE = input.shape[0]
        input_concat = torch.cat([input_train, input_test], dim=0)
        # this parameter used for softlabling
        label_concat = torch.cat(
            (torch.FloatTensor([1, 0]).repeat(input_train.shape[0], 1), torch.FloatTensor([0, 1]).repeat(input_test.shape[0], 1)), dim=0)
        label_concat = label_concat.to(DEVICE)

        prob = model_beta(input_concat, None, None, None, None)
        assert(F.softmax(prob.detach(), dim=1).cpu().numpy().all()>=0 and F.softmax(prob.detach(), dim=1).cpu().numpy().all()<=1)
        #loss_dis = bce_loss(F.softmax(prob, dim=1), label_concat)
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
        if i < 5 and epoch == 0:
            optimizer_beta.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_beta.step()

        if i < 5 and epoch == 0:
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
                epoch, i, len(train_loader), train_loss, train_acc*100.0))
            train_loss, train_acc = 0, 0
    schedular_alpha.step()
    schedular_beta.step()
    return model_alpha, model_beta, schedular_alpha, schedular_beta

def train_cbst(train_loader, model_alpha, optimizer_alpha, schedular_alpha, epoch):
    ## train loader sample number must be smaller than test loader
    model_alpha.train()
    ce_loss = nn.CrossEntropyLoss()
    train_loss, train_acc = 0, 0
    for i, (input, label, _) in enumerate(train_loader):
        label_train = label.reshape((-1,))
        label_train = label_train.to(DEVICE)
        input_train = input.to(DEVICE)
        BATCH_SIZE = input.shape[0]

        theta_out = model_alpha(input_train)
        source_pred = F.softmax(theta_out, dim=1)
        loss_theta = ce_loss(theta_out, label_train.long())

        # Backpropagate
        optimizer_alpha.zero_grad()
        loss_theta.backward()
        optimizer_alpha.step()

        train_loss += loss_theta
        train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == label_train.long()).float() / BATCH_SIZE
        if i % CONFIG["print_freq"] == 0:
            train_loss = train_loss / (CONFIG["print_freq"] * BATCH_SIZE)
            train_acc = train_acc / (CONFIG["print_freq"])
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss: {3:.4f} \t Train Acc: {4:.4f}'.format(
                epoch, i, len(train_loader), train_loss, train_acc * 100.0))
            train_loss, train_acc = 0, 0
    schedular_alpha.step()
    return model_alpha, schedular_alpha

def validate(test_loader, model_alpha, model_beta):
    # validate model and select samples for self-training
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    model_alpha.eval()
    model_beta.eval()
    top1_acc = AverageMeter()
    losses = AverageMeter()
    ce_loss = nn.CrossEntropyLoss()
    mis_ent, mis_num, brier_score, test_num = 0, 0, 0, 0
    image_name_list = []
    pred_logit_list = []
    label_list = []
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(test_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1, ))
            BATCH_SIZE = input.shape[0]
            test_num += BATCH_SIZE

            # Add flipping
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).to(DEVICE)
            pred2 = F.softmax(model_beta(fliplr(input), None, None, None, None).detach(), dim=1).to(DEVICE)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            r_target2 = (pred2[:, 0] / pred2[:, 1]).reshape(-1, 1)

            # Add flipping
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(), r_target.cuda()).detach()
            target_out2 = model_alpha(fliplr(input), torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(), r_target2.cuda()).detach()
            #prediction_t = F.softmax(target_out, dim=1)
            prediction_t = (F.softmax(target_out, dim=1) + F.softmax(target_out2, dim=1)) / 2

            test_loss = float(ce_loss(target_out, label.long()))
            losses.update(test_loss, BATCH_SIZE)
            prec1, gt_num = accuracy_new(prediction_t, label.long(), CONFIG["num_class"], topk=(1,))
            top1_acc.update(prec1[0], gt_num[0])
            mis_idx = (torch.argmax(prediction_t, dim=1) != label.long()).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            mis_ent += entropy(mis_pred) / math.log(CONFIG["num_class"], 2)
            mis_num += mis_idx.shape[0]
            # one-hot encoding
            label_onehot = torch.zeros(prediction_t.shape)
            label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
            for j in range(input.shape[0]):
                brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j].cpu().numpy())
            if i % CONFIG["print_freq"] == 0:
                print('Test: [{0}/{1}]\n'
                      'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
                      'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
                    i, len(test_loader), loss=losses, top1=top1_acc))
            for idx in range(BATCH_SIZE):
                image_name_list.append(input_name[idx])
                pred_logit_list.append(prediction_t[idx, :])
    brier_score = brier_score/test_num
    misent = mis_ent/mis_num
    print("Mis entropy: {}, Brier score: {}".format(misent, brier_score))

    return top1_acc.vec2sca_avg, top1_acc.avg, losses.avg, misent, brier_score, image_name_list, pred_logit_list, label_list

def validate_avh(test_loader, model_alpha, model_beta):
    # validate model and select samples for self-training, we change softmax scores with avh scores
    # we instead use avh scores for predictions / still adopt softmax?
    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    model_alpha.eval()
    model_beta.eval()
    # get essential params for avh evaluation
    w = None
    for name, param in model_alpha.named_parameters():
        if name == "final_layer.weight":
            w = param.detach().cpu().numpy()
            break

    top1_acc = AverageMeter()
    losses = AverageMeter()
    ce_loss = nn.CrossEntropyLoss()
    mis_ent, mis_num, brier_score, test_num = 0, 0, 0, 0
    image_name_list = []
    pred_logit_list = []
    label_list = []
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(test_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1,))
            BATCH_SIZE = input.shape[0]
            test_num += BATCH_SIZE

            # get avh output
            # lacking of r_target
            feature_inter = model_alpha.model(input).detach().cpu().numpy()
            prediction_avh = avh_score(feature_inter, w)

            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(), r_target.cuda()).detach()
            prediction_t_eval = F.softmax(target_out, dim=1)

            # use avh prediction results as final one
            prediction_t = torch.tensor(prediction_avh)

            test_loss = float(ce_loss(target_out, label.long()))
            losses.update(test_loss, BATCH_SIZE)
            prec1, gt_num = accuracy_new(prediction_t_eval, label.long(), CONFIG["num_class"], topk=(1,))
            top1_acc.update(prec1[0], gt_num[0])
            mis_idx = (torch.argmax(prediction_t_eval, dim=1) != label.long()).nonzero().reshape(-1, )
            mis_pred = prediction_t_eval[mis_idx]
            mis_ent += entropy(mis_pred) / math.log(CONFIG["num_class"], 2)
            mis_num += mis_idx.shape[0]
            # one-hot encoding
            label_onehot = torch.zeros(prediction_t_eval.shape)
            label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
            for j in range(input.shape[0]):
                brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t_eval[j].cpu().numpy())
            if i % CONFIG["print_freq"] == 0:
                print('Test: [{0}/{1}]\n'
                      'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
                      'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
                    i, len(test_loader), loss=losses, top1=top1_acc))
                print(r_target.reshape((-1,)))
            for idx in range(input.shape[0]):
                image_name_list.append(input_name[idx])
                pred_logit_list.append(prediction_t[idx, :])
                # label_list.append(label.data[idx])

    misent = mis_ent / mis_num
    brier_score = brier_score / test_num
    print("Mis entropy: {}, Brier score: {}".format(misent, brier_score))
    return top1_acc.vec2sca_avg, top1_acc.avg, losses.avg, misent, brier_score, image_name_list, pred_logit_list, label_list

def validate_cbst(test_loader, model_alpha):
    # validate model and select samples for self-training
    model_alpha.eval()
    top1_acc = AverageMeter()
    losses = AverageMeter()
    ce_loss = nn.CrossEntropyLoss()
    mis_ent, mis_num, brier_score, test_num = 0, 0, 0, 0
    image_name_list = []
    pred_logit_list = []
    label_list = []
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(test_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1,))
            BATCH_SIZE = input.shape[0]
            test_num += BATCH_SIZE
            target_out = model_alpha(input).detach()
            prediction_t = F.softmax(target_out, dim=1)
            test_loss = float(ce_loss(target_out, label.long()))
            losses.update(test_loss, BATCH_SIZE)
            prec1, gt_num = accuracy_new(prediction_t, label.long(), CONFIG["num_class"], topk=(1,))
            top1_acc.update(prec1[0], gt_num[0])
            mis_idx = (torch.argmax(prediction_t, dim=1) != label.long()).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            mis_ent += entropy(mis_pred) / math.log(CONFIG["num_class"], 2)
            mis_num += mis_idx.shape[0]
            # one-hot encoding
            label_onehot = torch.zeros(prediction_t.shape)
            label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
            for j in range(input.shape[0]):
                brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j].cpu().numpy())
            if i % CONFIG["print_freq"] == 0:
                print('Test: [{0}/{1}]\n'
                      'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
                      'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
                    i, len(test_loader), loss=losses, top1=top1_acc))
            for idx in range(input.shape[0]):
                image_name_list.append(input_name[idx])
                pred_logit_list.append(prediction_t[idx, :])
    brier_score = brier_score / test_num
    misent = mis_ent / mis_num
    print("Mis entropy: {}, Brier score: {}".format(misent, brier_score))

    return top1_acc.vec2sca_avg, top1_acc.avg, losses.avg, misent, brier_score, image_name_list, pred_logit_list, label_list

def pseudo_labeling_cbst(pred_logit_list, image_name_list, epoch):
    # Select images which would be transferred
    pred_logit = torch.stack(pred_logit_list)
    pred_max = torch.max(pred_logit, dim=1)
    confidence = pred_max[0].cpu().numpy()
    conf_idx = pred_max[1].cpu().numpy()

    # CBST
    #p = 0.2  # The only parameter need to be tuned, the portion of data to be converted
    #p = min(0.2 + 0.05*epoch, 0.5)
    p = 0.5
    class_specific_num = np.zeros(CONFIG["n_classes"])
    lambda_k = np.zeros(CONFIG["n_classes"])
    for j in range(conf_idx.shape[0]):
        class_specific_num[conf_idx[j]] += 1
    class_specific_convert_num = p * class_specific_num
    ## Get lambda_k and convert sample index
    convert_all_idx = np.zeros(1)
    for j in range(CONFIG["n_classes"]):
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
    save_path = CONFIG["tgt_train_list"]
    fo = open(save_path, "w")
    for idx in convert_all_idx:
        idx = int(idx)
        fo.write(image_name_list[idx] + ' ' + str(new_conf_idx[idx]) + '\n')
    fo.close()

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__=="__main__":
    print("Using init seed: ", CONFIG["rand_seed"])
    print ('Using device:', DEVICE)
    seed_torch(CONFIG["rand_seed"])

    train_and_val_rescue_var3()
    #train_and_val_cbst()