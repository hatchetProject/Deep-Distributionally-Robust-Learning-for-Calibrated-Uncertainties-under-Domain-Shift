"""
Try to show that self-training helps to make the problem closer towards the covariate setting
We compare ASG+ARM versus ASG+ARMST
"""

import numpy as np
import torch
import math
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from torchvision import transforms, datasets
from visda_exp import ImageClassdata
import random
from torch.optim import lr_scheduler

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
    #"name": "res101_visda_iid_eval",
    "lr": 1e-5,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_class": 12,
    "print_freq": 100,
    "epochs": 10,
    "rand_seed": 0,
}

class LastLayerData(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx]
        label = self.label[idx]
        return input, label

class ratio_estimator(nn.Module):
    def __init__(self):
        super(ratio_estimator, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        p = self.model(x)
        return p

class source_vis_extract(nn.Module):
    def __init__(self):
        super(source_vis_extract, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.model.fc.in_features
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
        )
        self.model.fc = extractor

    def forward(self, x_s):
        x = self.model(x_s)
        return x

def get_feat_src_tgt():
    # Get ARM last layer feature
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
    visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                    transform=data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True,
                                             **kwargs)

    ## ARM model
    model1 = source_vis_extract()
    resume_path = "runs/ASG_no_st_tune/alpha_best.pth.tar"
    state_pre = torch.load(resume_path)
    state = model1.state_dict()
    for key in state.keys():
        if key in state_pre.keys():
            state[key] = state_pre[key]
    model1.load_state_dict(state, strict=True)
    print("=> loaded checkpoint from '{}'".format(resume_path))

    model1 = model1.to(DEVICE)
    model1.eval()
    feature_concat1 = torch.zeros([1, 512])
    true_label_concat = torch.zeros([1, 1])
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            true_label_concat = np.concatenate([true_label_concat, label.cpu().numpy()])

            feature_last1 = model1(input).detach().cpu().numpy()
            feature_concat1 = np.concatenate([feature_concat1, feature_last1])

            if i % 100 == 0:
                print("{} sample batches processed".format(i))
    feature_concat1 = feature_concat1[1:]
    true_label_concat = true_label_concat[1:]
    np.save("log/arm_tgt_softmax.npy", feature_concat1)
    np.save("log/arm_tgt_true_label.npy", true_label_concat)

    visDA17_trainset = ImageClassdata(txt_file=CONFIG["src_gt_list"], root_dir=CONFIG["src_root"],
                                      transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(visDA17_trainset, batch_size=CONFIG["batch_size"], shuffle=True,
                                               **kwargs)
    feature_concat2 = torch.zeros([1, 512])
    true_label_concat_src = torch.zeros([1, 1])
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(train_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            true_label_concat_src = np.concatenate([true_label_concat_src, label.cpu().numpy()])

            feature_last2 = model1(input).detach().cpu().numpy()
            feature_concat2 = np.concatenate([feature_concat2, feature_last2])

            if i % 100 == 0:
                print("{} sample batches processed".format(i))
    feature_concat2 = feature_concat2[1:]
    true_label_concat_src = true_label_concat_src[1:]
    np.save("log/arm_src_softmax.npy", feature_concat2)
    np.save("log/arm_src_true_label.npy", true_label_concat_src)

def discriminate_by_class(model, train_loader, val_loader, optimizer, save_path):
    model = model.train()
    model = model.to(DEVICE)
    acc, test_num = 0, 0
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(CONFIG["epochs"]):
        for i, (input, label) in enumerate(train_loader):
            input = input.to(DEVICE)
            label = label.to(DEVICE)

            output = model(input)
            loss = ce_loss(output, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for j, (input, label) in enumerate(val_loader):
                input = input.to(DEVICE)
                label = label.to(DEVICE)
                test_num += label.shape[0]
                output = model(input)
                pred = F.softmax(output, dim=1)
                acc += torch.sum(torch.argmax(pred, dim=1) == label.long())
        acc = float(acc) * 100.0 / test_num
        print("Epoch {}, test acc: {}".format(epoch, acc))
        acc, test_num = 0, 0
    torch.save(model.state_dict(), save_path)

def train_re_by_class(src_feature_path, src_label_path, tgt_feature_path, tgt_label_path, method):
    src_feat = np.load(src_feature_path)
    src_label = np.load(src_label_path) # 12 class
    tgt_feat = np.load(tgt_feature_path)
    tgt_label = np.load(tgt_label_path)
    for label in range(CONFIG["n_classes"]):
        print("Performing training on class: {}".format(label))
        src_idx = np.where(src_label == label)[0]
        tgt_idx = np.where(tgt_label == label)[0]
        new_feat_src = src_feat[src_idx]
        new_feat_tgt = tgt_feat[tgt_idx]
        print(new_feat_src.shape, new_feat_tgt.shape)
        new_feat = np.concatenate([new_feat_src, new_feat_tgt], axis=0)
        new_label = np.concatenate([np.zeros(new_feat_src.shape[0]), np.ones(new_feat_tgt.shape[0])])
        new_feat = torch.tensor(new_feat)
        new_label = torch.tensor(new_label)
        class_dataset = LastLayerData(new_feat, new_label)
        train_size = int(0.9 * new_feat.shape[0])
        test_size = new_feat.shape[0] - train_size
        trainset, testset = torch.utils.data.dataset.random_split(class_dataset, [train_size, test_size])
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=True)

        model = ratio_estimator()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)
        save_path = "runs/ratio_estimator/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        discriminate_by_class(model, train_dataloader, val_dataloader, optimizer, save_path+method+str(label)+".pth.tar")

def train_re_all(src_feature_path, tgt_feature_path, method):
    print("Training with all features")
    src_feat = np.load(src_feature_path)
    tgt_feat = np.load(tgt_feature_path)
    feat = np.concatenate([src_feat, tgt_feat])
    label = np.concatenate([np.zeros(src_feat.shape[0]), np.ones(tgt_feat.shape[0])])
    dataset = LastLayerData(feat, label)
    train_size = int(0.9 * feat.shape[0])
    test_size = feat.shape[0] - train_size
    trainset, testset = torch.utils.data.dataset.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=True)

    model = ratio_estimator()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)
    save_path = "runs/ratio_estimator/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    discriminate_by_class(model, train_dataloader, val_dataloader, optimizer, save_path + method + "_all.pth.tar")

def train_re_by_class_combined(src_feature_path, src_label_path, tgt_feature_path, tgt_label_path):
    src_feat0 = np.load(src_feature_path[0])
    src_label0 = np.load(src_label_path[0])
    tgt_feat0 = np.load(tgt_feature_path[0])
    tgt_label0 = np.load(tgt_label_path[0])
    src_feat1 = np.load(src_feature_path[1])
    src_label1 = np.load(src_label_path[1])
    tgt_feat1 = np.load(tgt_feature_path[1])
    tgt_label1 = np.load(tgt_label_path[1])
    src_feat = np.concatenate([src_feat0, src_feat1])
    src_label = np.concatenate([src_label0, src_label1])
    tgt_feat = np.concatenate([tgt_feat0, tgt_feat1])
    tgt_label = np.concatenate([tgt_label0, tgt_label1])
    for label in range(CONFIG["n_classes"]):
        print("Performing training on class: {}".format(label))
        src_idx = np.where(src_label == label)[0]
        tgt_idx = np.where(tgt_label == label)[0]
        new_feat_src = src_feat[src_idx]
        new_feat_tgt = tgt_feat[tgt_idx]
        print(new_feat_src.shape, new_feat_tgt.shape)
        new_feat = np.concatenate([new_feat_src, new_feat_tgt], axis=0)
        new_label = np.concatenate([np.zeros(new_feat_src.shape[0]), np.ones(new_feat_tgt.shape[0])])
        new_feat = torch.tensor(new_feat)
        new_label = torch.tensor(new_label)
        class_dataset = LastLayerData(new_feat, new_label)
        train_size = int(0.9 * new_feat.shape[0])
        test_size = new_feat.shape[0] - train_size
        trainset, testset = torch.utils.data.dataset.random_split(class_dataset, [train_size, test_size])
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=True)

        model = ratio_estimator()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)
        save_path = "runs/ratio_estimator/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        discriminate_by_class(model, train_dataloader, val_dataloader, optimizer, save_path+str(label)+".pth.tar")

def train_re_all_combined(src_feature_path, tgt_feature_path):
    print("Training with all features")
    src_feat0 = np.load(src_feature_path[0])
    tgt_feat0 = np.load(tgt_feature_path[0])
    src_feat1 = np.load(src_feature_path[1])
    tgt_feat1 = np.load(tgt_feature_path[1])
    src_feat = np.concatenate([src_feat0, src_feat1])
    tgt_feat = np.concatenate([tgt_feat0, tgt_feat1])
    feat = np.concatenate([src_feat, tgt_feat])
    label = np.concatenate([np.zeros(src_feat.shape[0]), np.ones(tgt_feat.shape[0])])
    dataset = LastLayerData(feat, label)
    train_size = int(0.9 * feat.shape[0])
    test_size = feat.shape[0] - train_size
    trainset, testset = torch.utils.data.dataset.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=True)

    model = ratio_estimator()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)
    save_path = "runs/ratio_estimator/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    discriminate_by_class(model, train_dataloader, val_dataloader, optimizer, save_path + "_all.pth.tar")

def calc_shift_loss(sample_batch, model_all, model_class, r_y):
    # shift loss is defined by p_s(x)/p_t(x) - p_s(y)/p_t(y)
    pred_all = F.softmax(model_all(sample_batch), dim=1)
    pred_class = F.softmax(model_class(sample_batch), dim=1)
    r_all = pred_all[:, 0] / pred_all[:, 1]
    r_class = pred_class[:, 0] / pred_class[:, 1]
    score = r_all - r_y * r_class
    return torch.abs(score)

def eval_shift(src_feat_path, src_label_path, tgt_feat_path, tgt_label_path, method):
    print("Calculating method {}".format(method))
    src_feat = np.load(src_feat_path)
    src_label = np.load(src_label_path)  # 12 class
    tgt_feat = np.load(tgt_feat_path)
    tgt_label = np.load(tgt_label_path)
    save_path = "runs/ratio_estimator/"
    model_all = ratio_estimator()
    #model_all.load_state_dict(torch.load(save_path + method + "_all.pth.tar"))
    model_all.load_state_dict(torch.load(save_path + "_all.pth.tar"))
    total_score = np.zeros(CONFIG["n_classes"])
    for l in range(CONFIG["n_classes"]):
        src_idx = np.where(src_label == l)[0]
        tgt_idx = np.where(tgt_label == l)[0]
        new_feat_src = src_feat[src_idx]
        new_feat_tgt = tgt_feat[tgt_idx]
        new_feat = np.concatenate([new_feat_src, new_feat_tgt], axis=0)
        new_label = np.concatenate([np.zeros(new_feat_src.shape[0]), np.ones(new_feat_tgt.shape[0])])
        new_feat = torch.tensor(new_feat)
        new_label = torch.tensor(new_label)
        class_dataset = LastLayerData(new_feat, new_label)
        class_dataloader = torch.utils.data.DataLoader(class_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        r_class = float(src_idx.shape[0])/tgt_idx.shape[0]
        model_class = ratio_estimator()
        #model_class.load_state_dict(torch.load(save_path + method + str(l) + ".pth.tar"))
        model_class.load_state_dict(torch.load(save_path + str(l) + ".pth.tar"))
        test_num = 0
        for i, (input, _) in enumerate(class_dataloader):
            test_num += input.shape[0]
            batch_score = calc_shift_loss(input, model_all, model_class, r_class)
            batch_score = torch.sum(batch_score).detach().cpu().numpy()
            total_score[l] += batch_score
        total_score[l] /= test_num
    print("Class specific loss for method {} is {}".format(method, total_score))
    print("Total loss is {}".format(np.sum(np.abs(total_score))))
    return total_score, np.sum(np.abs(total_score))

def eval_epoch_by_epoch():
    directory = "log/cov_shift/"
    arm_class = []
    arm_total = []
    armst_class = []
    armst_total = []
    for i in range(0, 20, 2):
        print("Doing epoch {}".format(i))
        armst_src_feat = directory + "drst_src_feat_" + str(i) + ".npy"
        armst_src_label = directory + "drst_src_label_" + str(i) + ".npy"
        armst_tgt_feat = directory + "drst_tgt_feat_" +str(i) + ".npy"
        armst_tgt_label = directory + "drst_tgt_label_" + str(i) + ".npy"

        arm_src_feat = directory + "drl_src_feat_" + str(i) + ".npy"
        arm_src_label = directory + "drl_src_label_" + str(i) + ".npy"
        arm_tgt_feat = directory + "drl_tgt_feat_" + str(i) + ".npy"
        arm_tgt_label = directory + "drl_tgt_label_" + str(i) + ".npy"

        src_feat_path = [armst_src_feat, arm_src_feat]
        src_label_path = [armst_src_label, arm_src_label]
        tgt_feat_path = [armst_tgt_feat, arm_tgt_feat]
        tgt_label_path = [armst_tgt_label, armst_tgt_label]
        train_re_all_combined(src_feat_path, tgt_feat_path)
        train_re_by_class_combined(src_feat_path, src_label_path, tgt_feat_path, tgt_label_path)
        class_arm, total_arm = eval_shift(arm_src_feat, arm_src_label, arm_tgt_feat, arm_tgt_label, "ARM")
        class_armst, total_armst = eval_shift(armst_src_feat, armst_src_label, armst_tgt_feat, armst_tgt_label, "ARMST")
        arm_class.append(class_arm)
        arm_total.append(total_arm)
        armst_class.append(class_armst)
        armst_total.append(total_armst)
        np.save(directory + "arm_class_odd.npy", arm_class)
        np.save(directory + "arm_total_odd.npy", arm_total)
        np.save(directory + "armst_class_odd.npy", armst_class)
        np.save(directory + "armst_total_odd.npy", armst_total)

if __name__=="__main__":
    """
    #get_feat_src_tgt()
    method = "armst"
    armst_src_feat = "log/best_softmax_src.npy"
    armst_src_label = "log/tsne_true_label_src.npy"
    armst_tgt_feat = "log/best_softmax.npy"
    armst_tgt_label = "log/tsne_true_label.npy"
    #train_re_all(armst_src_feat, armst_tgt_feat, method)
    #train_re_by_class(armst_src_feat, armst_src_label, armst_tgt_feat, armst_tgt_label, method)
    #eval_shift(armst_src_feat, armst_src_label, armst_tgt_feat, armst_tgt_label, method)

    method = "arm"
    arm_src_feat = "log/arm_src_softmax.npy"
    arm_src_label = "log/arm_src_true_label.npy"
    arm_tgt_feat = "log/arm_tgt_softmax.npy"
    arm_tgt_label = "log/arm_tgt_true_label.npy"
    #train_re_all(arm_src_feat, arm_tgt_feat, method)
    #train_re_by_class(arm_src_feat, arm_src_label, arm_tgt_feat, arm_tgt_label, method)
    #eval_shift(arm_src_feat, arm_src_label, arm_tgt_feat, arm_tgt_label, method)
    src_feat_path = [armst_src_feat, arm_src_feat]
    src_label_path = [armst_src_label, arm_src_label]
    tgt_feat_path = [armst_tgt_feat, arm_tgt_feat]
    tgt_label_path = [armst_tgt_label, armst_tgt_label]
    #train_re_all_combined(src_feat_path, tgt_feat_path)
    #train_re_by_class_combined(src_feat_path, src_label_path, tgt_feat_path, tgt_label_path)
    eval_shift(arm_src_feat, arm_src_label, arm_tgt_feat, arm_tgt_label, "ARM")
    eval_shift(armst_src_feat, armst_src_label, armst_tgt_feat, armst_tgt_label, "ARMST")
    """
    eval_epoch_by_epoch()