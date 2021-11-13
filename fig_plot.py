"""
Plot figures for the experiment logs
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG_PATH = {
    "best_dir1": "runs/best_model_v1/",
    "best_dir2": "runs/best_model_v2/",
    "fixR_dir": "runs/fixR_model/",
    "cbst_dir": "runs/cbst_model/",
    "backbone_dir": "runs/res101_visda/",
}

import torchvision
import torch.nn as nn
from model_layers import ClassifierLayerAVH, GradLayer
class alpha_vis_old(nn.Module):
    def __init__(self, n_output):
        super(alpha_vis_old, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
        )
        self.model.fc = self.extractor
        self.final_layer = ClassifierLayerAVH(512, n_output, bias=True)

    def forward(self, x_s, y_s, r):
        x1 = self.model(x_s)
        x = self.final_layer(x1, y_s, r)
        return x

class beta_vis_old(nn.Module):
    def __init__(self):
        super(beta_vis_old, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 2)
        )
        self.model.fc = self.extractor
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p

def get_softmax():
    from visda_exp import CONFIG, ImageClassdata, avh_score
    from visda_exp import source_vis, beta_vis, alpha_vis
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

    resume_path = CONFIG_PATH["backbone_dir"]
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path + "epoch_19_checkpoint.pth.tar")
    iid_model = source_vis(12)
    iid_model.load_state_dict(checkpoint['state_dict'], strict=True)

    resume_path = CONFIG_PATH["fixR_dir"]
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint_alpha = torch.load(resume_path + "epoch_19_alpha_checkpoint.pth.tar")
    checkpoint_beta = torch.load(resume_path + "epoch_19_beta_checkpoint.pth.tar")
    alpha_fixR = alpha_vis(12)
    beta_fixR = beta_vis()
    alpha_fixR.load_state_dict(checkpoint_alpha)
    beta_fixR.load_state_dict(checkpoint_beta)

    resume_path = CONFIG_PATH["best_dir1"]
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint_alpha = torch.load(resume_path + "epoch_19_alpha_checkpoint.pth.tar")
    checkpoint_beta = torch.load(resume_path + "epoch_19_beta_checkpoint.pth.tar")
    alpha_best = alpha_vis(12)
    beta_best = beta_vis()
    alpha_best.load_state_dict(checkpoint_alpha)
    beta_best.load_state_dict(checkpoint_beta)

    resume_path = CONFIG_PATH["best_dir2"]
    print("=> Loading checkpoint '{}'".format(resume_path))
    #checkpoint_alpha = torch.load(resume_path + "alpha_best.pth.tar")
    #checkpoint_beta = torch.load(resume_path + "beta_best.pth.tar")
    checkpoint_alpha = torch.load(resume_path + "epoch_19_alpha_checkpoint.pth.tar")
    checkpoint_beta = torch.load(resume_path + "epoch_19_beta_checkpoint.pth.tar")
    alpha_best2 = alpha_vis(12)
    beta_best2 = beta_vis()
    alpha_best2.load_state_dict(checkpoint_alpha)
    beta_best2.load_state_dict(checkpoint_beta)

    resume_path = CONFIG_PATH["cbst_dir"]
    print("=> Loading checkpoint '{}'".format(resume_path))
    #checkpoint = torch.load(resume_path+"best.pth.tar")
    checkpoint = torch.load(resume_path + "epoch_19checkpoint.pth.tar")
    cbst_model = source_vis(12)
    cbst_model.load_state_dict(checkpoint, strict=True)

    iid_pred = np.zeros([1, 12])
    fixR_pred = np.zeros([1, 12])
    best_pred = np.zeros([1, 12])
    best_pred2 = np.zeros([1, 12])
    cbst_pred = np.zeros([1, 12])
    label_rec = np.zeros([1])
    for i, (input, label, _) in enumerate(val_loader):
        input = input.to(DEVICE)
        label = label.to(DEVICE)
        label = label.reshape((-1,))
        BATCH_SIZE = input.shape[0]
        label_rec = np.concatenate([label_rec, label.cpu().numpy()], axis=0)

        # IID
        iid_out = iid_model(input).detach()
        iid_softmax = F.softmax(iid_out, dim=1)
        iid_softmax = iid_softmax.cpu().numpy()
        iid_pred = np.concatenate([iid_pred, iid_softmax], axis=0)

        # fixed R
        pred = F.softmax(beta_fixR(input, None, None, None, None).detach(), dim=1)
        r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
        target_out = alpha_fixR(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])), r_target).detach()
        fixR_softmax = F.softmax(target_out, dim=1)
        fixR_softmax = fixR_softmax.cpu().numpy()
        fixR_pred = np.concatenate([fixR_pred, fixR_softmax], axis=0)

        # Best model v1
        pred = F.softmax(beta_best(input, None, None, None, None).detach(), dim=1)
        r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
        target_out = alpha_best(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])), r_target).detach()
        best_softmax = F.softmax(target_out, dim=1)
        best_softmax = best_softmax.cpu().numpy()
        best_pred = np.concatenate([best_pred, best_softmax], axis=0)

        # Best model v2
        pred = F.softmax(beta_best2(input, None, None, None, None).detach(), dim=1)
        r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
        target_out = alpha_best2(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])), r_target).detach()
        best_softmax2 = F.softmax(target_out, dim=1)
        best_softmax2 = best_softmax2.cpu().numpy()
        best_pred2 = np.concatenate([best_pred2, best_softmax2], axis=0)

        # CBST
        cbst_out = cbst_model(input).detach()
        cbst_softmax = F.softmax(cbst_out, dim=1)
        cbst_softmax = cbst_softmax.cpu().numpy()
        cbst_pred = np.concatenate([cbst_pred, cbst_softmax], axis=0)

        if (i+1) % 100 == 0:
            print ("100 batches processed")

    np.save("log/asg_softmax.npy", iid_pred[1:])
    np.save("log/asg_fixR_softmax.npy", fixR_pred[1:])
    np.save("log/asg_cbst_softmax.npy", cbst_pred[1:])
    np.save("log/best_model_softmax.npy", best_pred[1:])
    np.save("log/labels.npy", label_rec[1:])

def get_softmax_new():
    from visda_exp import CONFIG, ImageClassdata, avh_score
    from visda_exp import source_vis, beta_vis, alpha_vis
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

    # ASG
    model_asg = source_vis(12)
    resume_path = "runs/res101_asg/res101_vista17_best.pth.tar"
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    state_pre = checkpoint["state_dict"]
    for key in list(state_pre.keys()):
        state_pre["model." + key] = state_pre.pop(key)
    state = model_asg.state_dict()
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
    model_asg.load_state_dict(state, strict=True)
    print("=> loaded checkpoint '{}' (epoch {}), with starting precision {:.3f}"
          .format(resume_path, checkpoint['epoch'], best_prec1))

    """
    # CBST
    model_cbst = source_vis(12)
    resume_path = "runs/cbst_asg_model/"
    checkpoint = torch.load(resume_path + "epoch_11checkpoint.pth.tar")
    state = model_cbst.state_dict()
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model_cbst.load_state_dict(state, strict=True)
    print("=> Loaded checkpoint '{}'".format(resume_path))

    # Fix R
    alpha_fixR = alpha_vis(12)
    beta_fixR = beta_vis()
    checkpoint_alpha = torch.load("runs/asg_fixR/alpha_best.pth.tar")
    checkpoint_beta = torch.load("runs/asg_fixR/beta_best.pth.tar")
    alpha_fixR.load_state_dict(checkpoint_alpha)
    beta_fixR.load_state_dict(checkpoint_beta)
    print("Loaded checkpoint from {}".format("runs/asg_fixR/alpha_best.pth.tar"))
    """
    # TS
    ts_model = source_vis(12)
    """
    resume_path = "runs/cbst_asg_model/"
    checkpoint = torch.load(resume_path + "epoch_11checkpoint.pth.tar")
    state = ts_model.state_dict()
    for key in state.keys():
        if key in checkpoint.keys():
            state[key] = checkpoint[key]
        else:
            print("Param {} not found".format(key))
            raise ValueError("Model not loaded completely")
    ts_model.load_state_dict(state, strict=True)"""
    from temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(ts_model)
    scaled_model.load_state_dict(torch.load("runs/ts_model/ts.pth.tar"))
    #scaled_model.set_temperature(val_loader)
    print("Loaded checkpoint from {} for temperature scaling".format(resume_path))

    # Best model
    alpha_best = alpha_vis(12)
    beta_best = beta_vis()
    #checkpoint_alpha = torch.load("runs/best_model_comp13/alpha_best.pth.tar")
    #checkpoint_beta = torch.load("runs/best_model_comp13/beta_best.pth.tar")
    checkpoint_alpha = torch.load("runs/ASG_no_st/alpha_best.pth.tar")
    checkpoint_beta = torch.load("runs/ASG_no_st/beta_best.pth.tar")
    alpha_best.load_state_dict(checkpoint_alpha)
    beta_best.load_state_dict(checkpoint_beta)
    print("Loaded checkpoint from {}".format("ASG_no_st files"))

    model_asg = model_asg.to(DEVICE)
    #model_cbst = model_cbst.to(DEVICE)
    #alpha_fixR, beta_fixR = alpha_fixR.to(DEVICE), beta_fixR.to(DEVICE)
    scaled_model = scaled_model.to(DEVICE)
    alpha_best, beta_best = alpha_best.to(DEVICE), beta_best.to(DEVICE)

    asg_pred = np.zeros([1, 12])
    #cbst_pred = np.zeros([1, 12])
    #fixR_pred = np.zeros([1, 12])
    ts_pred = np.zeros([1, 12])
    best_pred = np.zeros([1, 12])
    label_rec = np.zeros([1])
    with torch.no_grad():
        for i, (input, label, _) in enumerate(val_loader):
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            label = label.reshape((-1,))
            BATCH_SIZE = input.shape[0]
            label_rec = np.concatenate([label_rec, label.cpu().numpy()], axis=0)

            # ASG
            asg_out = model_asg(input).detach()
            asg_softmax = F.softmax(asg_out, dim=1)
            asg_softmax = asg_softmax.cpu().numpy()
            asg_pred = np.concatenate([asg_pred, asg_softmax], axis=0)

            # CBST
            #cbst_out = model_cbst(input).detach()
            #cbst_softmax = F.softmax(cbst_out, dim=1)
            #cbst_softmax = cbst_softmax.cpu().numpy()
            #cbst_pred = np.concatenate([cbst_pred, cbst_softmax], axis=0)

            # TS
            ts_out = scaled_model(input).detach()
            ts_softmax = F.softmax(ts_out, dim=1)
            ts_softmax = ts_softmax.cpu().numpy()
            ts_pred = np.concatenate([ts_pred, ts_softmax], axis=0)

            # fixed R
            #pred = F.softmax(beta_fixR(input, None, None, None, None).detach(), dim=1)
            #r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            #target_out = alpha_fixR(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(), r_target.cuda()).detach()
            #fixR_softmax = F.softmax(target_out, dim=1)
            #fixR_softmax = fixR_softmax.cpu().numpy()
            #fixR_pred = np.concatenate([fixR_pred, fixR_softmax], axis=0)

            # Best model
            pred = F.softmax(beta_best(input, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = alpha_best(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(), r_target.cuda()).detach()
            best_softmax = F.softmax(target_out, dim=1)
            best_softmax = best_softmax.cpu().numpy()
            best_pred = np.concatenate([best_pred, best_softmax], axis=0)

            if (i+1) % 100 == 0:
                print ("{} batches processed".format((i+1)))

    np.save("log/asg_softmax.npy", asg_pred[1:])
    #np.save("log/asg_fixR_softmax.npy", fixR_pred[1:])
    #np.save("log/asg_cbst_softmax.npy", cbst_pred[1:])
    np.save("log/ts_softmax.npy", ts_pred[1:])
    np.save("log/best_model_softmax.npy", best_pred[1:])
    np.save("log/labels.npy", label_rec[1:])

def reliability_plot():
    label = np.load("log/labels.npy", allow_pickle=True)
    intervals = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 1.0]
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=2)
    # IID
    iid_softmax = np.load("log/iid_softmax.npy", allow_pickle=True)
    iid_conf = np.max(iid_softmax, axis=1)
    iid_acc = (np.argmax(iid_softmax, axis=1) == label)
    """
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((iid_conf >= intervals[i]) == (iid_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(iid_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(iid_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="IID", marker="^", color="#e50000", linewidth=2.5, markersize=12)
    """
    # IID + CBST
    cbst_softmax = np.load("log/cbst_softmax.npy", allow_pickle=True)
    cbst_conf = np.max(cbst_softmax, axis=1)
    cbst_acc = (np.argmax(cbst_softmax, axis=1) == label)

    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((cbst_conf >= intervals[i]) == (cbst_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(cbst_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(cbst_conf[int_idx]) - np.sum(cbst_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="CBST", marker="v", color="#96f97b", linewidth=2.5, markersize=12)

    # Fixed R
    fixr_softmax = np.load("log/fixR_softmax.npy", allow_pickle=True)
    fixr_conf = np.max(fixr_softmax, axis=1)
    fixr_acc = (np.argmax(fixr_softmax, axis=1) == label)

    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((fixr_conf >= intervals[i]) == (fixr_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(fixr_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(fixr_conf[int_idx]) - np.sum(fixr_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Fixed ratio", marker="o", color="#f97306", linewidth=2.5,
             markersize=12)

    # Ours, self-training with softmax score
    best_softmax = np.load("log/best_softmax.npy", allow_pickle=True)
    best_conf = np.max(best_softmax, axis=1)
    best_acc = (np.argmax(best_softmax, axis=1) == label)

    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((best_conf >= intervals[i]) == (best_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(best_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(best_conf[int_idx]) - np.sum(best_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Ours (softmax score ST)", marker="s", color="#c20078", linewidth=2.5,
             markersize=12)

    # Ours, self-training with AVH score
    best_softmax2 = np.load("log/best_softmax2.npy", allow_pickle=True)
    best_conf2 = np.max(best_softmax2, axis=1)
    best_acc2 = (np.argmax(best_softmax2, axis=1) == label)

    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((best_conf2 >= intervals[i]) == (best_conf2 < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(best_conf2[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(best_conf2[int_idx]) - np.sum(best_acc2[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Ours (AVH score ST)", marker="d", color="#15b01a", linewidth=2.5, markersize=12)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=24)
    plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.savefig("log/reliability_plot.jpg")

def reliability_plot_new():
    label = np.load("log/labels2.npy", allow_pickle=True)
    intervals = [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.95, 1.0]
    #intervals = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    plt.figure(figsize=(30, 15))
    plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=5)

    # ASG
    asg_softmax = np.load("log/asg_softmax2.npy", allow_pickle=True)
    asg_conf = np.max(asg_softmax, axis=1)
    asg_acc = (np.argmax(asg_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((asg_conf >= intervals[i]) == (asg_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(asg_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(asg_conf[int_idx]) - np.sum(asg_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Source", marker="^", color="skyblue", linewidth=9, markersize=36)
    """
    # ASG + CBST
    cbst_softmax = np.load("log/asg_cbst_softmax.npy", allow_pickle=True)
    cbst_conf = np.max(cbst_softmax, axis=1)
    cbst_acc = (np.argmax(cbst_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((cbst_conf >= intervals[i]) == (cbst_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(cbst_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(cbst_conf[int_idx]) - np.sum(cbst_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="ASG+CBST", marker="v", color="#96f97b", linewidth=2.5, markersize=12)

    # Fixed R
    fixr_softmax = np.load("log/asg_fixR_softmax.npy", allow_pickle=True)
    fixr_conf = np.max(fixr_softmax, axis=1)
    fixr_acc = (np.argmax(fixr_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((fixr_conf >= intervals[i]) == (fixr_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(fixr_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(fixr_conf[int_idx]) - np.sum(fixr_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="ASG+Fixed ratio", marker="o", color="#f97306", linewidth=2.5,
             markersize=12)
    """
    # Temperature scaling
    ts_softmax = np.load("log/ts_softmax2.npy", allow_pickle=True)
    ts_conf = np.max(ts_softmax, axis=1)
    ts_acc = (np.argmax(ts_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((ts_conf >= intervals[i]) == (ts_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(ts_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(ts_conf[int_idx]) - np.sum(ts_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="TS", marker="v", color="limegreen", linewidth=9,
             markersize=36)

    # Ours
    best_softmax = np.load("log/best_model_softmax2.npy", allow_pickle=True)
    best_conf = np.max(best_softmax, axis=1)
    best_acc = (np.argmax(best_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((best_conf >= intervals[i]) == (best_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(best_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(best_conf[int_idx]) - np.sum(best_acc[int_idx])) / np.sum(int_idx))
    print(x_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="DRL", marker="o", color="coral", linewidth=9,
             markersize=36)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=42)
    plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 72})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 72})
    #plt.legend(shadow=True, fontsize='x-large')
    plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.30), ncol=1, fontsize=42)
    plt.savefig("log/finals/fig7_e.jpg")

def reliability_plot_office():
    """
    from office_exp import source_office, alpha_office, beta_office, dataloader_office
    _, val_loader = dataloader_office("office/amazon", "office/webcam")
    # get the softmax`
    ## load model
    model_source = source_office(31)
    # this might be broken... office_iid, check the acc and select the epoch
    resume_path = "runs/office_iid/epoch_19.pth.tar"
    model_source.load_state_dict(torch.load(resume_path))
    model_source = model_source.to(DEVICE)
    model_source.eval()

    drst_alpha = alpha_office(31)
    drst_beta = beta_office()
    resume_path_alpha = "runs/office_best/aw_alpha.pth.tar"
    resume_path_beta = "runs/office_best/aw_beta.pth.tar"
    drst_alpha.load_state_dict(torch.load(resume_path_alpha))
    drst_beta.load_state_dict(torch.load(resume_path_beta))
    drst_alpha = drst_alpha.to(DEVICE)
    drst_beta = drst_beta.to(DEVICE)
    drst_alpha.eval()
    drst_beta.eval()

    ts_model = source_office(31)
    resume_path = "runs/office_iid/epoch_19.pth.tar"
    ts_model.load_state_dict(torch.load(resume_path))
    from temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(ts_model)
    scaled_model.set_temperature(val_loader)
    torch.save(scaled_model.state_dict(), "runs/office_iid/ts_model.pth.tar")
    scaled_model = scaled_model.to(DEVICE)
    scaled_model.eval()

    source_pred = np.zeros([1, 31])
    ts_pred = np.zeros([1, 31])
    drst_pred = np.zeros([1, 31])
    label_rec = np.zeros([1])
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1, ))
            BATCH_SIZE = input.shape[0]
            label_rec = np.concatenate([label_rec, label.cpu().numpy()], axis=0)

            output = model_source(input).detach()
            source_softmax = F.softmax(output, dim=1)
            source_softmax = source_softmax.cpu().numpy()
            source_pred = np.concatenate([source_pred, source_softmax], axis=0)

            ts_out = scaled_model(input).detach()
            ts_softmax = F.softmax(ts_out, dim=1)
            ts_softmax = ts_softmax.cpu().numpy()
            ts_pred = np.concatenate([ts_pred, ts_softmax], axis=0)

            pred = F.softmax(drst_beta(input, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = drst_alpha(input, torch.ones((BATCH_SIZE, 31)).cuda(),
                                    r_target.cuda()).detach()
            drst_softmax = F.softmax(target_out, dim=1)
            drst_softmax = drst_softmax.cpu().numpy()
            drst_pred = np.concatenate([drst_pred, drst_softmax], axis=0)
    source_pred = source_pred[1:]
    ts_pred = ts_pred[1:]
    drst_pred = drst_pred[1:]
    label_rec = label_rec[1:]
    np.save("log/office_src_softmax.npy", source_pred)
    np.save("log/office_ts_softmax.npy", ts_pred)
    np.save("log/office_drst_softmax.npy", drst_pred)
    np.save("log/office_label.npy", label_rec)
    """
    intervals = [0, 0.85, 0.87, 0.9, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
    plt.figure(figsize=(21, 15))
    plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=5)
    label = np.load("log/office_label.npy")
    # IID
    iid_softmax = np.load("log/office_src_softmax.npy", allow_pickle=True)
    iid_conf = np.max(iid_softmax, axis=1)
    iid_acc = (np.argmax(iid_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((iid_conf >= intervals[i]) == (iid_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(iid_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(iid_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Source", marker="^", color="skyblue", linewidth=18, markersize=36)

    # TS
    ts_softmax = np.load("log/office_ts_softmax.npy", allow_pickle=True)
    ts_conf = np.max(ts_softmax, axis=1)
    ts_acc = (np.argmax(ts_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((ts_conf >= intervals[i]) == (ts_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(ts_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(ts_conf[int_idx]) - np.sum(ts_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="TS", marker="v", color="limegreen", linewidth=18, markersize=36)

    # DRST
    drst_softmax = np.load("log/office_drst_softmax.npy", allow_pickle=True)
    drst_conf = np.max(drst_softmax, axis=1)
    drst_acc = (np.argmax(drst_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((drst_conf >= intervals[i]) == (drst_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(drst_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(drst_conf[int_idx]) - np.sum(drst_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="DRL", marker="o", color="coral", linewidth=18, markersize=36)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=42)
    plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 72})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 72})
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=42)
    #plt.legend(fontsize=36)
    #plt.title("Reliability plot on Office31 (A → W)", fontdict={"weight": "normal", "size": 42})
    plt.savefig("log/finals/fig7_c.jpg")

def reliability_plot_home():
    """
    from office_exp import source_office, alpha_office, beta_office, dataloader_office
    _, val_loader = dataloader_office("OfficeHome/Product", "OfficeHome/Art")
    # get the softmax
    ## load model
    model_source = source_office(65)
    # this might be broken... office_iid, check the acc and select the epoch
    resume_path = "runs/OfficeHome_iid/epoch_19.pth.tar"
    model_source.load_state_dict(torch.load(resume_path))
    model_source = model_source.to(DEVICE)
    model_source.eval()

    drst_alpha = alpha_office(65)
    drst_beta = beta_office()
    resume_path_alpha = "runs/office_best/pa_alpha.pth.tar"
    resume_path_beta = "runs/office_best/pa_beta.pth.tar"
    drst_alpha.load_state_dict(torch.load(resume_path_alpha))
    drst_beta.load_state_dict(torch.load(resume_path_beta))
    drst_alpha = drst_alpha.to(DEVICE)
    drst_beta = drst_beta.to(DEVICE)
    drst_alpha.eval()
    drst_beta.eval()

    ts_model = source_office(65)
    resume_path = "runs/OfficeHome_iid/epoch_19.pth.tar"
    ts_model.load_state_dict(torch.load(resume_path))
    from temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(ts_model)
    scaled_model.set_temperature(val_loader)
    torch.save(scaled_model.state_dict(), "runs/OfficeHome_iid/ts_model.pth.tar")
    scaled_model = scaled_model.to(DEVICE)
    scaled_model.eval()

    source_pred = np.zeros([1, 65])
    ts_pred = np.zeros([1, 65])
    drst_pred = np.zeros([1, 65])
    label_rec = np.zeros([1])
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1, ))
            BATCH_SIZE = input.shape[0]
            label_rec = np.concatenate([label_rec, label.cpu().numpy()], axis=0)

            output = model_source(input).detach()
            source_softmax = F.softmax(output, dim=1)
            source_softmax = source_softmax.cpu().numpy()
            source_pred = np.concatenate([source_pred, source_softmax], axis=0)

            ts_out = scaled_model(input).detach()
            ts_softmax = F.softmax(ts_out, dim=1)
            ts_softmax = ts_softmax.cpu().numpy()
            ts_pred = np.concatenate([ts_pred, ts_softmax], axis=0)

            pred = F.softmax(drst_beta(input, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = drst_alpha(input, torch.ones((BATCH_SIZE, 65)).cuda(),
                                    r_target.cuda()).detach()
            drst_softmax = F.softmax(target_out, dim=1)
            drst_softmax = drst_softmax.cpu().numpy()
            drst_pred = np.concatenate([drst_pred, drst_softmax], axis=0)
    source_pred = source_pred[1:]
    ts_pred = ts_pred[1:]
    drst_pred = drst_pred[1:]
    label_rec = label_rec[1:]
    np.save("log/home_src_softmax.npy", source_pred)
    np.save("log/home_ts_softmax.npy", ts_pred)
    np.save("log/home_drst_softmax.npy", drst_pred)
    np.save("log/home_label.npy", label_rec)
    """
    intervals = [0, 0.85, 0.87, 0.9, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
    plt.figure(figsize=(21, 15))
    plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=5)
    label = np.load("log/home_label.npy")
    # IID
    iid_softmax = np.load("log/home_src_softmax.npy", allow_pickle=True)
    iid_conf = np.max(iid_softmax, axis=1)
    iid_acc = (np.argmax(iid_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((iid_conf >= intervals[i]) == (iid_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(iid_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(iid_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Source", marker="^", color="skyblue", linewidth=18, markersize=36)

    # TS
    ts_softmax = np.load("log/home_ts_softmax.npy", allow_pickle=True)
    ts_conf = np.max(ts_softmax, axis=1)
    ts_acc = (np.argmax(ts_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((ts_conf >= intervals[i]) == (ts_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(ts_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(ts_conf[int_idx]) - np.sum(ts_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="TS", marker="v", color="limegreen", linewidth=18, markersize=36)

    # DRST
    drst_softmax = np.load("log/home_drst_softmax.npy", allow_pickle=True)
    drst_conf = np.max(drst_softmax, axis=1)
    drst_acc = (np.argmax(drst_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((drst_conf >= intervals[i]) == (drst_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(drst_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(drst_conf[int_idx]) - np.sum(drst_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="DRL", marker="o", color="coral", linewidth=18, markersize=36)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=42)
    plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 72})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 72})
    plt.ylim((-0.4, 0.6))
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, fontsize=42)
    #plt.legend(loc="lower right", fontsize=36)
    #plt.title("Reliability plot on OfficeHome (P → A)", fontdict={"weight": "normal", "size": 42})
    plt.savefig("log/finals/fig7_d.jpg")

def plot1():
    dic_best = np.load("log/visda_st.npy", allow_pickle=True).item()
    dic_best2 = np.load("log/visda_st_v2.npy", allow_pickle=True).item()
    dic_fixR = np.load("log/visda_fixR.npy", allow_pickle=True).item()
    dic_iid = np.load("log/visda_iid.npy", allow_pickle=True).item()
    dic_cbst = np.load("log/visda_cbst.npy", allow_pickle=True).item()

    acc_best = [x.cpu().numpy() for x in dic_best["acc"]]
    acc_best2 = [x.cpu().numpy() for x in dic_best2["acc"]]
    acc_fixR = [x.cpu().numpy() for x in dic_fixR["acc"]]
    acc_iid = [x.cpu().numpy() for x in dic_iid["acc"]]
    acc_cbst = [x.cpu().numpy() for x in dic_cbst["acc"]]

    brier_best = dic_best["brier"]
    brier_best2 = dic_best2["brier"]
    brier_fixR = dic_fixR["brier"]
    brier_iid = dic_iid["brier"]
    brier_cbst = dic_cbst["brier"]

    num_epoch = len(acc_best)
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), acc_best, label="Ours (softmax score ST)", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_best2, label="Ours (AVH score ST)", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_fixR, label="Fixed ratio", linewidth=3)
    #plt.plot(np.arange(num_epoch), acc_iid, label="IID", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_cbst, label="CBST", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Accuracy", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig("log/acc_compare.jpg")

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), brier_best, label="Ours (softmax score ST)", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_best2, label="Ours (AVH score ST)", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_fixR, label="Fixed ratio", linewidth=3)
    #plt.plot(np.arange(num_epoch), brier_iid, label="IID", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_cbst, label="CBST", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Brier score", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig("log/brier_compare.jpg")

def plot2():
    dic_best = np.load("log/ASG_no_st.npy", allow_pickle=True).item()
    acc_best = [x.cpu().numpy() for x in dic_best["acc"]]
    brier_best = dic_best["brier"]

    num_epoch = len(acc_best)
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), acc_best, label="Ours (softmax score ST)", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Accuracy", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig("log/asg_acc_compare.jpg")

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), brier_best, label="Ours (softmax score ST)", linewidth=3)

    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Brier score", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig("log/asg_brier_compare.jpg")

def plot3():
    dic_cbst = np.load("log/visda_cbst_asg.npy", allow_pickle=True).item()
    dic_fixR = np.load("log/asg_fixR.npy", allow_pickle=True).item()
    dic_best = np.load("log/var3_comp13.npy", allow_pickle=True).item()

    acc_best = [x.cpu().numpy() for x in dic_best["acc"]]
    acc_fixR = [x.cpu().numpy() for x in dic_fixR["acc"]]
    acc_cbst = [x.cpu().numpy() for x in dic_cbst["acc"]]

    brier_best = dic_best["brier"]
    brier_fixR = dic_fixR["brier"]
    brier_cbst = dic_cbst["brier"]

    num_epoch = len(acc_best)
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), acc_best, label="DRST", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_fixR, label="Fixed ratio", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_cbst, label="CBST", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Accuracy", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig("log/acc_compare.jpg")

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), brier_best, label="DRST", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_fixR, label="ASG + Fixed ratio", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_cbst, label="ASG + CBST", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Brier score", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig("log/brier_compare.jpg")

def eval_model():
    from visda_exp import CONFIG, ImageClassdata, avh_score, accuracy_new, entropy
    from visda_exp import source_vis, beta_vis, alpha_vis, validate, AverageMeter, selftrain_dataloader
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
    _, val_loader = selftrain_dataloader()

    #resume_path = "runs/best_model_comp13/"
    resume_path = "runs/best_model_54/"
    print("=> Loading checkpoint '{}'".format(resume_path))
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
    top1_acc = AverageMeter()
    losses = AverageMeter()
    ce_loss = nn.CrossEntropyLoss()
    mis_ent, mis_num, brier_score, test_num = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1,))
            BATCH_SIZE = input.shape[0]
            test_num += BATCH_SIZE
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).to(DEVICE)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(),
                                     r_target.cuda()).detach()
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
                    i, len(val_loader), loss=losses, top1=top1_acc))
                print(r_target.reshape((-1,)))
    brier_score = brier_score / test_num
    misent = mis_ent / mis_num
    print(top1_acc.vec2sca_avg, top1_acc.avg)
    #print("Mis entropy: {}, Brier score: {}".format(mis_ent, brier_score))

def eval_single_model():
    from visda_exp import CONFIG, ImageClassdata, avh_score, accuracy_new, entropy
    from visda_exp import source_vis, validate, AverageMeter, selftrain_dataloader
    import math
    from sklearn.metrics import brier_score_loss
    import torch.nn as nn
    from torchvision import models
    """
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/crst/"
    checkpoint = torch.load(resume_path + "model_best.pth.tar")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/cbst_asg_model/"
    checkpoint = torch.load(resume_path + "epoch_11checkpoint.pth.tar")
    state = model.state_dict()
    for key in state.keys():
        if "model." + key in checkpoint.keys():
            state[key] = checkpoint["model." + key]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model.load_state_dict(state, strict=True)
    """
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/res101_asg/res101_vista17_best.pth.tar"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path)
    state_pre = checkpoint["state_dict"]
    state = model.state_dict()
    for key in state.keys():
        if key in state_pre.keys():
            state[key] = state_pre[key]
        elif key == "fc.0.weight":
            state[key] = state_pre["fc_new.0.weight"]
        elif key == "fc.0.bias":
            state[key] = state_pre["fc_new.0.bias"]
        elif key == "fc.2.weight":
            state[key] = state_pre["fc_new.2.weight"]
        elif key == "fc.2.bias":
            state[key] = state_pre["fc_new.2.bias"]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model.load_state_dict(state, strict=True)

    print("=> loaded checkpoint '{}'".format(resume_path))
    model = model.to(DEVICE)
    model.eval()

    _, val_loader = selftrain_dataloader()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_func = nn.CrossEntropyLoss()

    brier_score, test_num, mis_ent, mis_num = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, label, _) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            test_num += input.shape[0]

            # compute output
            output = model(input)
            label = label.reshape(-1, )
            loss = loss_func(output, label.long())
            prediction_t = F.softmax(output, dim=1)

            # measure accuracy and record loss
            prec1, gt_num = accuracy_new(output.data, label.long(), CONFIG["num_class"], topk=(1,))
            losses.update(loss, input.size(0))
            top1.update(prec1[0], gt_num[0])
            mis_idx = (torch.argmax(prediction_t, dim=1) != label.long()).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            mis_ent += entropy(mis_pred) / math.log(CONFIG["num_class"], 2)
            mis_num += mis_idx.shape[0]
            label_onehot = torch.zeros(output.shape)
            label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
            for j in range(input.shape[0]):
                brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j].cpu().numpy())

            if i % CONFIG["print_freq"] == 0:
                print('Test: [{0}/{1}]\n'
                      'Loss {loss.val:.4f}\n ({loss.avg:.4f})\n'
                      'Prec@1-per-class {top1.val}\n ({top1.avg})\n'
                      'Prec@1-mean {top1.vec2sca_val:3f} ({top1.vec2sca_avg:.3f})'.format(
                    i, len(val_loader), loss=losses,
                    top1=top1))
    prec = top1.vec2sca_avg

    print("Best precision: ", prec)
    print("Class-specific precision", top1.avg)
    print("")

def ablation_st():
    dic_r0 = np.load("log/var3_r_zero.npy")
    dic_R1 = np.load("log/var3_R_zero.npy")
    dic_best = np.load("log/var3_comp13.npy")

    acc_r0 = [x.cpu().numpy() for x in dic_r0["acc"]]
    acc_R1 = [x.cpu().numpy() for x in dic_R1["acc"]]
    acc_best = [x.cpu().numpy() for x in dic_best["acc"]]

    brier_r0 = dic_r0["brier"]
    brier_R1 = dic_R1["brier"]
    brier_best = dic_best["brier"]

    import os
    directory = "log/ablation/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    num_epoch = len(acc_best)
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), acc_best, label="DRST", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_r0, label="r=0", linewidth=3)
    plt.plot(np.arange(num_epoch), acc_R1, label="R=1", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Accuracy", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig(directory + "acc_rR.jpg")

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(num_epoch), brier_best, label="DRST", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_r0, label="r=0", linewidth=3)
    plt.plot(np.arange(num_epoch), brier_R1, label="R=1", linewidth=3)
    plt.tick_params(labelsize=16)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 24})
    plt.ylabel("Brier score", fontdict={"weight": "normal", "size": 24})
    plt.legend(shadow=True, fontsize='x-large')
    plt.title("")
    plt.savefig(directory + "brier_rR.jpg")

def plot_with_error_bar_acc():
    plt.figure(figsize=(15, 10))
    x = np.arange(1, 21)
    # ASG + CBST
    acc_cbst = np.zeros((5, 20))
    brier_cbst = np.zeros((5, 20))
    misent_cbst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/visda_cbst_asg"+str(i)+".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_cbst[i] = list_metrics["acc"]
        brier_cbst[i] = list_metrics["brier"]
        misent_cbst[i] = list_metrics["misent"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_cbst[:, j]))
        mean_brier.append(np.mean(brier_cbst[:, j]))
        mean_misent.append(np.mean(misent_cbst[:, j]))
        std_acc.append(np.std(acc_cbst[:, j], ddof=1))
        std_brier.append(np.std(brier_cbst[:, j], ddof=1))
        std_misent.append(np.std(misent_cbst[:, j], ddof=1))
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = np.array(mean_acc), np.array(mean_brier), np.array(mean_misent), np.array(std_acc), np.array(std_brier), np.array(std_misent)
    print("CBST mean {} and std {}".format(mean_acc, std_acc))
    #plt.errorbar(x, mean_acc, yerr=std_acc, c="b")
    plt.plot(x, mean_acc, marker="*", c="b", markersize=16, linewidth=6, label="CBST", linestyle="--")
    plt.fill_between(x, mean_acc-std_acc, mean_acc+std_acc, facecolor="b", alpha=0.25)

    # CRST
    ## misent and loss not recorded
    acc_crst = np.zeros((5, 20))
    brier_crst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/crst_log" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_crst[i] = list_metrics["acc"]
        brier_crst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_crst[:, j]))
        mean_brier.append(np.mean(brier_crst[:, j]))
        std_acc.append(np.std(acc_crst[:, j], ddof=1))
        std_brier.append(np.std(brier_crst[:, j], ddof=1))
    mean_acc, mean_brier, std_acc, std_brier = np.array(mean_acc), np.array(mean_brier), np.array(std_acc), np.array(std_brier)
    # plt.errorbar(x, mean_acc, yerr=std_acc, c="b")
    plt.plot(x, mean_acc, marker="*", c="gray", markersize=16, linewidth=6, label="CRST", linestyle="-.")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, facecolor="gray", alpha=0.25)

    # R=1

    # r=0


    # DRST
    acc_drst = np.zeros((5, 20))
    brier_drst = np.zeros((5, 20))
    misent_drst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/var3_seed" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_drst[i] = list_metrics["acc"]
        brier_drst[i] = list_metrics["brier"]
        misent_drst[i] = list_metrics["misent"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_drst[:, j]))
        mean_brier.append(np.mean(brier_drst[:, j]))
        mean_misent.append(np.mean(misent_drst[:, j]))
        std_acc.append(np.std(acc_drst[:, j], ddof=1))
        std_brier.append(np.std(brier_drst[:, j], ddof=1))
        std_misent.append(np.std(misent_drst[:, j], ddof=1))
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = np.array(mean_acc), np.array(
        mean_brier), np.array(mean_misent), np.array(std_acc), np.array(std_brier), np.array(std_misent)
    plt.plot(x, mean_acc, marker="o", c="r", markersize=16, linewidth=6, label="DRST")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, facecolor="r", alpha=0.25)

    plt.tick_params(labelsize=28)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 42})
    plt.ylabel("Test accuracy", fontdict={"weight": "normal", "size": 42})
    plt.xlim((1, 21))
    plt.legend(fontsize=36)
    plt.title("Test accuracy comparison", fontdict={"size":42})
    plt.savefig("log/finals/fig4_a.jpg", bbox_inches="tight")

def plot_with_error_bar_brier():
    plt.figure(figsize=(15, 10))
    x = np.arange(1, 21)
    # ASG + CBST
    brier_cbst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/visda_cbst_asg"+str(i)+".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_cbst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_cbst[:, j]))
        std_brier.append(np.std(brier_cbst[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="*", c="b", markersize=16, linewidth=6, label="CBST", linestyle="--")
    plt.fill_between(x, mean_brier-std_brier, mean_brier+std_brier, facecolor="b", alpha=0.25)

    # CRST
    ## misent and loss not recorded
    brier_crst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/crst_log" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_crst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_crst[:, j]))
        std_brier.append(np.std(brier_crst[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="*", c="gray", markersize=16, linewidth=6, label="CRST", linestyle="-.")
    plt.fill_between(x, mean_brier - std_brier, mean_brier + std_brier, facecolor="gray", alpha=0.25)

    # R=1

    # r=0


    # DRST
    brier_drst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/var3_seed" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_drst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_drst[:, j]))
        std_brier.append(np.std(brier_drst[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="o", c="r", markersize=16, linewidth=6, label="DRST")
    plt.fill_between(x, mean_brier - std_brier, mean_brier + std_brier, facecolor="r", alpha=0.25)

    plt.tick_params(labelsize=28)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 42})
    plt.ylabel("Test brier score", fontdict={"weight": "normal", "size": 42})
    plt.xlim((1, 21))
    plt.legend(fontsize=36)
    plt.title("Test brier score comparison", fontdict={"size": 42})
    plt.savefig("log/finals/fig4_b.jpg", bbox_inches="tight")

def plot_shift_st():
    arm_even = np.load("log/cov_shift/arm_total.npy")
    armst_even = np.load("log/cov_shift/armst_total.npy")
    arm_odd = np.load("log/cov_shift/arm_total_odd.npy")
    armst_odd = np.load("log/cov_shift/armst_total_odd.npy")
    arm = np.zeros(20)
    armst = np.zeros(20)
    for i in range(20):
        if i % 2 == 0:
            arm[i] = arm_odd[int(i/2)]
            armst[i] = armst_odd[int(i/2)]
        else:
            arm[i] = arm_even[int(i/2)]
            armst[i] = armst_even[int(i/2)]

    x = np.arange(1, 21, 1)
    plt.figure(figsize=(15, 10))
    plt.plot(x, arm, marker="o", c="g", markersize=16, linewidth=6, label="DRL", linestyle="--")
    plt.plot(x, armst, marker="*", c="orange", markersize=16, linewidth=6, label="DRST")

    plt.tick_params(labelsize=28)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 42})
    #plt.ylabel("Distribution gap", fontdict={"weight": "normal", "size": 42})
    plt.xlim((1, 21))
    plt.legend(fontsize=36)
    #plt.title("Comparison of distribution gap", fontdict={"size": 42})
    plt.savefig("log/finals/fig4_c2.jpg", bbox_inches="tight")

def office_drl_boost():
    x = np.arange(1, 21)
    baseline = np.load("log/office_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    acc_base = baseline["acc"]
    brier_base = baseline["brier"]

    acc_boost = np.load("log/drl_boost_prec_office.npy", allow_pickle=True)
    brier_boost = np.load("log/drl_boost_brier_office.npy", allow_pickle=True)

    fig, ax1 = plt.subplots(figsize=(20, 15))
    ax2 = ax1.twinx()
    ax1.plot(x, acc_base, linestyle="dotted", linewidth=6, marker="o", markersize=16, color="r", label="Source acc")
    ax1.plot(x, acc_boost, linewidth=6, marker="*", markersize=16, color="r", label="DRL acc")

    ax2.plot(x, brier_base, linestyle="dotted", linewidth=6, marker="^", markersize=16, color="b", label="Source brier")
    ax2.plot(x, brier_boost, linewidth=6, marker="v", markersize=16, color="b", label = "DRL brier")

    ax1.set_xlabel("Epochs", fontdict={"size": 60})
    ax1.set_ylabel("Accuracy", color="r", fontdict={"size": 60})
    ax2.set_ylabel("Brier score", color="b", fontdict={"size": 60})
    ax1.tick_params(labelsize=42)
    ax2.tick_params(labelsize=42)
    ax1.set_xlim((1, 20))
    ax2.set_xlim((1, 20))
    fig.legend(loc=(0.64, 0.39), fontsize=33)
    plt.title("Accuracy and brier score plot on Office31 (A → W)", fontdict={"size":44})
    plt.savefig("log/finals/fig6_b.jpg", bbox_inches="tight")

def home_drl_boost():
    x = np.arange(1, 21)
    baseline = np.load("log/OfficeHome_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    acc_base = baseline["acc"]
    brier_base = baseline["brier"]

    acc_boost = np.load("log/drl_boost_prec_home.npy", allow_pickle=True)
    brier_boost = np.load("log/drl_boost_brier_home.npy", allow_pickle=True)

    fig, ax1 = plt.subplots(figsize=(20, 15))
    ax2 = ax1.twinx()
    ax1.plot(x, acc_base, linestyle="dotted", linewidth=6, marker="o", markersize=16, color="r", label="Source acc")
    ax1.plot(x, acc_boost, linewidth=6, marker="*", markersize=16, color="r", label="DRL acc")

    ax2.plot(x, brier_base, linestyle="dotted", linewidth=6, marker="^", markersize=16, color="b", label="Source brier")
    ax2.plot(x, brier_boost, linewidth=6, marker="v", markersize=16, color="b", label = "DRL brier")

    ax1.set_xlabel("Epochs", fontdict={"size": 48})
    ax1.set_ylabel("Accuracy", color="r", fontdict={"size": 48})
    ax2.set_ylabel("Brier score", color="b", fontdict={"size": 48})
    ax1.tick_params(labelsize=42)
    ax2.tick_params(labelsize=42)
    ax1.set_xlim((1, 20))
    ax2.set_xlim((1, 20))
    fig.legend(loc=(0.64, 0.32), fontsize=33)
    plt.title("Accuracy and brier score plot on OfficeHome (P → A)", fontdict={"size":42})
    plt.savefig("log/finals/fig6_c.jpg", bbox_inches="tight")

def visda_drl_boost():
    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18])
    #x = x - 1
    #acc_base = [acc_base[i] for i in x]
    #brier_base = [brier_base[i] for i in x]
    x = np.arange(1, 21)
    baseline = np.load("log/visda_iid_used.npy", allow_pickle=True)
    baseline = baseline.item()
    acc_base = baseline["acc"]
    brier_base = baseline["brier"]

    acc_boost = np.array(
        [63.170986, 64.5393, 64.40287, 63.116863, 63.505585, 63.875744, 63.54494, 64.23318,
         65.79307, 63.992, 65.448524, 63.96661, 63.94551, 64.8043, 64.610306, 65.59442,
         63.77011, 64.217354, 62.656, 63.558])
    brier_boost = np.array(
        [0.03949574, 0.03801172, 0.03889442, 0.03993579, 0.03947036, 0.03869169, 0.03938174,
         0.03941744, 0.0373686, 0.04003927, 0.03764815, 0.03923099, 0.03895577, 0.03832594,
         0.03916568, 0.03747537, 0.03939206, 0.03919888, 0.04093537, 0.03894782])

    fig, ax1 = plt.subplots(figsize=(20, 15))
    ax2 = ax1.twinx()
    ax1.plot(x, acc_base, linestyle="dotted", linewidth=6, marker="o", markersize=16, color="r", label="Source acc")
    ax1.plot(x, acc_boost, linewidth=6, marker="*", markersize=16, color="r", label="DRL acc")

    ax2.plot(x, brier_base, linestyle="dotted", linewidth=6, marker="^", markersize=16, color="b", label="Source brier")
    ax2.plot(x, brier_boost, linewidth=6, marker="v", markersize=16, color="b", label="DRL brier")

    ax1.set_xlabel("Epochs", fontdict={"size": 48})
    ax1.set_ylabel("Accuracy", color="r", fontdict={"size": 48})
    ax2.set_ylabel("Brier score", color="b", fontdict={"size": 48})
    ax1.tick_params(labelsize=42)
    ax2.tick_params(labelsize=42)
    ax1.set_xlim((1, 20))
    ax2.set_xlim((1, 20))
    fig.legend(loc=(0.64, 0.45), fontsize=33)
    plt.title("Accuracy and brier score plot on VisDA2017", fontdict={"size": 44})
    plt.savefig("log/finals/fig5_b.jpg", bbox_inches="tight")

def drl_boost_misent():
    x = np.arange(1, 21)
    baseline = np.load("log/office_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    misent_base = baseline["misent"]
    misent_boost = np.load("log/drl_boost_misent_office.npy", allow_pickle=True)
    plt.figure(figsize=(20, 15))
    plt.plot(x, misent_base, linewidth=12, marker="v", markersize=24, color="b", label="Source")
    plt.plot(x, misent_boost, linewidth=12, marker="o", markersize=24, color="r", label="DRL")
    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Misclassification entropy", fontdict={"size": 60})
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    #plt.legend(loc=(0.6, 0.39), fontsize=33)
    #plt.title("Misclassification entropy on Office31 (A → W)", fontdict={"size": 44})
    plt.savefig("log/finals/appendix_office.jpg", bbox_inches="tight")

    x = np.arange(1, 21)
    baseline = np.load("log/OfficeHome_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    misent_base = baseline["misent"]
    misent_boost = np.load("log/drl_boost_misent_home.npy", allow_pickle=True)
    plt.figure(figsize=(20, 15))
    plt.plot(x, misent_base, linewidth=12, marker="v", markersize=24, color="b", label="Source")
    plt.plot(x, misent_boost, linewidth=12, marker="o", markersize=24, color="r", label="DRL")
    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Misclassification entropy", fontdict={"size": 60})
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    #plt.legend(loc=(0.6, 0.39), fontsize=33)
    #plt.title("Misclassification entropy on OfficeHome (P → A)", fontdict={"size": 44})
    plt.savefig("log/finals/appendix_home.jpg", bbox_inches="tight")

    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18])
    #x = x - 1
    #misent_base = [misent_base[i] for i in x]
    x = np.arange(1, 21)
    baseline = np.load("log/visda_iid_used.npy", allow_pickle=True)
    baseline = baseline.item()
    misent_base = baseline["misent"]
    misent_boost = [0.5632861, 0.6304906, 0.559204, 0.5142653, 0.51861703, 0.64822364, 0.5326741, 0.55202854,
                    0.5952068, 0.6729688, 0.5943873 , 0.57159483, 0.5444791 , 0.5977465, 0.75289017,
                    0.5971629, 0.5696043, 0.5463317, 0.526643455, 0.582737]
    plt.figure(figsize=(20, 15))
    plt.plot(x, misent_base, linewidth=12, marker="v", markersize=24, color="b", label="Source")
    plt.plot(x, misent_boost, linewidth=12, marker="o", markersize=24, color="r", label="DRL")
    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Misclassification entropy", fontdict={"size": 60})
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    plt.legend(loc=(0.6, 0.39), fontsize=33)
    #plt.title("Misclassification entropy on VisDA2017", fontdict={"size": 44})
    plt.savefig("log/finals/appendix_visda.jpg", bbox_inches="tight")

def plot_with_error_bar_acc_new():
    plt.figure(figsize=(15, 10))
    x = np.arange(1, 21)
    # ASG + CBST
    acc_cbst = np.zeros((5, 20))
    brier_cbst = np.zeros((5, 20))
    misent_cbst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/visda_cbst_asg"+str(i)+".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_cbst[i] = list_metrics["acc"]
        brier_cbst[i] = list_metrics["brier"]
        misent_cbst[i] = list_metrics["misent"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_cbst[:, j]))
        mean_brier.append(np.mean(brier_cbst[:, j]))
        mean_misent.append(np.mean(misent_cbst[:, j]))
        std_acc.append(np.std(acc_cbst[:, j], ddof=1))
        std_brier.append(np.std(brier_cbst[:, j], ddof=1))
        std_misent.append(np.std(misent_cbst[:, j], ddof=1))
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = np.array(mean_acc), np.array(mean_brier), np.array(mean_misent), np.array(std_acc), np.array(std_brier), np.array(std_misent)
    print("CBST mean {} and std {}".format(mean_acc, std_acc))
    #plt.errorbar(x, mean_acc, yerr=std_acc, c="b")
    plt.plot(x, mean_acc, marker="*", c="b", markersize=16, linewidth=6, label="CBST", linestyle="--")
    plt.fill_between(x, mean_acc-std_acc, mean_acc+std_acc, facecolor="b", alpha=0.25)

    # CRST
    ## misent and loss not recorded
    acc_crst = np.zeros((5, 20))
    brier_crst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/crst_log" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_crst[i] = list_metrics["acc"]
        brier_crst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_crst[:, j]))
        mean_brier.append(np.mean(brier_crst[:, j]))
        std_acc.append(np.std(acc_crst[:, j], ddof=1))
        std_brier.append(np.std(brier_crst[:, j], ddof=1))
    mean_acc, mean_brier, std_acc, std_brier = np.array(mean_acc), np.array(mean_brier), np.array(std_acc), np.array(std_brier)
    # plt.errorbar(x, mean_acc, yerr=std_acc, c="b")
    plt.plot(x, mean_acc, marker="*", c="gray", markersize=16, linewidth=6, label="CRST", linestyle="-.")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, facecolor="gray", alpha=0.25)

    # R=1
    acc_R1 = np.zeros((5, 20))
    brier_R1 = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/R_one/" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_R1[i] = list_metrics["acc"]
        brier_R1[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_R1[:, j]))
        mean_brier.append(np.mean(brier_R1[:, j]))
        std_acc.append(np.std(acc_R1[:, j], ddof=1))
        std_brier.append(np.std(brier_R1[:, j], ddof=1))
    mean_acc, mean_brier, std_acc, std_brier = np.array(mean_acc), np.array(mean_brier), np.array(std_acc), np.array(
        std_brier)
    plt.plot(x, mean_acc, marker="*", c="orange", markersize=16, linewidth=6, label="DRST with R=1", linestyle=":")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, facecolor="yellow", alpha=0.25)

    # r=0
    acc_r0 = np.zeros((3, 20))
    brier_r0 = np.zeros((3, 20))
    list_num = [0, 2, 3]
    for i in range(3):
        list_metrics = np.load("log/r_zero/" + str(list_num[i]) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_r0[i] = list_metrics["acc"]
        brier_r0[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_r0[:, j]))
        mean_brier.append(np.mean(brier_r0[:, j]))
        std_acc.append(np.std(acc_r0[:, j], ddof=1))
        std_brier.append(np.std(brier_r0[:, j], ddof=1))
    mean_acc, mean_brier, std_acc, std_brier = np.array(mean_acc), np.array(mean_brier), np.array(std_acc), np.array(
        std_brier)
    plt.plot(x, mean_acc, marker="*", c="darkgreen", markersize=16, linewidth=6, label="DRST with r=0", linestyle=":")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, facecolor="green", alpha=0.25)


    # DRST
    acc_drst = np.zeros((5, 20))
    brier_drst = np.zeros((5, 20))
    misent_drst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/var3_seed" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        acc_drst[i] = list_metrics["acc"]
        brier_drst[i] = list_metrics["brier"]
        misent_drst[i] = list_metrics["misent"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_acc.append(np.mean(acc_drst[:, j]))
        mean_brier.append(np.mean(brier_drst[:, j]))
        mean_misent.append(np.mean(misent_drst[:, j]))
        std_acc.append(np.std(acc_drst[:, j], ddof=1))
        std_brier.append(np.std(brier_drst[:, j], ddof=1))
        std_misent.append(np.std(misent_drst[:, j], ddof=1))
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = np.array(mean_acc), np.array(
        mean_brier), np.array(mean_misent), np.array(std_acc), np.array(std_brier), np.array(std_misent)
    plt.plot(x, mean_acc, marker="o", c="r", markersize=16, linewidth=6, label="DRST")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, facecolor="r", alpha=0.25)

    plt.tick_params(labelsize=28)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 42})
    #plt.ylabel("Test accuracy", fontdict={"weight": "normal", "size": 42})
    plt.xlim((1, 21))
    plt.legend(fontsize=36)
    #plt.title("Test accuracy comparison", fontdict={"size":42})
    plt.savefig("log/finals/fig4_a_new2.jpg", bbox_inches="tight")

def plot_with_error_bar_brier_new():
    plt.figure(figsize=(15, 10))
    x = np.arange(1, 21)
    # ASG + CBST
    brier_cbst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/visda_cbst_asg"+str(i)+".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_cbst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_cbst[:, j]))
        std_brier.append(np.std(brier_cbst[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="*", c="b", markersize=16, linewidth=6, label="CBST", linestyle="--")
    plt.fill_between(x, mean_brier-std_brier, mean_brier+std_brier, facecolor="b", alpha=0.25)

    # CRST
    ## misent and loss not recorded
    brier_crst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/crst_log" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_crst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_crst[:, j]))
        std_brier.append(np.std(brier_crst[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="*", c="gray", markersize=16, linewidth=6, label="CRST", linestyle="-.")
    plt.fill_between(x, mean_brier - std_brier, mean_brier + std_brier, facecolor="gray", alpha=0.25)

    # R=1
    brier_R1 = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/R_one/" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_R1[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_R1[:, j]))
        std_brier.append(np.std(brier_R1[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="*", c="orange", markersize=16, linewidth=6, label="DRST with R=1", linestyle=":")
    plt.fill_between(x, mean_brier - std_brier, mean_brier + std_brier, facecolor="yellow", alpha=0.25)

    # r=0
    brier_r0 = np.zeros((3, 20))
    list_num = [0, 2, 3]
    for i in range(3):
        list_metrics = np.load("log/r_zero/" + str(list_num[i]) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_r0[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_r0[:, j]))
        std_brier.append(np.std(brier_r0[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="*", c="darkgreen", markersize=16, linewidth=6, label="DRST with r=0", linestyle=":")
    plt.fill_between(x, mean_brier - std_brier, mean_brier + std_brier, facecolor="green", alpha=0.25)

    # DRST
    brier_drst = np.zeros((5, 20))
    for i in range(5):
        list_metrics = np.load("log/var3_seed" + str(i) + ".npy", allow_pickle=True)
        list_metrics = list_metrics.item()
        brier_drst[i] = list_metrics["brier"]
    mean_acc, mean_brier, mean_misent, std_acc, std_brier, std_misent = [], [], [], [], [], []
    for j in range(20):
        mean_brier.append(np.mean(brier_drst[:, j]))
        std_brier.append(np.std(brier_drst[:, j], ddof=1))
    mean_brier, std_brier = np.array(mean_brier), np.array(std_brier)
    plt.plot(x, mean_brier, marker="o", c="r", markersize=16, linewidth=6, label="DRST")
    plt.fill_between(x, mean_brier - std_brier, mean_brier + std_brier, facecolor="r", alpha=0.25)

    plt.tick_params(labelsize=28)
    plt.xlabel("Epoch", fontdict={"weight": "normal", "size": 42})
    #plt.ylabel("Test brier score", fontdict={"weight": "normal", "size": 42})
    plt.xlim((1, 21))
    plt.legend(fontsize=36)
    #plt.title("Test brier score comparison", fontdict={"size": 42})
    plt.savefig("log/finals/fig4_b_new2.jpg", bbox_inches="tight")

def office_drl_boost_rebuttal(source, target):
    x = np.arange(1, 21)
    baseline = np.load("log/office_"+source[0]+target[0]+".npy", allow_pickle=True)
    baseline = baseline.item()
    acc_base = baseline["acc"]
    brier_base = baseline["brier"]

    acc_boost = np.load("log/rebuttal/drl_prec_"+source[0]+target[0]+".npy", allow_pickle=True)
    brier_boost = np.load("log/rebuttal/drl_brier"+source[0]+target[0]+".npy", allow_pickle=True)

    fig, ax1 = plt.subplots(figsize=(20, 15))
    ax2 = ax1.twinx()
    ax1.plot(x, acc_base, linestyle="dotted", linewidth=6, marker="o", markersize=16, color="r", label="Source acc")
    ax1.plot(x, acc_boost, linewidth=6, marker="*", markersize=16, color="r", label="DRL acc")

    ax2.plot(x, brier_base, linestyle="dotted", linewidth=6, marker="^", markersize=16, color="b", label="Source brier")
    ax2.plot(x, brier_boost, linewidth=6, marker="v", markersize=16, color="b", label = "DRL brier")

    ax1.set_xlabel("Epochs", fontdict={"size": 60})
    ax1.set_ylabel("Accuracy", color="r", fontdict={"size": 60})
    ax2.set_ylabel("Brier score", color="b", fontdict={"size": 60})
    ax1.tick_params(labelsize=42)
    ax2.tick_params(labelsize=42)
    ax1.set_xlim((1, 20))
    ax2.set_xlim((1, 20))
    fig.legend(loc=(0.64, 0.39), fontsize=33)
    plt.title("Accuracy and brier score plot on Office31 ("+source[0].upper()+" → "+target[0].upper()+")", fontdict={"size":44})
    plt.savefig("log/finals/fig_"+source[0]+target[0]+".jpg", bbox_inches="tight")

class ResModel(torch.nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        self.features = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Sequential()
        self.classifier = nn.Linear(num_ftrs, 31) #num_ftr = 2048

    def forward(self, x_s):
        phi = self.features(x_s)
        phi = phi.view(-1, 2048)
        y = self.classifier(phi)
        return phi, y

def reliability_plot_office_rebuttal(source, target):
    """
    from office_exp import source_office, alpha_office, beta_office, dataloader_office
    _, val_loader = dataloader_office("office/"+source, "office/"+target)
    # get the softmax
    ## load model
    model_source = source_office(31)
    # this might be broken... office_iid, check the acc and select the epoch
    resume_path = "runs/office_iid/"+source[0]+target[0]+"_19.pth.tar"
    model_source.load_state_dict(torch.load(resume_path))
    model_source = model_source.to(DEVICE)
    model_source.eval()

    drst_alpha = alpha_office(31)
    drst_beta = beta_office()
    #resume_path_alpha = "runs/office_best/"+source[0]+target[0]+"_drl_alpha.pth.tar"
    #resume_path_beta = "runs/office_best/"+source[0]+target[0]+"_drl_beta.pth.tar"
    resume_path_alpha = "runs/office_best/" + source[0] + target[0] + "_alpha.pth.tar"
    resume_path_beta = "runs/office_best/" + source[0] + target[0] + "_beta.pth.tar"
    drst_alpha.load_state_dict(torch.load(resume_path_alpha))
    drst_beta.load_state_dict(torch.load(resume_path_beta))
    drst_alpha = drst_alpha.to(DEVICE)
    drst_beta = drst_beta.to(DEVICE)
    drst_alpha.eval()
    drst_beta.eval()

    ts_model = source_office(31)
    resume_path = "runs/office_iid/"+source[0]+target[0]+"_19.pth.tar"
    ts_model.load_state_dict(torch.load(resume_path))
    from temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(ts_model)
    scaled_model.set_temperature(val_loader)
    torch.save(scaled_model.state_dict(), "runs/office_iid/ts_"+source[0]+target[0]+"_model.pth.tar")
    scaled_model = scaled_model.to(DEVICE)
    scaled_model.eval()

    source_pred = np.zeros([1, 31])
    ts_pred = np.zeros([1, 31])
    drst_pred = np.zeros([1, 31])
    label_rec = np.zeros([1])
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1, ))
            BATCH_SIZE = input.shape[0]
            label_rec = np.concatenate([label_rec, label.cpu().numpy()], axis=0)

            output = model_source(input).detach()
            source_softmax = F.softmax(output, dim=1)
            source_softmax = source_softmax.cpu().numpy()
            source_pred = np.concatenate([source_pred, source_softmax], axis=0)

            ts_out = scaled_model(input).detach()
            ts_softmax = F.softmax(ts_out, dim=1)
            ts_softmax = ts_softmax.cpu().numpy()
            ts_pred = np.concatenate([ts_pred, ts_softmax], axis=0)

            pred = F.softmax(drst_beta(input, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = drst_alpha(input, torch.ones((BATCH_SIZE, 31)).cuda(),
                                    r_target.cuda()).detach()
            drst_softmax = F.softmax(target_out, dim=1)
            drst_softmax = drst_softmax.cpu().numpy()
            drst_pred = np.concatenate([drst_pred, drst_softmax], axis=0)
    source_pred = source_pred[1:]
    ts_pred = ts_pred[1:]
    drst_pred = drst_pred[1:]
    label_rec = label_rec[1:]
    np.save("log/office_src_softmax"+source[0]+target[0]+".npy", source_pred)
    np.save("log/office_ts_softmax"+source[0]+target[0]+".npy", ts_pred)
    np.save("log/office_drst_softmax"+source[0]+target[0]+".npy", drst_pred)
    np.save("log/office_label"+source[0]+target[0]+".npy", label_rec)
    """
    intervals = [0, 0.85, 0.87, 0.9, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
    #intervals = [0, 0.7, 0.8, 0.83, 0.87, 0.91, 0.93, 0.95, 0.97, 0.98, 1.0]
    #intervals = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.83, 1.0]
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=5)
    label = np.load("log/office_label"+source[0]+target[0]+".npy")
    # IID
    iid_softmax = np.load("log/office_src_softmax"+source[0]+target[0]+".npy", allow_pickle=True)
    iid_conf = np.max(iid_softmax, axis=1)
    iid_acc = (np.argmax(iid_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((iid_conf >= intervals[i]) == (iid_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(iid_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(iid_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Source", marker="^", color="skyblue", linewidth=8, markersize=16)

    # TS
    ts_softmax = np.load("log/office_ts_softmax"+source[0]+target[0]+".npy", allow_pickle=True)
    ts_conf = np.max(ts_softmax, axis=1)
    #print("TS conf:", ts_conf)
    ts_acc = (np.argmax(ts_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((ts_conf >= intervals[i]) == (ts_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(ts_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(ts_conf[int_idx]) - np.sum(ts_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="TS", marker="o", color="limegreen", linewidth=8, markersize=16)

    # DRST
    drst_softmax = np.load("log/office_drst_softmax"+source[0]+target[0]+".npy", allow_pickle=True)
    drst_conf = np.max(drst_softmax, axis=1)
    drst_acc = (np.argmax(drst_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((drst_conf >= intervals[i]) == (drst_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(drst_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(drst_conf[int_idx]) - np.sum(drst_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="DRL", marker="*", color="coral", linewidth=8, markersize=16)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=28)
    plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 42})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 42})
    #plt.legend(fontsize=36, loc=4)
    #plt.title("Reliability plot on Office31 ("+source[0].upper() + " → "+target[0].upper()+")", fontdict={"weight": "normal", "size": 42})
    plt.savefig("log/finals/relia_"+source[0]+target[0]+".jpg")

def reliability_plot_office_rebuttal_with_vada(source, target):
    from office_exp import dataloader_office
    _, val_loader = dataloader_office("office/" + source, "office/" + target)
    # get the softmax
    ## load model
    model_vada = ResModel()
    # this might be broken... office_iid, check the acc and select the epoch
    #resume_path = "salad/examples/dirtt/log/20210828-030916_VADASolver/20210828-030916-checkpoint-ep19.pth"
    #resume_path = "salad/examples/dirtt/log_ad/20210828-123821_VADASolver/20210828-123821-checkpoint-ep19.pth"
    #resume_path = "salad/examples/dirtt/log_wa/20210828-125710_VADASolver/20210828-125710-checkpoint-ep19.pth"
    resume_path = "salad/examples/dirtt/log_wa/20210828-213223_VADASolver/20210828-213223-checkpoint-ep19.pth"
    model_vada = torch.load(resume_path)
    model_vada = model_vada.to(DEVICE)
    model_vada.eval()

    vada_pred = np.zeros([1, 31])
    label_vada = np.zeros([1])
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            label = label.reshape((-1,))
            BATCH_SIZE = input.shape[0]
            label_vada = np.concatenate([label_vada, label.cpu().numpy()], axis=0)

            _, output = model_vada(input)
            output = output.detach()
            vada_softmax = F.softmax(output, dim=1)
            vada_softmax = vada_softmax.cpu().numpy()
            vada_pred = np.concatenate([vada_pred, vada_softmax], axis=0)
            if i % 20 == 0:
                print("{} batches finished".format(i))

    vada_pred = vada_pred[1:]
    label_vada = label_vada[1:]
    np.save("log/office_vada_softmax" + source[0] + target[0] + ".npy", vada_pred)
    np.save("log/office_label_vada" + source[0] + target[0] + ".npy", label_vada)

    intervals = [0, 0.85, 0.87, 0.9, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
    plt.figure(figsize=(21, 15))
    plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=5)
    #label = np.load("log/office_label.npy")
    label = np.load("log/office_label" + source[0] + target[0] + ".npy")
    # IID
    #iid_softmax = np.load("log/office_src_softmax.npy", allow_pickle=True)
    iid_softmax = np.load("log/office_src_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    iid_conf = np.max(iid_softmax, axis=1)
    iid_acc = (np.argmax(iid_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((iid_conf >= intervals[i]) == (iid_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(iid_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(iid_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Source", marker="^", color="skyblue", linewidth=18,
             markersize=36)

    # TS
    #ts_softmax = np.load("log/office_ts_softmax.npy", allow_pickle=True)
    ts_softmax = np.load("log/office_ts_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    ts_conf = np.max(ts_softmax, axis=1)
    ts_acc = (np.argmax(ts_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((ts_conf >= intervals[i]) == (ts_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(ts_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(ts_conf[int_idx]) - np.sum(ts_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="TS", marker="v", color="limegreen", linewidth=18,
             markersize=36)

    # DRST
    #drst_softmax = np.load("log/office_drst_softmax.npy", allow_pickle=True)
    drst_softmax = np.load("log/office_drst_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    drst_conf = np.max(drst_softmax, axis=1)
    drst_acc = (np.argmax(drst_softmax, axis=1) == label)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((drst_conf >= intervals[i]) == (drst_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(drst_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(drst_conf[int_idx]) - np.sum(drst_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="DRL", marker="o", color="coral", linewidth=18, markersize=36)

    # VADA
    label_vada = np.load("log/office_label_vada"+source[0]+target[0]+".npy")
    vada_softmax = np.load("log/office_vada_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    vada_conf = np.max(vada_softmax, axis=1)
    vada_acc = (np.argmax(vada_softmax, axis=1) == label_vada)
    x_value = []
    y_value = []
    for i in range(len(intervals) - 1):
        int_idx = ((vada_conf >= intervals[i]) == (vada_conf < intervals[i + 1]))  # 0 and 1
        x_value.append(np.sum(vada_conf[int_idx]) / np.sum(int_idx))
        y_value.append((np.sum(vada_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
    print(y_value)
    plt.plot(np.arange(0, 100, 10) / 100., y_value, label="VADA", marker="s", color="yellow", linewidth=18,
             markersize=36)

    x_value = [str(x)[:5] for x in x_value]
    x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
    plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
    plt.tick_params(labelsize=28)
    plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 42})
    plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 42})
    plt.legend(fontsize=36, loc=4)
    #plt.title("Reliability plot on Office31 ("+source[0].upper() + " → "+target[0].upper()+")", fontdict={"weight": "normal", "size": 42})
    plt.savefig("log/relia_"+source[0]+target[0]+"_vada2.jpg")

def visda_drl_boost_brier():
    #x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18])
    #x = x - 1
    #acc_base = [acc_base[i] for i in x]
    #brier_base = [brier_base[i] for i in x]
    x = np.arange(1, 21)
    baseline = np.load("log/visda_iid_used.npy", allow_pickle=True)
    baseline = baseline.item()
    brier_base = baseline["brier"]

    brier_boost = np.array(
        [0.03949574, 0.03801172, 0.03889442, 0.03993579, 0.03947036, 0.03869169, 0.03938174,
         0.03941744, 0.0373686, 0.04003927, 0.03764815, 0.03923099, 0.03895577, 0.03832594,
         0.03916568, 0.03747537, 0.03939206, 0.03919888, 0.04093537, 0.03894782])

    fig = plt.figure(figsize=(20, 15))

    plt.plot(x, brier_base, linewidth=9, marker='v', markersize=36, color="r", label="Source")
    plt.plot(x, brier_boost, linewidth=9, marker='o', markersize=36, color="b", label="DRL")

    plt.xlabel("Epochs", fontsize=72)
    plt.ylabel("Brier score", fontsize=72)
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    #fig.legend(loc=(0.64, 0.45), fontsize=33)
    #fig.legend(fontsize=42, loc=(0.72, 0.48))
    #plt.title("Brier score over epochs", fontdict={"size": 56})
    fig.legend(loc='center left', bbox_to_anchor=(0.9, 0.50), ncol=1, fontsize=72)
    plt.savefig("log/finals/brier_visda.jpg", bbox_inches="tight")

def office_drl_boost_brier():
    x = np.arange(1, 21)
    baseline = np.load("log/office_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    brier_base = baseline["brier"]
    brier_boost = np.load("log/drl_boost_brier_office.npy", allow_pickle=True)

    fig = plt.figure(figsize=(20, 15))

    plt.plot(x, brier_base, linewidth=9, marker='v', markersize=36, color="r", label="Source")
    plt.plot(x, brier_boost, linewidth=9, marker='o', markersize=36, color="b", label="DRL")

    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Brier score", fontdict={"size": 72})
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    #fig.legend(loc=(0.72, 0.48), fontsize=42)
    #plt.title("Brier score over epochs", fontdict={"size":56})
    plt.savefig("log/finals/brier_aw.jpg", bbox_inches="tight")

def home_drl_boost_brier():
    x = np.arange(1, 21)
    baseline = np.load("log/OfficeHome_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    brier_base = baseline["brier"]

    brier_boost = np.load("log/drl_boost_brier_home.npy", allow_pickle=True)

    fig = plt.figure(figsize=(20, 15))

    plt.plot(x, brier_base, linewidth=9, marker='v', markersize=36, color="r", label="Source")
    plt.plot(x, brier_boost, linewidth=9, marker='o', markersize=36, color="b", label="DRL")

    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Brier score", fontdict={"size": 72})
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    #fig.legend(loc=(0.72, 0.48), fontsize=42)
    #fig.legend(loc='center left', frameon=True, framealpha=1, bbox_to_anchor=(0.28, 0.91), ncol=2, fontsize=42)
    #plt.title("Brier score over epochs", fontdict={"size":56})
    plt.savefig("log/finals/brier_pa.jpg", bbox_inches="tight")

def office_brier_appendix(source, target):
    x = np.arange(1, 21)
    baseline = np.load("log/office_"+source[0]+target[0]+".npy", allow_pickle=True)
    #baseline = np.load("log/office_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    brier_base = baseline["brier"]

    brier_boost = np.load("log/rebuttal/drl_brier"+source[0]+target[0]+".npy", allow_pickle=True)
    #brier_boost = np.load("log/drl_boost_brier_office.npy", allow_pickle=True)
    brier_bayes = np.load("models/"+source[0]+target[0]+"_brier.npy")[:20]

    plt.figure(figsize=(20, 15))

    plt.plot(x, brier_base, linewidth=12, marker="v", markersize=28, color="b", label="Source")
    plt.plot(x, brier_bayes, linewidth=12, marker="*", markersize=28, color="g", label="LL-SVI")
    plt.plot(x, brier_boost, linewidth=12, marker="o", markersize=28, color="r", label="DRL")

    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Brier score", fontdict={"size": 72})
    plt.tick_params(labelsize=54)
    plt.xlim((1, 20))
    plt.legend(loc=(0.3, 0.39), ncol=3, fontsize=33)
    #plt.title("Accuracy and brier score plot on Office31 ("+source[0].upper()+" → "+target[0].upper()+")", fontdict={"size":44})
    plt.savefig("log/finals/fig_brier_"+source[0]+target[0]+"_app.jpg", bbox_inches="tight")

def office_acc_appendix(source, target):
    x = np.arange(1, 21)
    baseline = np.load("log/office_"+source[0]+target[0]+".npy", allow_pickle=True)
    #baseline = np.load("log/office_iid.npy", allow_pickle=True)
    baseline = baseline.item()
    acc_base = baseline["acc"]

    acc_boost = np.load("log/rebuttal/drl_prec_"+source[0]+target[0]+".npy", allow_pickle=True)
    #acc_boost = np.load("log/drl_boost_prec_office.npy", allow_pickle=True)
    acc_bayes = np.load("models/"+source[0]+target[0]+"_acc_new.npy")[:20] * 100

    plt.figure(figsize=(20, 15))
    plt.plot(x, acc_base, linewidth=12, marker="v", markersize=28, color="b", label="Source")
    plt.plot(x, acc_boost, linewidth=12, marker="o", markersize=28, color="r", label="DRL")
    plt.plot(x, acc_bayes, linewidth=12, marker="*", markersize=28, color="g", label="LL-SVI")

    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Accuracy", fontdict={"size": 72})
    plt.tick_params(labelsize=54)
    plt.xlim((1, 20))
    #fig.legend(loc=(0.64, 0.39), fontsize=33)
    #plt.title("Accuracy and brier score plot on Office31 ("+source[0].upper()+" → "+target[0].upper()+")", fontdict={"size":44})
    plt.savefig("log/finals/fig_acc_"+source[0]+target[0]+"_app.jpg", bbox_inches="tight")

def ece_compare_appendix(source, target):
    from imagenet_train import ece_score
    bins = 15
    #label = np.load("log/office_label" + source[0] + target[0] + ".npy")
    #iid_softmax = np.load("log/office_src_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    #ts_softmax = np.load("log/office_ts_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    #drl_softmax = np.load("log/office_drst_softmax" + source[0] + target[0] + ".npy", allow_pickle=True)
    label = np.load("log/office_label.npy")
    iid_softmax = np.load("log/office_src_softmax.npy", allow_pickle=True)
    ts_softmax = np.load("log/office_ts_softmax.npy", allow_pickle=True)
    drl_softmax = np.load("log/office_drst_softmax.npy", allow_pickle=True)
    label = np.array(label).astype(np.int32)
    acc_iid = np.zeros(label.shape[0])
    acc_ts = np.zeros(label.shape[0])
    acc_drl = np.zeros(label.shape[0])
    for j in range(label.shape[0]):
        acc_iid[j] = iid_softmax[j][label[j]]
        acc_ts[j] = ts_softmax[j][label[j]]
        acc_drl[j] = drl_softmax[j][label[j]]
    conf_iid = np.max(iid_softmax, axis=1)
    conf_ts = np.max(ts_softmax, axis=1)
    conf_drl = np.max(drl_softmax, axis=1)
    ece_iid = ece_score(acc_iid, conf_iid, bins) / label.shape[0]
    ece_ts = ece_score(acc_ts, conf_ts, bins) / label.shape[0]
    ece_drl = ece_score(acc_drl, conf_drl, bins) / label.shape[0]
    ece_bayes = np.load("models/"+source[0]+target[0]+"_ece.npy")
    print(ece_iid, ece_ts, ece_drl, ece_bayes)

def ece_visda():
    from imagenet_train import ece_score
    bins = 10
    label = np.load("log/labels4.npy", allow_pickle=True)
    iid_softmax = np.load("log/asg_softmax2.npy", allow_pickle=True)
    ts_softmax = np.load("log/ts_softmax2.npy", allow_pickle=True)
    cbst_softmax = np.load("log/asg_cbst_softmax.npy", allow_pickle=True)
    drl_softmax = np.load("log/best_model_softmax4.npy", allow_pickle=True)
    acc_iid = np.zeros(label.shape[0])
    acc_ts = np.zeros(label.shape[0])
    acc_cbst = np.zeros(label.shape[0])
    acc_drl = np.zeros(label.shape[0])
    label = np.array(label).astype(np.int32)
    for j in range(label.shape[0]):
        acc_iid[j] = iid_softmax[j][label[j]]
        acc_ts[j] = ts_softmax[j][label[j]]
        acc_cbst[j] = cbst_softmax[j][label[j]]
        acc_drl[j] = drl_softmax[j][label[j]]
    conf_iid = np.max(iid_softmax, axis=1)
    conf_ts = np.max(ts_softmax, axis=1)
    conf_cbst = np.max(cbst_softmax, axis=1)
    conf_drl = np.max(drl_softmax, axis=1)
    ece_iid = ece_score(acc_iid, conf_iid, bins) / label.shape[0]
    ece_ts = ece_score(acc_ts, conf_ts, bins) / label.shape[0]
    ece_cbst = ece_score(acc_cbst, conf_cbst, bins) / label.shape[0]
    ece_drl = ece_score(acc_drl, conf_drl, bins) / label.shape[0]
    print(ece_iid, ece_ts, ece_cbst, ece_drl)

def brier_vada(source, target):
    """
    from office_exp import dataloader_office
    from sklearn.metrics import brier_score_loss
    _, val_loader = dataloader_office("office/" + source, "office/" + target)
    brier_vada = []
    # get the softmax
    ## load model
    for k in range(20):
        brier_score = 0
        test_num = 0
        model_vada = ResModel()
        # this might be broken... office_iid, check the acc and select the epoch
        #resume_path = "salad/examples/dirtt/log/20210828-030916_VADASolver/20210828-030916-checkpoint-ep"+str(k)+".pth"
        #resume_path = "salad/examples/dirtt/log_ad/20210828-123821_VADASolver/20210828-123821-checkpoint-ep" + str(k) + ".pth"
        #resume_path = "salad/examples/dirtt/log_wa/20210828-125710_VADASolver/20210828-125710-checkpoint-ep"+ str(k) + ".pth"
        resume_path = "salad/examples/dirtt/log_wa/20210828-213223_VADASolver/20210828-213223-checkpoint-ep"+str(k)+".pth"
        model_vada = torch.load(resume_path)
        model_vada = model_vada.to(DEVICE)
        model_vada.eval()

        vada_pred = np.zeros([1, 31])
        label_vada = np.zeros([1])
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                label = label.to(DEVICE)
                input = input.to(DEVICE)
                label = label.reshape((-1,))
                test_num += input.shape[0]
                BATCH_SIZE = input.shape[0]
                label_vada = np.concatenate([label_vada, label.cpu().numpy()], axis=0)

                _, output = model_vada(input)
                output = output.detach()
                prediction_t = F.softmax(output, dim=1)
                vada_softmax = F.softmax(output, dim=1)
                vada_softmax = vada_softmax.cpu().numpy()
                vada_pred = np.concatenate([vada_pred, vada_softmax], axis=0)
                label_onehot = torch.zeros(output.shape)
                label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
                for j in range(input.shape[0]):
                    brier_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j].cpu().numpy())
                if i % 20 == 0:
                    print("{} batches finished".format(i))
        brier_vada.append(brier_score / test_num)
    np.save("log/office_vada_brier_"+source[0]+target[0]+".npy", np.array(brier_vada))
    """
    brier_vada = np.load("log/office_vada_brier_"+source[0]+target[0]+".npy")
    x = np.arange(1, 21)
    #baseline = np.load("log/office_iid.npy", allow_pickle=True)
    baseline = np.load("log/office_" + source[0] + target[0] + ".npy", allow_pickle=True)
    baseline = baseline.item()
    brier_base = baseline["brier"]
    #brier_boost = np.load("log/drl_boost_brier_office.npy", allow_pickle=True)
    brier_boost = np.load("log/rebuttal/drl_brier" + source[0] + target[0] + ".npy", allow_pickle=True)

    fig = plt.figure(figsize=(20, 15))

    plt.plot(x, brier_base, linewidth=9, marker='v', markersize=36, color="r", label="Source")
    plt.plot(x, brier_boost, linewidth=9, marker='o', markersize=36, color="b", label="DRL")
    plt.plot(x, brier_vada, linewidth=9, marker='s', markersize=36, color="y", label="VADA")

    plt.xlabel("Epochs", fontdict={"size": 72})
    plt.ylabel("Brier score", fontdict={"size": 72})
    plt.tick_params(labelsize=42)
    plt.xlim((1, 20))
    fig.legend(loc=(0.72, 0.74), fontsize=42)
    # plt.title("Brier score over epochs", fontdict={"size":56})
    plt.savefig("log/brier_"+source[0]+target[0]+"_vada2.jpg", bbox_inches="tight")

# reusing same names as tsne_visualization, be careful of that
if __name__=="__main__":
    #plot1()
    #plot2()
    #plot3()
    #get_softmax_new()
    #reliability_plot_new()
    #eval_model()
    #eval_single_model()

    #office_drl_boost()
    #home_drl_boost()
    #visda_drl_boost()

    #eval_model()

    #home_drl_boost()

    #reliability_plot_office()
    #reliability_plot_home()

    # fig4
    #plot_with_error_bar_acc()
    #plot_with_error_bar_brier()
    #plot_shift_st()

    #drl_boost_misent()
    #plot_with_error_bar_acc_new()
    #plot_with_error_bar_brier_new()
    source = "webcam"
    target = "amazon"
    #office_drl_boost_rebuttal(source, target)
    #reliability_plot_office_rebuttal(source, target)
    #reliability_plot_office_rebuttal_with_vada(source, target)
    #office_brier_appendix(source, target)
    #office_acc_appendix(source, target)
    #ece_compare_appendix(source, target)
    #ece_visda()

    #visda_drl_boost_brier()
    #office_drl_boost_brier()
    #home_drl_boost_brier()

    brier_vada(source, target)

    """
    TS for 
    aw: [0.02109511418766111, 0.019994604150955643, 0.018372231965012004, 0.027238710437143314, 0.02406474673496682, 0.01744241071024634, 0.017853976838187534, 0.01366130051600948, 0.013197630948005192, 0.012812765175752123, 0.012560053186123087, 0.01279564738173117, 0.012577705853697116, 0.014017913086561721, 0.013414748423770767, 0.013168746743931767, 0.01340269414345967, 0.012991495533067655, 0.012824932126673853, 0.012919619304188248]
    PA: [0.01418108771319445, 0.013393530715272477, 0.012966472156910935, 0.0129248142551241, 0.012492917641482035, 0.012287964070434252, 0.01310164755615466, 0.01201626461219822, 0.0118817719317709, 0.012422845308739128, 0.01249721218215017, 0.015147923207778032, 0.013558797594011001, 0.013583340569022617, 0.01344637033099914, 0.013771064529614678, 0.015147895298085624, 0.015147853994920878, 0.015147928278347095, 0.015147928993177559]
    VisDA: 
    """
