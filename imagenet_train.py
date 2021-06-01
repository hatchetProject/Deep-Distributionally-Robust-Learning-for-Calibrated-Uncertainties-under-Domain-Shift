import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from model_layers_imgnet import ClassifierLayerAVH, GradLayer

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-c', '--num-classes', default=1000, type=int,
                    metavar='C', help='number of classes (default: 1000)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--tgt-dir', '--tgt-dir', default=None, help='Path to target dataset')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    """
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    """
    testdir = args.tgt_dir
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        """
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        """

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_both(optimizer_alpha, optimizer_beta, epoch):
    """Adopt the same setting as original imagenet training process"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer_alpha.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_beta.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class alpha_vis(nn.Module):
    def __init__(self, n_output):
        super(alpha_vis, self).__init__()
        if args.pretrained:
            self.model = models.__dict__[args.arch](pretrained=True)
            print("Using pretrained model")
        else:
            self.model = models.__dict__[args.arch](pretrained=False)
        if args.arch == "alexnet" or args.arch == "vgg19":
            num_ftrs = self.model.classifier[6].in_features
        elif args.arch == "resnet50":
            num_ftrs = self.model.fc.in_features
        elif args.arch == "densenet121":
            num_ftrs = self.model.classifier.in_features
        else:
            raise Exception("Model arch wrongly specified")
        if args.pretrained:
            self.final_layer = ClassifierLayerAVH(2048, n_output, bias=True)
            if args.arch == "alexnet" or args.arch == "vgg19":
                self.final_layer.weight = self.model.classifier[6].weight
                self.final_layer.bias = self.model.classifier[6].bias
            elif args.arch == "resnet50":
                self.final_layer.weight = self.model.fc.weight
                self.final_layer.bias = self.model.fc.bias
            elif args.arch == "densenet121":
                self.final_layer.weight = self.model.classifier.weight
                self.final_layer.bias = self.model.classifier.bias
            else:
                raise Exception("Model arch wrongly specified")
        else:
            self.final_layer = ClassifierLayerAVH(num_ftrs, n_output, bias=True)
        extractor = torch.nn.Sequential(
        )
        if args.arch == "alexnet" or args.arch == "vgg19":
            self.model.classifier[6] = extractor
        elif args.arch == "resnet50":
            self.model.fc = extractor
        elif args.arch == "densenet121":
            self.model.classifier = extractor
        else:
            raise Exception("Model arch wrongly specified")

    def forward(self, x_s, y_s, r):
        x1 = self.model(x_s)
        x = self.final_layer(x1, y_s, r)
        return x

class alpha_vis_bottom(nn.Module):
    def __init__(self, n_output):
        super(alpha_vis_bottom, self).__init__()
        if args.pretrained:
            self.model = models.__dict__[args.arch](pretrained=True)
            print("Using pretrained model")
        else:
            self.model = models.__dict__[args.arch](pretrained=False)
        if args.arch == "alexnet" or args.arch == "vgg19":
            num_ftrs = self.model.classifier[6].in_features
        elif args.arch == "resnet50":
            num_ftrs = self.model.fc.in_features
        elif args.arch == "densenet121":
            num_ftrs = self.model.classifier.in_features
        else:
            raise Exception("Model arch wrongly specified")
        if args.pretrained:
            self.final_layer = ClassifierLayerAVH(2048, n_output, bias=True)
            if args.arch == "alexnet" or args.arch == "vgg19":
                self.final_layer.weight = self.model.classifier[6].weight
                self.final_layer.bias = self.model.classifier[6].bias
            elif args.arch == "resnet50":
                self.final_layer.weight = self.model.fc.weight
                self.final_layer.bias = self.model.fc.bias
            elif args.arch == "densenet121":
                self.final_layer.weight = self.model.classifier.weight
                self.final_layer.bias = self.model.classifier.bias
            else:
                raise Exception("Model arch wrongly specified")
        else:
            self.final_layer = ClassifierLayerAVH(num_ftrs, n_output, bias=True)
        extractor = torch.nn.Sequential(
        )
        if args.arch == "alexnet" or args.arch == "vgg19":
            self.model.classifier[6] = extractor
        elif args.arch == "resnet50":
            self.model.fc = extractor
        elif args.arch == "densenet121":
            self.model.classifier = extractor
        else:
            raise Exception("Model arch wrongly specified")

    def forward(self, x_s, y_s, r):
        x1 = self.model(x_s)
        x = self.final_layer(x1, y_s, r)
        return x1, x

class beta_vis(nn.Module):
    def __init__(self):
        super(beta_vis, self).__init__()
        if args.pretrained:
            self.model = models.__dict__[args.arch](pretrained=True)
        else:
            self.model = models.__dict__[args.arch](pretrained=False)
        if args.arch == "alexnet" or args.arch == "vgg19":
            num_ftrs = self.model.classifier[6].in_features
        elif args.arch == "resnet50":
            num_ftrs = self.model.fc.in_features
        elif args.arch == "densenet121":
            num_ftrs = self.model.classifier.in_features
        else:
            raise Exception("Model arch wrongly specified")
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2)
        )
        if args.arch == "alexnet" or args.arch == "vgg19":
            self.model.classifier[6] = extractor
        elif args.arch == "resnet50":
            self.model.fc = extractor
        elif args.arch == "densenet121":
            self.model.classifier = extractor
        else:
            raise Exception("Model arch wrongly specified")
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p

class beta_vis_test(nn.Module):
    def __init__(self, arch):
        super(beta_vis_test, self).__init__()
        if args.pretrained:
            self.model = models.__dict__[arch](pretrained=True)
        else:
            self.model = models.__dict__[arch](pretrained=False)
        if arch == "alexnet" or arch == "vgg19":
            num_ftrs = self.model.classifier[6].in_features
        elif arch == "resnet50":
            num_ftrs = self.model.fc.in_features
        elif arch == "densenet121":
            num_ftrs = self.model.classifier.in_features
        else:
            raise Exception("Wrong model type specified")
        extractor = torch.nn.Sequential(
            nn.Linear(num_ftrs, 2048),
            nn.Tanh(),
            nn.Linear(2048, 2)
        )
        if arch == "alexnet" or arch == "vgg19":
            self.model.classifier[6] = extractor
        elif arch == "resnet50":
            self.model.fc = extractor
        elif arch == "densenet121":
            self.model.classifier = extractor
        else:
            raise Exception("Wrong model type specified")
        self.grad = GradLayer()

    def forward(self, x, nn_output, prediction, p_t, pass_sign):
        p = self.model(x)
        p = self.grad(p, nn_output, prediction, p_t, pass_sign)
        return p

class ImageClassdata(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_list, label_list, root_dir, transform=transforms.ToTensor()):
        self.image_list = img_list
        self.label = label_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_name)
        image = img.convert('RGB')
        lbl = self.label[idx]
        label = torch.from_numpy(np.array([lbl]))

        if self.transform:
            image = self.transform(image)

        return image, label

class ImageClassdataFreq(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, img_list, label_list, root_dir, freq_list, transform=transforms.ToTensor()):
        self.image_list = img_list
        self.label = label_list
        self.root_dir = root_dir
        self.freq = freq_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        img = Image.open(img_name)
        image = img.convert('RGB')
        lbl = self.label[idx]
        label = torch.from_numpy(np.array([lbl]))
        frequency = self.freq[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, frequency

def imgnet_v2_dataset(transform):
    global args
    args = parser.parse_args()
    filepath = "CRST/imagenet_v2/imagenetv2-matched-frequency-format-val/"
    file_list = []
    label_list = []
    for path in os.listdir(filepath):
        name = os.path.join(filepath, path)
        for p in os.listdir(name):
            list_tmp = os.path.join(path, p)
            file_list.append(list_tmp)
            label_list.append(int(path))
    file_list = np.array(file_list)
    label_list = np.array(label_list)
    return ImageClassdata(img_list=file_list, label_list=label_list, root_dir=filepath,
                          transform=transform)

def divide_freq(bottom, top, transform):
    global args
    args = parser.parse_args()
    frequency_list = np.load('frequency.npy')
    name_list = np.load('filenames.npy')
    filepath = "CRST/imagenet_v2/imagenetv2-matched-frequency-format-val/"
    file_list = [0] * frequency_list.shape[0]
    label_list = [0] * frequency_list.shape[0]
    for path in os.listdir(filepath):
        name = os.path.join(filepath, path)
        for p in os.listdir(name):
            name_idx = np.where(name_list == p[:-5])[0][0]
            list_tmp = os.path.join(path, p)
            file_list[name_idx] = list_tmp
            label_list[name_idx] = int(path)
    truth_i = (frequency_list >= bottom) & (frequency_list <= top)
    index_i = np.where(truth_i == True)[0]
    file_list = np.array(file_list)
    label_list = np.array(label_list)
    return ImageClassdata(img_list=file_list[index_i], label_list=label_list[index_i], root_dir=filepath, transform=transform)

def divide_freq_half(transform):
    """
    For returning the validation loader for Temperature scaling, which is half of the validation set with labels
    :param transform: transformation type
    :return: validation set, test set under with different frequencies
    """
    global args
    args = parser.parse_args()
    frequency_list = np.load('frequency.npy')
    name_list = np.load('filenames.npy')
    filepath = "CRST/imagenet_v2/imagenetv2-matched-frequency-format-val/"
    frequency_valid = []
    file_list_valid = []
    label_list_valid = []
    frequency_test = []
    file_list_test = []
    label_list_test = []
    for path in os.listdir(filepath):
        name = os.path.join(filepath, path)
        for i, p in enumerate(os.listdir(name)):
            name_idx = np.where(name_list == p[:-5])[0][0]
            if i < 5:
                list_tmp = os.path.join(path, p)
                file_list_valid.append(list_tmp)
                label_list_valid.append(int(path))
                frequency_valid.append(frequency_list[name_idx])
            else:
                list_tmp = os.path.join(path, p)
                file_list_test.append(list_tmp)
                label_list_test.append(int(path))
                frequency_test.append(frequency_list[name_idx])
    file_list_valid = np.array(file_list_valid)
    label_list_valid = np.array(label_list_valid)
    file_list_test = np.array(file_list_test)
    label_list_test = np.array(label_list_test)
    frequency_valid = np.array(frequency_valid)
    frequency_test = np.array(frequency_test)
    valid_loader = ImageClassdata(img_list=file_list_valid, label_list=label_list_valid, root_dir=filepath,
                                  transform=transform)
    test_loader = ImageClassdata(img_list=file_list_test, label_list=label_list_test, root_dir=filepath,
                                  transform=transform)
    freq = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    valid_set = []
    test_set = []
    for i in range(5):
        bottom = freq[i]
        top = freq[i+1]
        truth_i = (frequency_valid >= bottom) & (frequency_valid <= top)
        index_i = np.where(truth_i == True)[0]
        valid_set.append(ImageClassdata(img_list=file_list_valid[index_i], label_list=label_list_valid[index_i],
                                        root_dir=filepath, transform=transform))
    for i in range(5):
        bottom = freq[i]
        top = freq[i+1]
        truth_i = (frequency_test >= bottom) & (frequency_test <= top)
        index_i = np.where(truth_i == True)[0]
        test_set.append(ImageClassdata(img_list=file_list_test[index_i], label_list=label_list_test[index_i],
                                        root_dir=filepath, transform=transform))
    return valid_loader, test_loader, valid_set, test_set

def imagenet_drl():
    global args, best_prec1
    args = parser.parse_args()

    """
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model_alpha = models.__dict__[args.arch](pretrained=True)
        model_beta = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model_alpha = models.__dict__[args.arch]()
        model_beta = models.__dict__[args.arch]()
    """

    model_alpha = alpha_vis(args.num_classes)
    model_beta = beta_vis()
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model_alpha.features = torch.nn.DataParallel(model_alpha.model.features)
        model_alpha.cuda()
        model_beta.features = torch.nn.DataParallel(model_beta.model.features)
        model_beta.cuda()
    else:
        model_alpha = torch.nn.DataParallel(model_alpha).cuda()
        model_beta = torch.nn.DataParallel(model_beta).cuda()

    # Do self-training with new parametric form, use different label selection criterions
    optimizer_alpha = torch.optim.SGD(model_alpha.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_beta = torch.optim.SGD(model_beta.parameters(), args.lr * 0.1,#args.lr * 0.001,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    prec1_best, best_epoch = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_alpha.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_set = imgnet_v2_dataset(transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    """
    testdir = args.tgt_dir
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    """
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate_both(optimizer_alpha, optimizer_beta, epoch)

        train_drl(train_loader, val_loader, model_alpha, model_beta, optimizer_alpha, optimizer_beta, epoch)

        prec1, prec5, losses = validate_drl(val_loader, model_alpha, model_beta)

        directory = "imagenet_runs_new_new2/"+str(args.arch) + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if (prec1 > prec1_best):
            prec1_best = prec1
            best_epoch = epoch

        torch.save(model_alpha.state_dict(), directory + "alpha_epoch_" + str(epoch) + ".pth.tar")
        torch.save(model_beta.state_dict(), directory + "beta_epoch_" + str(epoch) + ".pth.tar")
        print("")
        print("Best precision achieved at epoch: ", best_epoch)
        print(
            "Current top 1 precision: {:.3f}, top 5 precision {:3f}, loss: {:.3f}, best precision achieved: {:.3f}".format(
                prec1, prec5, losses, prec1_best))

def validate_drl(test_loader, model_alpha, model_beta):
    # validate model and select samples for self-training
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    model_alpha = model_alpha.cuda()
    model_beta = model_beta.cuda()
    model_alpha.eval()
    model_beta.eval()
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            input = input.cuda()
            label = label.cuda(async=True)
            label = label.reshape((-1, )).long()
            BATCH_SIZE = input.shape[0]

            # Add flipping
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).cuda()
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)

            target_out = model_alpha(input, torch.ones((BATCH_SIZE, args.num_classes)).cuda(), r_target.cuda()).detach()
            #target_out = model_alpha(input, torch.ones((BATCH_SIZE, args.num_classes)).cuda(), torch.ones(r_target.shape).cuda()).detach()
            #prediction_t = F.softmax(target_out, dim=1)
            test_loss = float(ce_loss(target_out, label))
            prec1, prec5 = accuracy(target_out.data, label, topk=(1, 5))
            losses.update(test_loss, input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader), loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def train_drl(train_loader, test_loader, model_alpha, model_beta, optimizer_alpha, optimizer_beta, epoch):
    ## train loader sample number must be smaller than test loader
    model_alpha = model_alpha.cuda()
    model_beta = model_beta.cuda()
    model_alpha.train()
    model_beta.train()
    iter_train = iter(train_loader)
    iter_test = iter(test_loader)
    bce_loss = nn.BCEWithLogitsLoss()
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    ce_loss = nn.CrossEntropyLoss()
    train_loss, train_acc = 0, 0
    #for i, (input, label) in enumerate(train_loader):
    for i in range(len(train_loader)):
        try:
            input, label = iter_train.next()
        except:
            pass
        try:
            input_test, _ = iter_test.next()
        except:
            iter_test = iter(test_loader)
            input_test, _ = iter_test.next()
        input = input.cuda()
        label_train = label.cuda(async=True)
        input_test = input_test.cuda()

        BATCH_SIZE = input.shape[0]
        input_concat = torch.cat([input, input_test], dim=0)
        # this parameter used for softlabling
        label_concat = torch.cat(
            (torch.FloatTensor([1, 0]).repeat(input.shape[0], 1), torch.FloatTensor([0, 1]).repeat(input_test.shape[0], 1)), dim=0)
        label_concat = label_concat.cuda()

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
        p_t_target = p_t[BATCH_SIZE:]
        label_train_onehot = torch.zeros([BATCH_SIZE, args.num_classes])
        for j in range(BATCH_SIZE):
            label_train_onehot[j][label_train[j].long()] = 1

        theta_out = model_alpha(input, label_train_onehot.cuda(), r_source.detach().cuda())
        source_pred = F.softmax(theta_out, dim=1)
        nn_out = model_alpha(input_test, torch.ones((input_test.shape[0], args.num_classes)).cuda(), r_target.detach().cuda())

        pred_target = F.softmax(nn_out, dim=1)
        prob_grad_r = model_beta(input_test, nn_out.detach(), pred_target.detach(), p_t_target.detach(),
                                    sign_variable)
        loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape).cuda()))
        loss_theta = torch.sum(theta_out)

        # Backpropagate
        if i < 400 and epoch == 0:
            optimizer_beta.zero_grad()
            loss_dis.backward(retain_graph=True)
            optimizer_beta.step()

        if i < 320 and epoch == 0:
            optimizer_beta.zero_grad()
            loss_r.backward(retain_graph=True)
            optimizer_beta.step()

        if (i + 1) % 1 == 0:
            optimizer_alpha.zero_grad()
            loss_theta.backward()
            optimizer_alpha.step()

        train_loss += float(ce_loss(theta_out.detach(), label_train.long()))
        train_acc += torch.sum(torch.argmax(source_pred.detach(), dim=1) == label_train.long()).float() / BATCH_SIZE
        if i % args.print_freq == 0:
            train_loss = train_loss/(args.print_freq*BATCH_SIZE)
            train_acc = train_acc/(args.print_freq)
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss: {3:.4f} \t Train Acc: {4:.4f}'.format(
                epoch, i, len(train_loader), train_loss, train_acc*100.0))
            train_loss, train_acc = 0, 0
    print("A sample of density ratio: ", r[0])

def test_density_ratio():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    batch_size = 64
    workers = 4
    dataset1 = divide_freq(0, 0.2, transform)
    val_loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset2 = divide_freq(0.2, 0.4, transform)
    val_loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset3 = divide_freq(0.4, 0.6, transform)
    val_loader3 = torch.utils.data.DataLoader(
        dataset3, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset4 = divide_freq(0.6, 0.8, transform)
    val_loader4 = torch.utils.data.DataLoader(
        dataset4, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset5 = divide_freq(0.8, 1.0, transform)
    val_loader5 = torch.utils.data.DataLoader(
        dataset5, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader_list = [val_loader1, val_loader2, val_loader3, val_loader4, val_loader5]
    model_file_path = "imagenet_runs/alexnet/"
    model_beta = beta_vis()
    epoch_num = 7
    metric_mat = np.zeros((5, epoch_num))
    for p in range(epoch_num):
        print("Epoch {} starting".format(p))
        state = torch.load(os.path.join(model_file_path, "beta_epoch_"+str(p)+".pth.tar"))
        #print(state.keys())
        #print("---------------------")
        #print(model_beta.state_dict().keys())
        model_beta.load_state_dict(state, strict=False)
        model_beta.cuda()
        model_beta.eval()
        for k in range(5):
            val_loader = val_loader_list[k]
            for i, (img, label) in enumerate(val_loader):
                input = img.cuda()
                pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).cuda()
                r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                metric_mat[k][p] += torch.sum(r_target).detach().cpu().numpy()
    print(metric_mat)
    return metric_mat

def plot_hsf():
    import matplotlib.pyplot as plt
    metric_mat = test_density_ratio()
    color_list = ['b', 'o', 'g', 'r', 'p']
    num = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    epochs = metric_mat.shape[1]
    for i in np.arange(5):
        plt.plot(np.arange(epochs), metric_mat[i], c=color_list[i], label="Frequency=["+str(num[i])+", "+str(num[i+1])+"]")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Density Ratio")
    plt.savefig("")

def ts_model():
    """
    Do Temperature scaling on different models
    :return: The saved models
    """
    from temperature_scaling import ModelWithTemperature, _ECELoss
    from sklearn.metrics import brier_score_loss
    batch_size = 64
    workers = 4
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    valid_data, _, _,  test_set = divide_freq_half(transform)
    val_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    model = models.__dict__[args.arch](pretrained=True)
    model = model.cuda()
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(val_loader)
    scaled_model.eval()
    model.eval()
    model_alpha = alpha_vis(args.num_classes)
    model_beta = beta_vis()
    checkpoint_alpha = "imagenet_runs/" + args.arch + "/alpha_epoch_3.pth.tar"
    checkpoint_beta = "imagenet_runs/" + args.arch + "/beta_epoch_3.pth.tar"
    model_alpha.load_state_dict(torch.load(checkpoint_alpha), strict=False)
    model_beta.load_state_dict(torch.load(checkpoint_beta), strict=False)
    model_alpha.cuda()
    model_beta.cuda()
    model_alpha.eval()
    model_beta.eval()

    print("=> using pre-trained model '{}'".format(args.arch))
    brier_list_src = []
    ece_list_src = []
    brier_list_ts = []
    ece_list_ts = []
    brier_list_drl = []
    ece_list_drl = []
    bins = 1
    for i in range(len(test_set)):
        test_data = test_set[i]
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        pred_src = np.zeros((1, 1000))
        pred_ts = np.zeros((1, 1000))
        pred_drl = np.zeros((1, 1000))
        label_list = np.zeros([1])
        test_num = 0
        b_score = 0
        b_score_src = 0
        b_score_drl = 0
        ece = 0
        ece_src = 0
        ece_drl = 0

        with torch.no_grad():
            for data, label in test_loader:
                label = label.cpu().long()
                test_num += data.shape[0]
                label_list = np.concatenate([label_list, label.cpu().numpy().reshape(-1, )], axis=0)
                data = data.cuda()
                target_out = scaled_model(data)
                prediction_t = F.softmax(target_out, dim=1).detach().cpu().numpy()
                pred_ts = np.concatenate([pred_ts, prediction_t], axis=0)

                target_out_src = model(data)
                prediction_t_src = F.softmax(target_out_src, dim=1).detach().cpu().numpy()
                pred_src = np.concatenate([pred_src, prediction_t_src], axis=0)
                label_onehot = torch.zeros(prediction_t.shape)
                label_onehot.scatter_(1, label.cpu().long().reshape(-1, 1), 1)
                acc = np.zeros(label.shape[0])
                acc_src = np.zeros(label.shape[0])
                for j in range(data.shape[0]):
                    #b_score += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j])
                    #b_score_src += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t_src[j])
                    acc[j] = prediction_t[j][label[j]]
                    acc_src = prediction_t_src[j][label[j]]
                b_score += brier_score(prediction_t, label.reshape(-1, ))
                b_score_src += brier_score(prediction_t_src, label.reshape(-1, ))
                conf = np.max(prediction_t, axis=1)
                ece += ece_score(acc, conf, bins)
                conf_src = np.max(prediction_t_src, axis=1)
                ece_src += ece_score(acc_src, conf_src, bins)

                pred = F.softmax(model_beta(data, None, None, None, None).detach(), dim=1)
                r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                target_out = model_alpha(data, torch.ones((data.shape[0], args.num_classes)), r_target).detach()
                prediction_t = F.softmax(target_out, dim=1).detach().cpu().numpy()
                pred_drl = np.concatenate([pred_drl, prediction_t], axis=0)
                acc = np.ones(label.shape[0])
                for j in range(data.shape[0]):
                    #b_score_drl += brier_score_loss(label_onehot[j].cpu().numpy(), prediction_t[j])
                    acc[j] = prediction_t[j][label[j]]
                b_score_drl += brier_score(prediction_t, label.reshape(-1, ))
                conf = np.max(prediction_t, axis=1)
                ece_drl += ece_score(acc, conf, bins)

        ece /= test_num
        ece_src /= test_num
        ece_drl /= test_num
        b_score = np.sqrt(b_score / test_num)
        b_score_src = np.sqrt(b_score_src / test_num)
        b_score_drl = np.sqrt(b_score_drl / test_num)
        print("Source: Val loader {} has brier score {}, ECE {}".format(i, b_score_src, ece_src))
        print("TS: Val loader {} has brier score {}, ECE {}".format(i, b_score, ece))
        print("DRL: Val loader {} has brier score {}, ECE {}".format(i, b_score_drl, ece_drl))
        brier_list_ts.append(b_score)
        ece_list_ts.append(ece)
        brier_list_src.append(b_score_src)
        ece_list_src.append(ece_src)
        brier_list_drl.append(b_score_drl)
        ece_list_drl.append(ece_drl)

        pred_src = pred_src[1:]
        pred_ts = pred_ts[1:]
        pred_drl = pred_drl[1:]
        label_list = label_list[1:]

        directory = "imagenet_runs/" + args.arch +"/" + str(i) + "/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        np.save(directory + "src.npy", pred_src)
        np.save(directory + "ts.npy", pred_ts)
        np.save(directory + "drl.npy", pred_drl)
        np.save(directory + "label.npy", label_list)

    print(brier_list_src)
    print(brier_list_ts)
    print(brier_list_drl)
    print(ece_list_src)
    print(ece_list_ts)
    print(ece_list_drl)

def reliability_plot():
    """
    Plot the realiability plot for different frequencies
    :return: the saved plots
    """
    global args
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    for j in range(5):
        directory = "imagenet_runs/" + args.arch + "/" + str(j) + "/"
        src_pred = np.load(directory + "src.npy")
        ts_pred = np.load(directory + "ts.npy")
        drl_pred = np.load(directory + "drl.npy")
        label = np.load(directory + "label.npy")
        label = label.astype(np.int)

        #intervals = [0, 0.85, 0.87, 0.9, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]
        #intervals = [0, 0.7, 0.8, 0.83, 0.87, 0.91, 0.93, 0.95, 0.97, 0.98, 1.0]
        intervals = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.83, 1.0]
        #intervals = [x/10 for x in intervals]
        #intervals[-1] = 1
        plt.figure(figsize=(15, 10))
        plt.plot([0, 1], np.zeros(2), c='k', linestyle='--', linewidth=5)
        # IID
        iid_conf = np.max(src_pred, axis=1)
        iid_acc = (np.argmax(ts_pred, axis=1) == label)

        x_value = []
        y_value = []
        for i in range(len(intervals) - 1):
            int_idx = ((iid_conf >= intervals[i]) == (iid_conf < intervals[i + 1]))  # 0 and 1
            x_value.append(np.sum(iid_conf[int_idx]) / np.sum(int_idx))
            y_value.append((np.sum(iid_conf[int_idx]) - np.sum(iid_acc[int_idx])) / np.sum(int_idx))
        #print(y_value)
        plt.plot(np.arange(0, 100, 10) / 100., y_value, label="Source", marker="^", color="#e50000", linewidth=5,
                 markersize=12, linestyle="-.")

        # TS
        ts_conf = np.max(ts_pred, axis=1)
        # print("TS conf:", ts_conf)
        ts_acc = (np.argmax(ts_pred, axis=1) == label)
        x_value = []
        y_value = []
        for i in range(len(intervals) - 1):
            int_idx = ((ts_conf >= intervals[i]) == (ts_conf < intervals[i + 1]))  # 0 and 1
            x_value.append(np.sum(ts_conf[int_idx]) / np.sum(int_idx))
            y_value.append((np.sum(ts_conf[int_idx]) - np.sum(ts_acc[int_idx])) / np.sum(int_idx))
        #print(y_value)
        plt.plot(np.arange(0, 100, 10) / 100., y_value, label="TS", marker="o", color="#96f97b", linewidth=5,
                 markersize=12, linestyle=":")

        # DRST
        drst_conf = np.max(drl_pred, axis=1)
        #print(drst_conf)
        drst_acc = (np.argmax(drl_pred, axis=1) == label)
        x_value = []
        y_value = []
        for i in range(len(intervals) - 1):
            int_idx = ((drst_conf >= intervals[i]) == (drst_conf < intervals[i + 1]))  # 0 and 1
            #print(int_idx)
            x_value.append(np.sum(drst_conf[int_idx]) / np.sum(int_idx))
            y_value.append((np.sum(drst_conf[int_idx]) - np.sum(drst_acc[int_idx])) / np.sum(int_idx))
        #print(y_value)
        plt.plot(np.arange(0, 100, 10) / 100., y_value, label="DRL", marker="*", color="blue", linewidth=5,
                 markersize=12)

        x_value = [str(x)[:5] for x in x_value]
        x_axis = np.concatenate((x_value[0::2], np.array([x_value[-1]])))
        plt.xticks(np.array([0, 20, 40, 60, 80, 95]) / 100., x_axis, rotation=0)
        plt.tick_params(labelsize=28)
        plt.xlabel("Confidence (max prob)", fontdict={"weight": "normal", "size": 42})
        plt.ylabel("Confidence - Accuracy", fontdict={"weight": "normal", "size": 42})
        plt.legend(fontsize=36, loc=4)
        plt.title("Reliability plot on Imagenet-v2",
                  fontdict={"weight": "normal", "size": 42})
        plt.savefig(directory + "relia_plot.jpg")

def ece_score(acc, conf, bin):
    """
    Calculate the ece score
    :param acc: accuracy numpy array list
    :param conf: confidence numpy array list
    :param bin: integer number of bins chosen
    :return: the sum of ece score without dividing the total
    """
    bins = np.arange(0., bin+1) / bin
    passed = 0
    ece = 0
    for i in range(bin):
        #acc_idx_bot = acc >= bins[i]
        #acc_idx_top = acc < bins[i+1]
        conf_idx_bot = conf > bins[i]
        conf_idx_top = conf <= bins[i+1]
        #acc_idx = np.where((acc_idx_bot & acc_idx_top) == 1)[0]
        conf_idx = np.where((conf_idx_bot & conf_idx_top) == 1)[0]
        #acc_idx = conf_idx
        if acc.shape == ():
            acc_i = np.array([acc])
        else:
            acc_i = acc[conf_idx]
        conf_i = conf[conf_idx]
        if acc_i.shape[0] == 0 and conf_i.shape[0] > 0:
            ece = ece + conf_idx.shape[0] * np.abs(np.mean(conf_i))
            #pass
        elif acc_i.shape[0] > 0 and conf_i.shape[0] == 0:
            ece = ece + conf_idx.shape[0] * np.abs(np.mean(acc_i))
            #pass
        elif acc_i.shape[0] > 0 and conf_i.shape[0] > 0:
            ece = ece + conf_idx.shape[0] * np.abs(np.mean(acc_i) - np.mean(conf_i))
        else:
            pass
    #ece = ece / acc.shape[0]
    return ece

def brier_score(pred, label):
    """
    The brier score calculation
    :param pred: prediction list
    :param label: label list
    :return: sum of brier score
    """
    conf = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        conf[i] = pred[i][label[i]]
    b_score = np.sum(np.square(1 - conf))
    return b_score

def bar_plot_freq():
    """
    Plot the model-density_ratio plot under different frequencies
    :return: The saved plot
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size = 64
    workers = 4
    dataset1 = divide_freq(0, 0.2, transform)
    val_loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset2 = divide_freq(0.2, 0.4, transform)
    val_loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset3 = divide_freq(0.4, 0.6, transform)
    val_loader3 = torch.utils.data.DataLoader(
        dataset3, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset4 = divide_freq(0.6, 0.8, transform)
    val_loader4 = torch.utils.data.DataLoader(
        dataset4, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset5 = divide_freq(0.8, 1.0, transform)
    val_loader5 = torch.utils.data.DataLoader(
        dataset5, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader_list = [val_loader1, val_loader2, val_loader3, val_loader4, val_loader5]
    model_names = ['alexnet', 'vgg19', 'resnet50', 'densenet121']
    directory = "imagenet_runs/"
    metric_mat = np.zeros((4, 5))
    for i in range(4):
        if i != 1:
            pass
        else:
            print("Inferring model {}".format(i+1))
            model_beta = beta_vis_test(arch=model_names[i])
            model_beta.cuda()
            state_path = "imagenet_runs_new_new2/" + model_names[i] + "/beta_epoch_0.pth.tar"
            state = torch.load(state_path)
            model_beta.load_state_dict(state, strict=False)
            model_beta.eval()
            for k in range(5):
                val_loader = val_loader_list[k]
                for j, (img, label) in enumerate(val_loader):
                    input = img.cuda()
                    pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).cuda()
                    r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1).detach().cpu().numpy()
                    r_target[r_target > 500] = 500
                    metric_mat[i][k] += np.sum(r_target)
                metric_mat[i][k] /= len(val_loader.dataset)
    print(metric_mat)
    """
    metric_mat = np.array([[2.58291555, 3.32628701, 3.3149397,  3.19687839, 4.08868733],
                          [1.13778437, 2.20619964, 1.76003343, 1.68781681, 1.55706423],
                          [2.06015784, 2.06383605, 2.07206284, 2.0737225,  2.07184063],
                          [0.68593349, 0.69250847, 0.69196763, 0.69217597, 0.69309759]])
    # [2.58291555, 3.32628701, 3.3149397,  3.19687839, 4.08868733]
    # [1.13778437, 2.20619964, 1.76003343, 1.68781681, 1.55706423] / [2.13162191 4.57577255 4.42264096 3.91283836 3.57433475]
     / [0.63808699 0.68513046 0.65969879 0.6510038  0.65225395] / [1.50414179 2.13127441 2.14219937 1.91853542 1.76633564]
     / [1.33363382 2.26399752 1.73866417 1.8613818  1.85290257] / [1.49322307 2.01572422 2.22646512 1.96153746 1.92842411]
     / 
    # [2.06015784, 2.06383605, 2.07206284, 2.0737225, 2.07184063]
    # [0.68593349, 0.69250847, 0.69196763, 0.69217597, 0.69309759]

    # Plot graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    x = np.arange(4)
    barwidth = 0.15
    labels = ["Freq=[0.0, 0.2]", "Freq=[0.2, 0.4]", "Freq=[0.4, 0.6]", "Freq=[0.6, 0.8]", "Freq=[0.8, 1.0]"]
    model_str = ["AlexNet", "VGG-19", "ResNet-50", "DenseNet-121"]
    for i in range(5):
        if i == 2:
            plt.bar(x+barwidth*i, metric_mat[:, i], barwidth, align="center", label=labels[i], tick_label=model_str)
        else:
            plt.bar(x + barwidth * i, metric_mat[:, i], barwidth, align="center", label=labels[i])
    plt.legend()
    plt.tick_params(labelsize=28)
    plt.xlabel("Model Structures", fontdict={"weight": "normal", "size": 42})
    plt.ylabel("Density Ratio", fontdict={"weight": "normal", "size": 42})
    plt.legend(fontsize=32)
    plt.title("Density Ratio vs HSF",
              fontdict={"weight": "normal", "size": 42})
    plt.savefig("imagenet_runs/bar_plot.jpg")
    """

def bar_plot_freq_binary():
    """
    Plot the model-density_ratio plot under different frequencies
    :return: The saved plot
    """
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size = 64
    workers = 4
    dataset1 = divide_freq(0, 0.2, transform)
    val_loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset2 = divide_freq(0.2, 1.0, transform)
    val_loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader_list = [val_loader1, val_loader2]
    model_names = ['alexnet', 'vgg19', 'resnet50', 'densenet121']
    directory = "imagenet_runs/"
    metric_mat = np.zeros((4, 2))
    for i in range(4):
        print("Inferring model {}".format(i+1))
        model_beta = beta_vis_test(arch=model_names[i])
        model_beta.cuda()
        state_path = "imagenet_runs_new_new/" + model_names[i] + "/beta_epoch_0.pth.tar"
        state = torch.load(state_path)
        model_beta.load_state_dict(state, strict=False)
        model_beta.eval()
        for k in range(2):
            val_loader = val_loader_list[k]
            for j, (img, label) in enumerate(val_loader):
                input = img.cuda()
                pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).cuda()
                r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1).detach().cpu().numpy()
                #r_target[r_target > 100] = 100
                metric_mat[i][k] += np.sum(r_target)
            metric_mat[i][k] /= len(val_loader.dataset)
    print(metric_mat)
    """
    metric_mat = np.array([[2.32302593, 2.76441319],
                          [1.19713678, 3.1773393]])
                          #[1.79744598, 1.80362982],
                          #[0.65727895, 0.66488595]])
    """
    0.2 as separation
    [[2.32302593, 2.76441319]
    [1.19713678, 3.1773393 ]
    [1.79744598, 1.80362982]
    [0.65727895, 0.66488595]]
    
    [[2.32302612 3.58060116]
    [1.19713618 6.80610453]
    [0.97459426 0.98169553]
    [0.599146   0.6063117 ]]
    """


    # Plot graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    x = np.array([0, 0.33])
    barwidth = 0.1
    labels = ["Low HSF", "High HSF"]
    model_str = ["AlexNet", "VGG-19"]#, "ResNet-50", "DenseNet-121"]
    for i in range(2):
        if i == 1:
            plt.bar(x + barwidth * i, metric_mat[:, i], barwidth, align="edge", label=labels[i], tick_label=model_str)
        else:
            plt.bar(x - barwidth * i, metric_mat[:, i], barwidth, align="edge", label=labels[i])
    #plt.xticks(x, model_str, rotation=0)
    plt.tick_params(labelsize=56)
    #plt.xlabel("Model Structures", fontdict={"weight": "normal", "size": 42})
    plt.ylabel("Density Ratio", fontdict={"weight": "normal", "size": 56})
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2, fontsize=48)
    #plt.title("Density Ratio vs HSF",
    #          fontdict={"weight": "normal", "size": 42})
    plt.savefig("imagenet_runs/bar_plot_binary.jpg")

def plot_freq_prob_color():
    import matplotlib.pyplot as plt
    global args
    args = parser.parse_args()

    frequency_list = np.load('frequency.npy')
    name_list = np.load('filenames.npy')
    filepath = "CRST/imagenet_v2/imagenetv2-matched-frequency-format-val/"
    file_list = [0] * frequency_list.shape[0]
    label_list = [0] * frequency_list.shape[0]
    for path in os.listdir(filepath):
        name = os.path.join(filepath, path)
        for p in os.listdir(name):
            name_idx = np.where(name_list == p[:-5])[0][0]
            list_tmp = os.path.join(path, p)
            file_list[name_idx] = list_tmp
            label_list[name_idx] = int(path)
    file_list = np.array(file_list)
    label_list = np.array(label_list)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size = 64
    workers = 4
    dataset = ImageClassdataFreq(img_list=file_list, label_list=label_list, freq_list=frequency_list,
                                 root_dir=filepath, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    dataset_ts = ImageClassdata(img_list=file_list, label_list=label_list,
                                 root_dir=filepath, transform=transform)
    val_loader_ts = torch.utils.data.DataLoader(
        dataset_ts, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    model_alpha = alpha_vis(args.num_classes)
    model_beta = beta_vis()
    #checkpoint_alpha = "imagenet_runs_new_new/" + args.arch + "/alpha_epoch_17.pth.tar"
    #checkpoint_beta = "imagenet_runs_new_new/" + args.arch + "/beta_epoch_17.pth.tar"
    checkpoint_alpha = "imagenet_runs/" + args.arch + "/alpha_epoch_8.pth.tar"
    checkpoint_beta = "imagenet_runs/" + args.arch + "/beta_epoch_8.pth.tar"
    model_alpha.load_state_dict(torch.load(checkpoint_alpha), strict=False)
    model_beta.load_state_dict(torch.load(checkpoint_beta), strict=False)
    from temperature_scaling import ModelWithTemperature, _ECELoss
    model = models.__dict__[args.arch](pretrained=True)
    #model = model.cuda()
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(val_loader_ts)
    model.cpu()
    scaled_model.cpu()
    scaled_model.eval()
    model.eval()

    pred_drl = np.zeros((1, 1000))
    pred_src = np.zeros((1, 1000))
    pred_ts = np.zeros((1, 1000))
    dr = np.zeros((1))
    frequency = np.zeros((1))
    with torch.no_grad():
        model_alpha.eval()
        model_beta.eval()
        for i, (data, label, freq) in enumerate(val_loader):
            if i % 10 == 0:
                print("{} batches processed".format(i))
            label_list = np.concatenate([label_list, label.cpu().numpy().reshape(-1, )], axis=0)

            target_out = scaled_model(data)
            prediction_t = F.softmax(target_out, dim=1).detach().cpu().numpy()
            pred_ts = np.concatenate([pred_ts, prediction_t], axis=0)

            target_out_src = model(data)
            prediction_t_src = F.softmax(target_out_src, dim=1).detach().cpu().numpy()
            pred_src = np.concatenate([pred_src, prediction_t_src], axis=0)

            pred = F.softmax(model_beta(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = model_alpha(data, torch.ones((data.shape[0], args.num_classes)), r_target).detach()
            prediction_t = F.softmax(target_out, dim=1).detach().cpu().numpy()

            dr = np.concatenate([dr, r_target.detach().cpu().numpy().reshape(-1, )], axis=0)
            pred_drl = np.concatenate([pred_drl, prediction_t], axis=0)
            frequency = np.concatenate([frequency, freq], axis=0)

    dr = dr[1:]
    pred_src = pred_src[1:]
    pred_ts = pred_ts[1:]
    pred_drl = pred_drl[1:]
    frequency = frequency[1:]

    conf_src = np.max(pred_src, axis=1)
    conf_ts = np.max(pred_ts, axis=1)
    conf_drl = np.max(pred_drl, axis=1)
    
    #np.save("imagenet_runs/freq_val.npy", frequency)
    #np.save("imagenet_runs/dr_val.npy", dr)

    #frequency = np.load("imagenet_runs/freq_val.npy")
    #dr = np.load("imagenet_runs/dr_val.npy")

    fig = plt.figure(figsize=(15, 10))
    xedges = np.arange(0.0, 1.08, 0.08)
    #yedges = np.arange(0.0, 1.04, 0.04)
    yedges = np.arange(0.0, 1.52, 0.08)
    H, xedges, yedges = np.histogram2d(frequency, dr, bins=(xedges, yedges))
    H = H.T
    im = plt.imshow(H, interpolation="nearest", origin="low", extent=[xedges[0], xedges[-1], yedges[0], 1])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("# of Samples", rotation=-90, va="bottom", fontsize=36)
    cbar.ax.tick_params(labelsize=18)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    plt.tick_params(labelsize=18)
    plt.xlabel("Human Selection Frequency", fontdict={'size': 36})
    plt.ylabel("Density Ratio", fontdict={'size': 36})
    plt.savefig("imagenet_runs/color_dr_new.jpg")

    """
    fig = plt.figure(figsize=(15, 10))
    xedges = np.arange(0.0, 1.08, 0.08)
    yedges = np.arange(0.0, 1.04, 0.04)
    H, xedges, yedges = np.histogram2d(frequency, conf_drl, bins=(xedges, yedges))
    H = H.T
    im = plt.imshow(H, interpolation="nearest", origin="low", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("# of Samples", rotation=-90, va="bottom", fontsize=36)
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    plt.xlabel("Human Selection Frequency", fontdict={'size': 36})
    plt.ylabel("DRL Confidence", fontdict={'size': 36})
    #plt.savefig("imagenet_runs/color_conf_drl.jpg")

    fig = plt.figure(figsize=(15, 10))
    xedges = np.arange(0.0, 1.08, 0.08)
    yedges = np.arange(0.0, 1.04, 0.04)
    H, xedges, yedges = np.histogram2d(frequency, conf_ts, bins=(xedges, yedges))
    H = H.T
    im = plt.imshow(H, interpolation="nearest", origin="low", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("# of Samples", rotation=-90, va="bottom", fontsize=36)
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    plt.xlabel("Human Selection Frequency", fontdict={'size': 36})
    plt.ylabel("TS Confidence", fontdict={'size': 36})
    #plt.savefig("imagenet_runs/color_conf_ts.jpg")

    fig = plt.figure(figsize=(15, 10))
    xedges = np.arange(0.0, 1.08, 0.08)
    yedges = np.arange(0.0, 1.04, 0.04)
    H, xedges, yedges = np.histogram2d(frequency, conf_src, bins=(xedges, yedges))
    H = H.T
    im = plt.imshow(H, interpolation="nearest", origin="low", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("# of Samples", rotation=-90, va="bottom", fontsize=36)
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    plt.xlabel("Human Selection Frequency", fontdict={'size': 36})
    plt.ylabel("Source Confidence", fontdict={'size': 36})
    #plt.savefig("imagenet_runs/color_conf_src.jpg")
    """

def conf_hsf():
    """
    Find the relation of prediction confidence and HSF
    :return: The averaged confidence
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size = 64
    workers = 4
    dataset1 = divide_freq(0, 0.2, transform)
    val_loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset2 = divide_freq(0.2, 0.4, transform)
    val_loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset3 = divide_freq(0.4, 0.6, transform)
    val_loader3 = torch.utils.data.DataLoader(
        dataset3, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset4 = divide_freq(0.6, 0.8, transform)
    val_loader4 = torch.utils.data.DataLoader(
        dataset4, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    dataset5 = divide_freq(0.8, 1.0, transform)
    val_loader5 = torch.utils.data.DataLoader(
        dataset5, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader_list = [val_loader1, val_loader2, val_loader3, val_loader4, val_loader5]

    conf_avg = []
    print("Inferring model {}".format(args.arch))
    model_alpha = alpha_vis(args.num_classes)
    model_beta = beta_vis()
    checkpoint_alpha = "imagenet_runs_new/" + args.arch + "/beta_epoch_3.pth.tar"
    checkpoint_beta = "imagenet_runs_new/" + args.arch + "/beta_epoch_3.pth.tar"
    model_alpha.load_state_dict(torch.load(checkpoint_alpha), strict=False)
    model_beta.load_state_dict(torch.load(checkpoint_beta), strict=False)
    model_alpha.cuda()
    model_beta.cuda()
    model_alpha.eval()
    model_beta.eval()

    for k in range(5):
        avg_val = 0
        val_loader = val_loader_list[k]
        for j, (data, label) in enumerate(val_loader):
            data = data.cuda()
            pred = F.softmax(model_beta(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = model_alpha(data, torch.ones((data.shape[0], args.num_classes)), r_target).detach()
            prediction_t = F.softmax(target_out, dim=1).detach().cpu().numpy()
            avg_val += np.sum(np.max(prediction_t, axis=1))
        avg_val /= len(val_loader.dataset)
        conf_avg.append(avg_val)
    print(conf_avg)
    #runs/alexnet/20 [0.009523834956526266, 0.010177904888704249, 0.010064073676171839, 0.010169655494148545, 0.010786558681004785]

def avh_score(x, w):
    """
    AVH score is used to replace the prediction probability
    x with shape (1, num_features), w with shape (num_features, n_classes)
    :return: avh score of a single sample, with type float
    """
    avh_score = np.arccos(np.dot(x, w.transpose())/(np.linalg.norm(x)*np.linalg.norm(w)))
    avh_score = avh_score / np.sum(avh_score)
    return avh_score

def avh_dr_hsf():
    import matplotlib.pyplot as plt
    global args
    args = parser.parse_args()
    """
    frequency_list = np.load('frequency.npy')
    name_list = np.load('filenames.npy')
    filepath = "CRST/imagenet_v2/imagenetv2-matched-frequency-format-val/"
    file_list = [0] * frequency_list.shape[0]
    label_list = [0] * frequency_list.shape[0]
    for path in os.listdir(filepath):
        name = os.path.join(filepath, path)
        for p in os.listdir(name):
            name_idx = np.where(name_list == p[:-5])[0][0]
            list_tmp = os.path.join(path, p)
            file_list[name_idx] = list_tmp
            label_list[name_idx] = int(path)
    file_list = np.array(file_list)
    label_list = np.array(label_list)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size = 64
    workers = 4
    dataset = ImageClassdataFreq(img_list=file_list, label_list=label_list, freq_list=frequency_list,
                                 root_dir=filepath, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


    model_alpha = alpha_vis_bottom(args.num_classes)
    model_beta = beta_vis()
    checkpoint_alpha = "imagenet_runs/" + args.arch + "/alpha_epoch_10.pth.tar"
    checkpoint_beta = "imagenet_runs/" + args.arch + "/beta_epoch_10.pth.tar"
    model_alpha.load_state_dict(torch.load(checkpoint_alpha), strict=False)
    model_beta.load_state_dict(torch.load(checkpoint_beta), strict=False)

    avh_dr_list = np.zeros((1))
    frequency = np.zeros((1))

    w = model_alpha.final_layer.weight.data
    w = w.cpu().numpy()

    with torch.no_grad():
        model_alpha.eval()
        model_beta.eval()
        for i, (data, label, freq) in enumerate(val_loader):
            if i % 10 == 0:
                print("{} batches processed".format(i))
            label = label.cpu().numpy().reshape(-1, )
            label_list = np.concatenate([label_list, label], axis=0)

            pred = F.softmax(model_beta(data, None, None, None, None).detach(), dim=1)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            embedding, _ = model_alpha(data, torch.ones((data.shape[0], args.num_classes)), r_target)
            r_target = r_target.detach().cpu().numpy()
            embedding = embedding.detach().cpu().numpy()
            avh = avh_score(embedding, w)
            tmp = np.zeros(avh.shape[0])
            for j in range(avh.shape[0]):
                tmp[j] = avh[j][label[j]]
            avh = r_target.reshape(-1, ) * tmp

            avh_dr_list = np.concatenate([avh_dr_list, avh], axis=0)
            frequency = np.concatenate([frequency, freq], axis=0)
    avh_dr_list = avh_dr_list[1:]
    frequency = frequency[1:]

    np.save("imagenet_runs/avh_dr.npy", avh_dr_list)
    np.save("imagenet_runs/freq_avh_dr.npy", frequency)
    """
    avh_dr_list = np.load("imagenet_runs/avh_dr.npy") * 1e4
    frequency = np.load("imagenet_runs/freq_avh_dr.npy")
    print(np.max(avh_dr_list))
    print(np.min(avh_dr_list))
    avh_dr_list = (avh_dr_list - np.mean(avh_dr_list))/0.1

    fig = plt.figure(figsize=(15, 10))
    xedges = np.arange(0.0, 1.08, 0.08)
    yedges = np.arange(0.0, 1.04, 0.04)
    H, xedges, yedges = np.histogram2d(frequency, avh_dr_list, bins=(xedges, yedges))
    H = H.T
    im = plt.imshow(H, interpolation="nearest", origin="low", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("# of Samples", rotation=-90, va="bottom", fontsize=36)
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    plt.xlabel("Human Selection Frequency", fontdict={'size': 36})
    plt.ylabel("Density Ratio", fontdict={'size': 36})
    plt.savefig("imagenet_runs/avh_dr_alexnet.jpg")

if __name__ == '__main__':
    #imagenet_drl()
    #test_density_ratio()
    #main()
    #bar_plot_freq()
    bar_plot_freq_binary()
    #ts_model()
    #reliability_plot()
    #plot_freq_prob_color()
    #conf_hsf()
    #avh_dr_hsf()


# AlexNet, VGG-19, ResNet-50, and DenseNet-121
# Alexnet and VGG has initial lr 0.01. The other two 0.1
# CUDA_VISIBLE_DEVICES=0 python imagenet_train.py -a alexnet --lr 0.01 --tgt-dir

#[2.58291555 3.32628701 3.3149397  3.19687839 4.08868733]
#[1.1971366  5.76735824 4.37317019 8.5476759  4.35897263]
#[1.21449738 1.21467633 1.21806817 1.21676748 1.22086449] // [0.58732552 0.59361886 0.59559961 0.59684119 0.59703413]
#[0.66336011 0.66817579 0.66872291 0.6693422  0.6703458 ]

"""
[[0.97229409 0.95253672 0.88140023 0.84967211 0.84128868]                      
 [1.11632009 1.09676516 1.03275625 1.00927243 0.97114593]                      
 [0.80342513 0.80219394 0.80174972 0.80057589 0.80127554]                      
 [1.4347434  1.43383087 1.43715699 1.43844939 1.4493234 ]]  
"""


"""
source: [0.001008764577988246, 0.0008569086701185267, 0.0007200288150604599, 0.0005186348780959712, 0.00032430531340512]
TS: [0.0010033893938315077, 0.0008533564322908344, 0.0007176940132867467, 0.0005178722550028677, 0.00032417293895774946]
DRL: [0.0009991918776664146, 0.0009992098530086269, 0.0009991793385683084, 0.0009992252470639484, 0.00099923903148058]

[0.433276668522093, 0.41495298428667915, 0.443778551227591, 0.39484069324350646, 0.3278765969503864]
[0.28345583417745446, 0.2437211305233499, 0.21107980442706345, 0.1501849106942696, 0.09199750537058798]
[0.016814313611771298, 0.016577781908349437, 0.016784010881571894, 0.016675725387592454, 0.016684575552601406]
"""