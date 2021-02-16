"""
DeepCORAL Implementation. We train it, and use the pretrained model to
extract aligned features. We use the feature obtained from the second last layer
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import torch.utils.data as Data
import urllib
import pickle
import gzip
import numpy as np
from torchvision import transforms, datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = {
    'data_path': 'OfficeHome/',
    'kwargs' : {'num_workers': 4},
    'batch_size': 64,
    'epoch': 200,
    'lr': 1e-3,
    'momentum': .9,
    'seed': 200,
    'log_interval': 1,
    'l2_decay': 0,
    'lambda': 10,
    'backbone': 'resnet50',
    'n_class': 65,
}

fc_layer = {
    'alexnet': 256 * 6 * 6,
    'resnet50': 2048,
    "None": 256,
}

def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, ),
                                          std=(0.5, ))])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root="data",
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=64,
        shuffle=True)

    return mnist_data_loader

def get_svhn(train):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Grayscale(),
                                      transforms.Resize([28, 28]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, ),
                                          std=(0.5, ))])

    if train:
        mode = "train"
    else:
        mode = "test"
    # dataset and data loader
    svhn_dataset = datasets.SVHN(root="data",
                                   split=mode,
                                   transform=pre_process,
                                   download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=64,
        shuffle=True)

    return svhn_data_loader

class USPS(Data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num of Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f)
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

def get_usps(train):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5, ),
                                          std=(0.5, ))])

    # dataset and data loader
    usps_dataset = USPS(root="data",
                        train=train,
                        transform=pre_process,
                        download=True)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=64,
        shuffle=True)

    return usps_data_loader

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

def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE).mm(source)
    cs = (source.t().mm(source) - (tmp_s.t().mm(tmp_s)) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE).mm(target)
    ct = (target.t().mm(target) - (tmp_t.t().mm(tmp_t)) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

def Relaxed_CORAL(source, target, beta=0.5):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(DEVICE).mm(source)
    cs = (source.t().mm(source) - (tmp_s.t().mm(tmp_s)) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(DEVICE).mm(target)
    ct = (target.t().mm(target) - (tmp_t.t().mm(tmp_t)) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - (1+beta)*ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

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
            #n_features = model_resnet.fc.in_features
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
        source = source.view(source.size(0), fc_layer[self.backbone])
        source = self.compress(source)
        if self.backbone == 'alexnet':
            source = self.fc(source)
        if self.isTrain:
            target = self.sharedNet(target)
            target = target.view(target.size(0), fc_layer[self.backbone])
            target = self.compress(target)
            if self.backbone == 'alexnet':
                target = self.fc(target)

            coral_loss = CORAL(source, target)

        clf = self.cls_fc(source)
        return clf, coral_loss

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

class source_digit_net(nn.Module):
    def __init__(self, num_classes):
        super(source_digit_net, self).__init__()
        model_resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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

        self.cls_fc.weight.data.normal_(0, 0.005)

    def forward(self, source):
        source = self.sharedNet(source)
        source = source.view(source.size(0), 2048)
        source = self.compress(source)
        clf = self.cls_fc(source)
        return clf

class extractNet(nn.Module):
    def __init__(self):
        super(extractNet, self).__init__()
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

    def forward(self, input):
        x = self.sharedNet(input)
        return x

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, source_name="None", target_name="None"):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for e in range(25):
        # Train
        model.train()
        model.isTrain = True
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, loss_coral = model(data_source, data_target)
            loss_cls = criterion(label_source_pred, label_source.view(-1, ))
            loss = loss_cls + CFG['lambda'] * loss_coral
            loss.backward()
            optimizer.step()
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)], \
                    total_Loss: {:.6f}, \
                    cls_Loss: {:.6f}, \
                    coral_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    100. * i / n_batch, loss.item(), loss_cls.item(), loss_coral.item()))

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(target_test_loader.dataset)
        with torch.no_grad():
            model.isTrain = False
            for data, target in target_test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                s_output, _ = model(data, None)
                test_loss += criterion(s_output, target.view(-1, ))
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target.view(-1, ).data)

        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
        print('source: {} to target: {}, max correct: {}, max accuracy{: .4f}%\n'.format(
            source_name, target_name, correct, 100. * correct / len_target_dataset))
    torch.save(model, "models/deepcoral_"+source_name+"_"+target_name+".pkl")

def train_relaxed(source_loader, target_train_loader, target_test_loader, model, optimizer, source_name, target_name):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    beta = 0.5
    for e in range(25):
        # Train
        model.train()
        model.isTrain = True
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, loss_coral = model(data_source, data_target, beta)
            loss_cls = criterion(label_source_pred, label_source.view(-1, ))
            loss = loss_cls + CFG['lambda'] * loss_coral
            loss.backward()
            optimizer.step()
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)], \
                        total_Loss: {:.6f}, \
                        cls_Loss: {:.6f}, \
                        coral_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    100. * i / n_batch, loss.item(), loss_cls.item(), loss_coral.item()))

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(target_test_loader.dataset)
        with torch.no_grad():
            model.isTrain = False
            for data, target in target_test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                s_output, _ = model(data, None, None)
                test_loss += criterion(s_output, target.view(-1, ))
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target.view(-1, ).data)

        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
        print('source: {} to target: {}, max correct: {}, max accuracy{: .4f}%\n'.format(
            source_name, target_name, correct, 100. * correct / len_target_dataset))
    torch.save(model, "models/relaxed_deepcoral_" + source_name + "_" + target_name + ".pkl")

def load_data(src, tar, root_dir):
    source_loader = dataloader(
        root_dir, src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = dataloader(
        root_dir, tar, CFG['batch_size'], False, CFG['kwargs'])
    target_test_loader = dataloader(
        root_dir, tar, CFG['batch_size'], False, CFG['kwargs'])
    return source_loader, target_train_loader, target_test_loader

def load_digit_data(domain, train):
    if domain == "mnist":
        return get_mnist(train)
    elif domain == "svhn":
        return get_svhn(train)
    elif domain == "usps":
        return get_usps(train)
    else:
        print ("Invalid domain name")
        return None

def deep_coral_feature(source_name, target_name):
    # Note: The digit dataset all has 101 as suffix in the saved file name
    #source_loader = dataloader(CFG["data_path"], source_name, CFG["batch_size"], False, CFG["kwargs"])
    #target_loader = dataloader(CFG["data_path"], target_name, CFG["batch_size"], False, CFG["kwargs"])
    source_loader = load_digit_data(source_name, train=True)
    target_loader = load_digit_data(target_name, train=True)
    model = torch.load("models/deepcoral_"+source_name+"_"+target_name+".pkl")
    #extractModel = extractNet().to(DEVICE)
    model.eval()
    #extractModel.eval()
    print ("Start extracting features")
    with torch.no_grad():
        #sourceAligned = torch.FloatTensor(torch.ones(1, 2048, 1, 1)).to(DEVICE)
        #targetAligned = torch.FloatTensor(torch.ones(1, 2048, 1, 1)).to(DEVICE)
        #sourceOrig = torch.FloatTensor(torch.ones(1, 2048, 1, 1)).to(DEVICE)
        #targetOrig = torch.FloatTensor(torch.ones(1, 2048, 1, 1)).to(DEVICE)
        sourceAligned = torch.FloatTensor(torch.ones(1, 256)).to(DEVICE)
        targetAligned = torch.FloatTensor(torch.ones(1, 256)).to(DEVICE)
        sourceOrig = torch.FloatTensor(torch.ones(1, 256)).to(DEVICE)
        targetOrig = torch.FloatTensor(torch.ones(1, 256)).to(DEVICE)
        sourceY = torch.LongTensor([0])
        targetY = torch.LongTensor([0])

        for data, target_s in source_loader:
            data = data.to(DEVICE)
            aligned_source = model.compress(model.sharedNet(data).view(-1, 2048)).detach()
            sourceAligned = torch.cat((sourceAligned, aligned_source), dim=0)
            #orig = extractModel(data)
            sourceY = torch.cat((sourceY, target_s.view(-1, )), dim=0)
            #sourceOrig = torch.cat((sourceOrig, orig), dim=0)
        for data, target_t in target_loader:
            data = data.to(DEVICE)
            aligned_target = model.compress(model.sharedNet(data).view(-1, 2048)).detach()
            targetAligned = torch.cat((targetAligned, aligned_target), dim=0)
            #orig = extractModel(data)
            targetY = torch.cat((targetY, target_t.view(-1, )), dim=0)
            #targetOrig = torch.cat((targetOrig, orig), dim=0)
    sourceAligned = sourceAligned.view(sourceAligned.shape[0], 256)
    targetAligned = targetAligned.view(targetAligned.shape[0], 256)
    sourceOrig = sourceOrig.view(sourceOrig.shape[0], 256)
    targetOrig = targetOrig.view(targetOrig.shape[0], 256)
    print (sourceAligned.shape, targetAligned.shape, sourceOrig.shape, targetOrig.shape, sourceY.shape, targetY.shape)
    torch.save(sourceAligned[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_sourceAligned101.pkl")
    torch.save(targetAligned[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_targetAligned101.pkl")
    #torch.save(sourceOrig[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_sourceOriginal101.pkl")
    #torch.save(targetOrig[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_targetOriginal101.pkl")
    torch.save(sourceY[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_sourceY101.pkl")
    torch.save(targetY[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_targetY101.pkl")

def deep_coral_feature_2(source_loader, target_loader, source_name, target_name):
    model = torch.load("models/deepcoral_"+source_name+"_"+target_name+".pkl")
    model.eval()
    with torch.no_grad():
        #sourceAligned = torch.FloatTensor(torch.ones(1, 256, 1, 1)).to(DEVICE)
        #targetAligned = torch.FloatTensor(torch.ones(1, 256, 1, 1)).to(DEVICE)
        sourceAligned = torch.FloatTensor(torch.ones(1, 2048)).to(DEVICE)
        targetAligned = torch.FloatTensor(torch.ones(1, 2048)).to(DEVICE)
        sourceY = torch.LongTensor([0])
        targetY = torch.LongTensor([0])

        for data, target_s in source_loader:
            data = data.to(DEVICE)
            aligned_source = model.sharedNet(data).detach()
            aligned_source = aligned_source.view(-1, 2048)
            sourceAligned = torch.cat((sourceAligned, aligned_source), dim=0)
            sourceY = torch.cat((sourceY, target_s), dim=0)
        for data, target_t in target_loader:
            data = data.to(DEVICE)
            aligned_target = model.sharedNet(data).detach()
            aligned_target = aligned_target.view(-1, 2048)
            targetAligned = torch.cat((targetAligned, aligned_target), dim=0)
            targetY = torch.cat((targetY, target_t), dim=0)
    sourceAligned = sourceAligned.view(sourceAligned.shape[0], 2048)
    targetAligned = targetAligned.view(targetAligned.shape[0], 2048)
    print (sourceAligned.shape, targetAligned.shape, sourceY.shape, targetY.shape)
    torch.save(sourceAligned[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_sourceAligned.pkl")
    torch.save(targetAligned[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_targetAligned.pkl")
    torch.save(sourceY[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_sourceY.pkl")
    torch.save(targetY[1:], "aligned_data/deepcoral/"+source_name+"_"+target_name+"_targetY.pkl")

def deep_coral_feature_relaxed(source_loader, target_loader, source_name, target_name):
    model = torch.load("models/relaxed_deepcoral_" + source_name + "_" + target_name + ".pkl")
    model.eval()
    with torch.no_grad():
        # sourceAligned = torch.FloatTensor(torch.ones(1, 256, 1, 1)).to(DEVICE)
        # targetAligned = torch.FloatTensor(torch.ones(1, 256, 1, 1)).to(DEVICE)
        sourceAligned = torch.FloatTensor(torch.ones(1, 2048)).to(DEVICE)
        targetAligned = torch.FloatTensor(torch.ones(1, 2048)).to(DEVICE)
        sourceY = torch.LongTensor([0])
        targetY = torch.LongTensor([0])

        for data, target_s in source_loader:
            data = data.to(DEVICE)
            aligned_source = model.sharedNet(data).detach()
            aligned_source = aligned_source.view(-1, 2048)
            sourceAligned = torch.cat((sourceAligned, aligned_source), dim=0)
            sourceY = torch.cat((sourceY, target_s), dim=0)
        for data, target_t in target_loader:
            data = data.to(DEVICE)
            aligned_target = model.sharedNet(data).detach()
            aligned_target = aligned_target.view(-1, 2048)
            targetAligned = torch.cat((targetAligned, aligned_target), dim=0)
            targetY = torch.cat((targetY, target_t), dim=0)
    sourceAligned = sourceAligned.view(sourceAligned.shape[0], 2048)
    targetAligned = targetAligned.view(targetAligned.shape[0], 2048)
    print (sourceAligned.shape, targetAligned.shape, sourceY.shape, targetY.shape)
    torch.save(sourceAligned[1:], "aligned_data/deepcoral/" + source_name + "_" + target_name + "_sourceAlignedRelaxed.pkl")
    torch.save(targetAligned[1:], "aligned_data/deepcoral/" + source_name + "_" + target_name + "_targetAlignedRelaxed.pkl")
    torch.save(sourceY[1:], "aligned_data/deepcoral/" + source_name + "_" + target_name + "_sourceYRelaxed.pkl")
    torch.save(targetY[1:], "aligned_data/deepcoral/" + source_name + "_" + target_name + "_targetYRelaxed.pkl")

def entropy(p):
    p[p<1e-20] = 1e-20
    return -torch.sum(p.mul(torch.log2(p)))

def test_models():
    office_domains = ["amazon", "dslr", "webcam"]
    home_domains = ["Art", "Clipart", "Product", "RealWorld"]
    for i in range(4):
        for j in range(4):
            if i != j:
                #s_domain =office_domains[i]
                #t_domain = office_domains[j]
                #model = torch.load("models/deepcoral_"+s_domain+"_"+t_domain+".pkl")
                #target_test_loader = dataloader("office/", t_domain, CFG['batch_size'], False, CFG['kwargs'])
                s_domain = home_domains[i]
                t_domain = home_domains[j]
                model = torch.load("models/deepcoral_home_"+s_domain+"_"+t_domain+".pkl")
                target_test_loader = dataloader("OfficeHome/", t_domain, CFG['batch_size'], False, CFG['kwargs'])
                model.eval()
                test_loss = 0
                correct = 0
                clas_entropy = 0
                mis_entropy = 0
                criterion = torch.nn.CrossEntropyLoss()
                len_target_dataset = len(target_test_loader.dataset)
                mis_num = 0
                with torch.no_grad():
                    model.isTrain = False
                    for data, target in target_test_loader:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        s_output, _ = model(data, None)
                        test_loss += criterion(s_output, target)
                        pred = torch.max(s_output, 1)[1]
                        correct += torch.sum(pred == target.data)
                        clas_entropy += entropy(F.softmax(s_output, dim=1))/math.log(31, 2)
                        mis_idx = (torch.argmax(s_output, dim=1) != target).nonzero().reshape(-1, )
                        mis_pred = s_output[mis_idx]
                        mis_entropy += entropy(F.softmax(mis_pred, dim=1))/math.log(31, 2)
                        mis_num += mis_idx.shape[0]
                test_loss /= len_target_dataset
                print('\n{} to {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Clas_Entropy: {}, Mis_Entropy: {}\n'.format(
                    s_domain, t_domain, test_loss, correct, len_target_dataset,
                    100. * float(correct) / len_target_dataset, clas_entropy/len_target_dataset, mis_entropy/mis_num))

def source_only(source_name, target_name):
    print ("Source only for "+source_name+" to "+target_name)
    source_loader = dataloader("OfficeHome/", source_name, 64, True, CFG["kwargs"])
    target_loader = dataloader("OfficeHome/", target_name, 64, True, CFG["kwargs"])
    N_CLASSES = 65
    #source_loader = load_digit_data(source_name, train=True)
    #target_loader = load_digit_data(target_name, train=True)
    model = source_net(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])
    for e in range(CFG['epoch']):
        test_num, entropy_clas, test_loss, test_acc, mis_entropy_clas, mis_num, cor_entropy_clas, cor_num = 0, 0, 0, 0, 0, 0, 0, 0
        # Train
        model.train()
        iter_source = iter(source_loader)
        n_batch = len(source_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_source, label_source = data_source.to(DEVICE), label_source.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred = model(data_source)
            loss_cls = criterion(label_source_pred, label_source.view(-1, ))
            loss = loss_cls
            loss.backward()
            optimizer.step()
            """
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)], \
                    total_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    100. * i / n_batch, loss.item()))"""

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(target_loader.dataset)
        test_num = len_target_dataset
        with torch.no_grad():
            model.isTrain = False
            for data, target in target_loader:
                data, target = data.to(DEVICE), target.view(-1, ).to(DEVICE)
                s_output = model(data)
                test_loss += criterion(s_output, target)
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target.data)

                target_out = s_output
                prediction_t = F.softmax(target_out, dim=1)
                entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
                # test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
                test_loss += float(criterion(target_out, target))
                test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == target, dim=0))
                mis_idx = (torch.argmax(prediction_t, dim=1) != target).nonzero().reshape(-1, )
                mis_pred = prediction_t[mis_idx]
                cor_idx = (torch.argmax(prediction_t, dim=1) == target).nonzero().reshape(-1, )
                cor_pred = prediction_t[cor_idx]
                mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
                mis_num += mis_idx.shape[0]
                cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
                cor_num += cor_idx.shape[0]
            """print (
                "Source Only: test_loss:{:.3f}, test_acc: {:.4f}, ent_cla: {:.3f}, mis_ent_clas: {:.3f}, ent_clas_cor: {:.3f}").format(
                test_loss * 1e3 / test_num, test_acc / test_num,
                entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
            )"""

        test_loss /= len_target_dataset
        print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
            target_name, test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
        #print('source: {} to target: {}, max correct: {}, max accuracy{: .4f}%\n'.format(
        #    source_name, target_name, correct, 100. * correct / len_target_dataset))

    torch.save(model, "models/sourceOnly_"+source_name+"_"+target_name+".pkl")

def get_loss_ent(source_name, target_name):
    # Get the metrics for source only models
    print ("Source only for " + source_name + " to " + target_name)
    model = torch.load("models/sourceOnly_"+source_name+"_"+target_name+".pkl")
    test_loader = dataloader("OfficeHome/", target_name, 64, True, CFG["kwargs"])
    #test_loader = load_digit_data(target_name, train=True)
    N_CLASSES = 65
    ce_func = nn.CrossEntropyLoss()
    model.eval()
    test_num, entropy_clas, test_loss, test_acc, mis_entropy_clas, mis_num, cor_entropy_clas, cor_num = 0, 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE)
            label = label.view(-1, ).to(DEVICE)
            test_num += data.shape[0]
            target_out = model(data).detach()
            prediction_t = F.softmax(target_out, dim=1)
            #print entropy(prediction_t)
            entropy_clas += entropy(prediction_t) / math.log(N_CLASSES, 2)
            #test_loss += float(ce_func(target_out, torch.argmax(label, dim=1)))
            test_loss += float(ce_func(target_out, label))
            test_acc += float(torch.sum(torch.argmax(prediction_t, dim=1) == label, dim=0))
            mis_idx = (torch.argmax(prediction_t, dim=1) != label).nonzero().reshape(-1, )
            mis_pred = prediction_t[mis_idx]
            cor_idx = (torch.argmax(prediction_t, dim=1) == label).nonzero().reshape(-1, )
            cor_pred = prediction_t[cor_idx]
            mis_entropy_clas += entropy(mis_pred) / math.log(N_CLASSES, 2)
            mis_num += mis_idx.shape[0]
            cor_entropy_clas += entropy(cor_pred) / math.log(N_CLASSES, 2)
            cor_num += cor_idx.shape[0]
    print (
        "Source Only: test_loss:{:.3f}, test_acc: {:.4f}, ent_cla: {:.3f}, mis_ent_clas: {:.3f}, ent_clas_cor: {:.3f}").format(
        test_loss * 1e3 / test_num, test_acc / test_num,
        entropy_clas / test_num, mis_entropy_clas / mis_num, cor_entropy_clas / cor_num
    )

if __name__ == '__main__':
    torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed(CFG["seed"])
    domains = ["Art", "Clipart", "Product", "RealWorld"]
    for source in domains:
        for target in domains:
            if source != target:
                source_loader, target_train_loader, target_test_loader = load_data(source, target, CFG['data_path'])
                deep_coral_feature_relaxed(source_loader, target_test_loader, source, target)
             

