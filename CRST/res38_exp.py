"""
Change the model from deeplab v2 to resnet38
"""

import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from deeplab.model import Res_Deeplab
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import GTA5DataSet
import matplotlib.pyplot as plt
import random
import timeit
import torchvision.transforms as transforms
import util
import math
import sys
from collections import OrderedDict
from functools import partial
import torch.nn as nn
import torch
affine_par = True

start = timeit.default_timer()

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32)  # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32)  # BGR
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 4
DATA_DIRECTORY = 'dataset/gta5'
DATA_LIST_PATH = 'dataset/list/gta5/train.lst'
NUM_CLASSES = 19
IGNORE_LABEL = 255
INPUT_SIZE = '504,504'
TRAIN_SCALE = '0.5,1.5'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_STEPS = 100000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = ''
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = 'gta_src_train/'
WEIGHT_DECAY = 0.0005
MODEL = 'DeeplabRes101'
LOG_FILE = 'log'
PIN_MEMORY = True
GPU = '0'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bnrelu(channels):
    return nn.Sequential(nn.BatchNorm2d(channels),
                         nn.ReLU(inplace=True))


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm_act=bnrelu,
                 dropout=None
                 ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        nn = torch.nn

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2d(in_channels,
                                    channels[0],
                                    3,
                                    stride=stride,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0], channels[1],
                                    3,
                                    stride=1,
                                    padding=dilation,
                                    bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2d(in_channels,
                           channels[0],
                           1,
                           stride=stride,
                           padding=0,
                           bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2d(channels[0],
                                    channels[1],
                                    3, stride=1,
                                    padding=dilation, bias=False,
                                    groups=groups,
                                    dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2],
                                    1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        """
        This is the standard forward function for non-distributed batch norm
        """
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class WiderResNet(nn.Module):

    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0
                 ):
        """Wider ResNet with pre-activation (identity mapping) blocks

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and \
            a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024),
                    (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels, channels[mod_id],
                                          norm_act=norm_act)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id <= 4:
                self.add_module("pool%d" %
                                (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            out = self.classifier(out)

        return out


class WiderResNetA2(nn.Module):

    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0,
                 dilation=False,
                 adapter=False,
                 ):
        """Wider ResNet with pre-activation (identity mapping) blocks

        This variant uses down-sampling by max-pooling in the first two blocks and \
         by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer
            \with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the
            \down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        nn = torch.nn
        nn.Dropout = nn.Dropout2d
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = torch.nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))
        ]))
        if adapter:
            self.add_module("mod_adapter1",  nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                    (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            adapters = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels,
                                          channels[mod_id], norm_act=norm_act,
                                          stride=stride, dilation=dil, dropout=drop)
                ))
                if adapter:
                    adapters.append((
                        "adapter%d" %(block_id + 1),
                        nn.Conv2d(channels[mod_id][-1], 1, kernel_size=3, stride=1, padding=1, bias=True)
                    ))
                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" %
                                (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            if adapter:
                self.add_module("mod%d" % (mod_id + 2), nn.ModuleDict(OrderedDict(blocks)))
                self.add_module("mod_adapter%d" % (mod_id + 2), nn.ModuleDict(OrderedDict(adapters)))
            else:
                self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(OrderedDict([
                ("avg_pool", GlobalAvgPool2d()),
                ("fc", nn.Linear(in_channels, classes))
            ]))

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        else:
            return out


_NETS = {
    "16": {"structure": [1, 1, 1, 1, 1, 1]},
    "20": {"structure": [1, 1, 1, 3, 1, 1]},
    "38": {"structure": [3, 3, 6, 3, 1, 1]},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "wider_resnet" + name
    setattr(sys.modules[__name__], net_name, partial(WiderResNet, **params))
    __all__.append(net_name)
for name, params in _NETS.items():
    net_name = "wider_resnet" + name + "_a2"
    setattr(sys.modules[__name__], net_name, partial(WiderResNetA2, **params))
    __all__.append(net_name)


def filler_bilinear(m):
    w = m.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fixed_up(ratio=2, channels=1):
    dconv = nn.ConvTranspose2d(channels, channels, kernel_size=ratio * 2,
                               stride=ratio, padding=ratio//2, bias=False, groups=channels)
    dconv.weight.requires_grad = False
    #dconv.weight.requires_grad = True
    filler_bilinear(dconv)
    return dconv

def Res38_Backbone(num_classes=19, args=None):
    print('Learning with ResNet-38...')
    model = WiderResNetA2([3, 3, 6, 3, 1, 1], dilation=True)
    wider_resnet38_path = 'res38_init_imgnet.pth'  # TODO: Set your own model path here
    checkpoint = torch.load(wider_resnet38_path)
    model = torch.nn.DataParallel(model)  # TODO: Check more details!
    print('loading rew38 params:', len(checkpoint['state_dict'].keys()))
    state = model.state_dict()
    state_pre = checkpoint['state_dict']
    for key in state.keys():
        if key in state_pre.keys():
            state[key] = state_pre[key]
        else:
            print ("Missing key param {}".format(key))
            raise ValueError("Key missing")
    model.load_state_dict(state, strict=True)
    print ("Loaded model with best precision {}".format(checkpoint["best_prec1"]))
    model = model.module
    model = nn.Sequential(
        model,
        nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0, bias=True, groups=1),
        fixed_up(8, num_classes)
    )
    return model


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="The base network.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--train-scale", type=str, default=TRAIN_SCALE,
                        help="The scale for multi-scale training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--pin-memory", type=bool, default=PIN_MEMORY,
                        help="Whether to pin memory in train & eval.")
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug', help='True means logging debug info.',
                        default=False, action='store_true')
    return parser.parse_args()

args = get_arguments()


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model[0])

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model[1].parameters())
    b.append(model[2].parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    logger = util.set_logger(args.snapshot_dir, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    lscale, hscale = map(float, args.train_scale.split(','))
    train_scale = (lscale, hscale)

    cudnn.enabled = True

    # Create network.
    model = Res38_Backbone(num_classes=19)

    model.eval()  # use_global_stats = True
    # model.train()
    model.to(DEVICE)
    cudnn.benchmark = True
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                    train_scale=train_scale,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN, std=IMG_STD),
        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=args.pin_memory)
    #optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate},
    #                       {'params': get_10x_lr_params(model), 'lr': 10 * args.learning_rate}],
    #                      lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2.5e-4, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD([{'params': model[0].parameters(), 'lr': 2.5e-5},
                           {'params': model[1].parameters(), 'lr':2.5e-4}], lr=2.5e-4, momentum=0.9,
                          weight_decay=5e-4)

    optimizer.zero_grad()

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.to(DEVICE)
        labels = labels.long().to(DEVICE)  # shape = 4, 504, 504

        optimizer.zero_grad()

        #adjust_learning_rate(optimizer, i_iter)

        pred = model(images)  # shape = 4, 19, 500, 500, [batch_size, num_class, img_width, img_height]
        loss = loss_calc(pred, labels)
        loss.backward()
        optimizer.step()

        # print('iter = ', i_iter, 'of', args.num_steps,'completed, loss = ', loss.data.cpu().numpy())
        logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps - 1:
            print('save model ...')
            #torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'VOC12_scenes_' + str(args.num_steps) + '.pth'))
            torch.save(model.state_dict(), "src_model/gta5/res38_src.pth.tar")
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            #torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'VOC12_scenes_' + str(i_iter) + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
