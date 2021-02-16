import torch.nn as nn
import math
import torch
from .wider_resnet import WiderResNetA2
affine_par = True


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
    dconv = nn.ConvTranspose2d(channels, channels, kernel_size=ratio * 2, stride=ratio, padding=ratio//2, bias=False, groups=channels)
    dconv.weight.requires_grad = False
    filler_bilinear(dconv)
    return dconv


def Res38_Backbone(num_classes=19, args=None):
    print('Learning with ResNet-38...')
    model = WiderResNetA2([3, 3, 6, 3, 1, 1], dilation=True)
    wider_resnet38_path = 'res38_init_imgnet.pth'  # TODO: Set your own model path here
    checkpoint = torch.load(wider_resnet38_path)
    print('loading rew38 params:', len(checkpoint['state_dict'].keys()))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = nn.Sequential(
        model,
        nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0, bias=True, groups=1),
        fixed_up(8, num_classes)
    )
    return model
