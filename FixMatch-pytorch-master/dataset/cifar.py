import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import ConcatDataset

from randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_cifar9(args, root):
    print("Using CIFAR9 dataset")
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    base_dataset = datasets.CIFAR10(root, train=True, download=False)
    new_idx, last_3_idx = reset_cifar10_class(base_dataset.targets)
    labels = np.array(base_dataset.targets)
    labels[last_3_idx] -= 1
    new_labels = labels[new_idx]

    base_dataset_test = datasets.CIFAR10(root, train=False, download=False)
    new_idx_test, last_3_idx_test = reset_cifar10_class(base_dataset_test.targets)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, new_labels)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, new_label_idx=new_idx, last_idx=last_3_idx)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std), new_label_idx=new_idx, last_idx=last_3_idx)

    #test_dataset = datasets.CIFAR10(
    #    root, train=False, transform=transform_val, download=False)
    test_dataset = CIFAR10SSL(
        root, None, train=False,
        transform=transform_val, new_label_idx=new_idx_test, last_idx=last_3_idx_test)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, None

def get_stl9(args, root):
    print("Using STL9 dataset")
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.STL10(root, split="train", download=False)
    labels = np.array(base_dataset.labels)
    not_monkey_idx, last_2_idx, bird_idx, car_idx = reset_stl10_class(labels)
    labels[last_2_idx] -= 1
    new_labels = labels[not_monkey_idx]

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, new_labels)

    train_labeled_dataset = STL10SSL(
        root, train_labeled_idxs, split="train", transform=transform_labeled, new_label_idx=not_monkey_idx,
        last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx))
    train_unlabeled_dataset = STL10SSL(
        root, indexs=None, split="train", transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        download=False,
        new_label_idx=not_monkey_idx, last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx)
    )

    tgt_tgt_dataset = datasets.STL10(root, split='test', download=False)
    tgt_tgt_label = np.array(tgt_tgt_dataset.labels)
    not_monkey_idx, last_2_idx, bird_idx, car_idx = reset_stl10_class(tgt_tgt_label)
    test_dataset = STL10SSL(
        root, indexs=None, split="test", transform=transform_val, download=False,
        new_label_idx=not_monkey_idx, last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx))
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, None

def get_cifar_stl_9(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=False)
    new_idx, last_3_idx = reset_cifar10_class(base_dataset.targets)
    labels = np.array(base_dataset.targets)
    labels[last_3_idx] -= 1
    new_labels = labels[new_idx]

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, new_labels)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, new_label_idx=new_idx, last_idx=last_3_idx)

    tgt_dataset = datasets.STL10(root, split='train', download=False)
    tgt_label = np.array(tgt_dataset.labels)
    not_monkey_idx, last_2_idx, bird_idx, car_idx = reset_stl10_class(tgt_label)

    train_unlabeled_dataset_stl = STL10SSL(
        root, indexs=None, split="train", transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        download=False,
        new_label_idx=not_monkey_idx, last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx)
    )

    #train_unlabeled_dataset_cifar = CIFAR10SSL(
    #    root, train_unlabeled_idxs, train=True,
    #    transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std), new_label_idx=new_idx, last_idx=last_3_idx)

    #train_unlabeled_dataset = ConcatDataset([train_unlabeled_dataset_cifar, train_unlabeled_dataset_stl])
    train_unlabeled_dataset = train_unlabeled_dataset_stl

    #train_unlabeled_dataset = CIFAR10SSL(
    #    root, train_unlabeled_idxs, train=True,
    #    transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    tgt_tgt_dataset = datasets.STL10(root, split='test', download=False)
    tgt_tgt_label = np.array(tgt_tgt_dataset.labels)
    not_monkey_idx, last_2_idx, bird_idx, car_idx = reset_stl10_class(tgt_tgt_label)
    test_dataset = STL10SSL(
        root, indexs=None, split="test", transform=transform_val, download=False,
        new_label_idx=not_monkey_idx, last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx))

    base_dataset_test = datasets.CIFAR10(root, train=False, download=False)
    new_idx_test, last_3_idx_test = reset_cifar10_class(base_dataset_test.targets)
    test_src_dataset = CIFAR10SSL(
        root, None, train=False,
        transform=transform_val, new_label_idx=new_idx_test, last_idx=last_3_idx_test)


    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, test_src_dataset

def get_stl_cifar_9(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.STL10(root, split="train", download=False)
    labels = np.array(base_dataset.labels)
    not_monkey_idx, last_2_idx, bird_idx, car_idx = reset_stl10_class(labels)
    labels[last_2_idx] -= 1
    new_labels = labels[not_monkey_idx]

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, new_labels)

    train_labeled_dataset = STL10SSL(
        root, train_labeled_idxs, split="train", transform=transform_labeled, new_label_idx=not_monkey_idx,
        last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx))

    tgt_dataset = datasets.CIFAR10(root, train=True, download=False)
    new_idx, last_3_idx = reset_cifar10_class(tgt_dataset.targets)

    train_unlabeled_dataset_cifar = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        new_label_idx=new_idx, last_idx=last_3_idx)

    train_unlabeled_dataset = train_unlabeled_dataset_cifar

    tgt_tgt_dataset = datasets.CIFAR10(root, train=False, download=False)
    tgt_tgt_label = np.array(tgt_tgt_dataset.targets)
    new_idx, last_3_idx = reset_cifar10_class(tgt_tgt_label)
    test_dataset = CIFAR10SSL(
        root, indexs=None, train=False,
        transform=transform_val, new_label_idx=new_idx, last_idx=last_3_idx)

    base_dataset_test = datasets.STL10(root, split='test', download=False)
    not_monkey_idx, last_2_idx, bird_idx, car_idx = reset_stl10_class(np.array(base_dataset_test.labels))

    test_src_dataset = STL10SSL(
        root, indexs=None, split="test", transform=transform_val, download=False,
        new_label_idx=not_monkey_idx, last_idx=last_2_idx, reverse_indices=(bird_idx, car_idx))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, test_src_dataset

def get_mnist(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = datasets.MNIST(root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = MNISTSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, download=False)
    train_unlabled_dataset = MNISTSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch_ms(mean=normal_mean, std=normal_std), download=False)
    test_dataset = MNISTSSL(root, indexs=None, train=False, transform=transform_val, download=False)
    return train_labeled_dataset, train_unlabled_dataset, test_dataset, None

def get_mnist_svhn(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = datasets.MNIST(root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = MNISTSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled, download=False)

    train_unlabeled_dataset = SVHNSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch_ms(mean=normal_mean, std=normal_std), download=False)

    test_dataset = datasets.SVHN(
        root, split='test', transform=transform_val, download=False)

    test_src_dataset = MNISTSSL(
        root, None, train=False,
        transform=transform_labeled, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, test_src_dataset

def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_unbalanced_cifar100(args, root):
    """
    Creating an unbalanced CIFAR100 dataset with label shift by sampling different classes with different probability
    :param args: arguments
    :param root: root directory
    :return: unbalanced dataset
    """
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split_unbalanced(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    #assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(int(num_expand_x))])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

def x_u_split_unbalanced(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    prob = np.arange(250, 750, 5) / 1000.
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    #unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        label_per_class_i = int(label_per_class * prob[i])
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class_i, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    #assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(int(num_expand_x))])
    np.random.shuffle(labeled_idx)
    unlabeled_idx = labeled_idx
    return labeled_idx, unlabeled_idx

def reset_cifar10_class(labels):
    """
    Eliminate the frog class from dataset
    :param args: arguments
    :param labels: labels
    :return: the reconstructed dataset indices
    """
    labels = np.array(labels)
    not_frog_idx = np.where(labels != 6)[0]
    last_3_idx = np.where(labels > 6)[0]
    return not_frog_idx, last_3_idx

def reset_stl10_class(labels):
    """
    Eliminate the monkey class from dataset
    :param args: arguments
    :param labels: the reconstructed dataset indices
    :return: the reconstructed dataset indices
    """
    labels = np.array(labels)
    not_monkey_idx = np.where(labels != 7)[0]
    last_2_idx = np.where(labels > 7)[0]
    bird_idx = np.where(labels == 1)[0]
    car_idx = np.where(labels == 2)[0]
    return not_monkey_idx, last_2_idx, bird_idx, car_idx

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class TransformFixMatch_ms(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=32)])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=32),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, new_label_idx=None, last_idx=None):
        super(CIFAR10SSL, self).__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.targets = np.array(self.targets)
        if last_idx is not None:
            self.targets[last_idx] -= 1
        if new_label_idx is not None:
            self.data = self.data[new_label_idx]
            self.targets = np.array(self.targets)[new_label_idx]
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

class STL10SSL(datasets.STL10):
    def __init__(self, root, indexs, split,
                 transform=None, target_transform=None,
                 download=False, new_label_idx=None, last_idx=None, reverse_indices=None):
        super(STL10SSL, self).__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.targets = np.array(self.labels)
        if last_idx is not None:
            self.targets[last_idx] -= 1
        if reverse_indices is not None:
            bird_idx, car_idx = reverse_indices
            self.targets[bird_idx] = 2
            self.targets[car_idx] = 1
        if new_label_idx is not None:
            self.data = self.data[new_label_idx]
            self.targets = np.array(self.targets)[new_label_idx]
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.uint8(img)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

class MNISTSSL(datasets.MNIST):
    def __init__(self, root, indexs, train,
                 transform=None, target_transform=None,
                 download=False):
        super(MNISTSSL, self).__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.length = self.targets.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy())
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

class SVHNSSL(datasets.MNIST):
    def __init__(self, root, indexs, train,
                 transform=None, target_transform=None,
                 download=False):
        super(SVHNSSL, self).__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.length = self.targets.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy())
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100SSL, self).__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'ucifar100': get_unbalanced_cifar100,
                   'cifar9': get_cifar9,
                   'stl9': get_stl9,
                   'mnist': get_mnist,
                   'cs': get_cifar_stl_9,
                   'sc': get_stl_cifar_9,
                   'ms': get_mnist_svhn}

"""
CIFAR10: 
airplane,    automobile, bird,   cat, deer, dog,     frog,  horse,     ship, truck

STL10: 
airplane,    bird,       car,    cat, deer, dog,     horse, monkey,    ship, truck


"""