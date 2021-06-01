"""
New training process for DRST
"""

import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'ucifar100', 'cifar9', 'cs', 'sc', 'ms'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    global best_acc
    def create_model(args):
        model_alpha, model_beta = None, None
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model_alpha = models.build_wideresnet_alpha(depth=args.model_depth,
                                                        widen_factor=args.model_width,
                                                        dropout=0,
                                                        num_classes=args.num_classes)
            model_beta = models.build_wideresnet_beta(depth=args.model_depth,
                                                        widen_factor=args.model_width,
                                                        dropout=0,
                                                        num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model_alpha.parameters()) * 2 / 1e6))
        return model_alpha, model_beta

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        "Process rank: {args.local_rank},"
        "device: {args.device}, "
        "n_gpu: {args.n_gpu}, "
        "distributed training: {bool(args.local_rank != -1)}, "
        "16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        if not os.path.exists(args.out):
            os.makedirs(args.out)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10' or args.dataset == 'ms':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar9' or args.dataset == "cs" or args.dataset == "sc":
        args.num_classes = 9
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100' or args.dataset == 'ucifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset, test_src_dataset = DATASET_GETTERS[args.dataset](args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    test_src_loader = DataLoader(
        test_src_dataset,
        sampler=SequentialSampler(test_src_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model_alpha, model_beta = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model_alpha.to(args.device)
    model_beta.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters_alpha = [
        {'params': [p for n, p in model_alpha.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model_alpha.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    grouped_parameters_beta = [
        {'params': [p for n, p in model_beta.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model_beta.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_alpha = optim.SGD(grouped_parameters_alpha, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    optimizer_beta = optim.SGD(grouped_parameters_beta, lr=args.lr * 0.1,
                                momentum=0.9, nesterov=args.nesterov)

    args.epochs = int(math.ceil(args.total_steps / args.eval_step))
    scheduler_alpha = get_cosine_schedule_with_warmup(
        optimizer_alpha, args.warmup, args.total_steps)
    scheduler_beta = get_cosine_schedule_with_warmup(
        optimizer_beta, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model_alpha = ModelEMA(args, model_alpha, args.ema_decay)
        ema_model_beta = ModelEMA(args, model_beta, args.ema_decay)

    args.start_epoch = 0

    #checkpoint = torch.load("results/cifar9@"+str(args.num_labeled)+".5/model_best.pth.tar")
    #checkpoint = torch.load("results/stl9@" + str(args.num_labeled) + "/model_best.pth.tar")
    #checkpoint = torch.load("results/cs_pretrained@" + str(args.num_labeled) + "/model_best.pth.tar")
    #checkpoint = torch.load("results/cifar100@" + str(args.num_labeled) + "/model_best.pth.tar")
    #checkpoint = torch.load("results/ucifar100@" + str(args.num_labeled) + "_new/model_best.pth.tar")
    checkpoint = torch.load("results/sc@" + str(args.num_labeled) + "/model_best.pth.tar")
    #checkpoint = torch.load("results/ms@" + str(args.num_labeled) + "/model_best.pth.tar")
    logger.info("==> Resuming from checkpoint..")
    print("Best acc of checkpoint: {}".format(checkpoint['acc']))
    model_alpha.load_state_dict(checkpoint['state_dict'])
    if args.use_ema:
        ema_model_alpha.ema.load_state_dict(checkpoint['ema_state_dict'])
        ema_model_beta.ema.load_state_dict(checkpoint['ema_state_dict'], strict=False)
    else:
        ema_model_alpha = None
        ema_model_beta = None
    logger.info("==> Checkpoint resumed!")
    """
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        #best_acc = checkpoint['best_acc']
        #args.start_epoch = checkpoint['epoch']
        model_alpha.load_state_dict(checkpoint['state_dict'])
        #model_beta.load_state_dict(checkpoint['state_dict']) # optional?
        if args.use_ema:
            ema_model_alpha.ema.load_state_dict(checkpoint['ema_state_dict'])
            ema_model_beta.ema.load_state_dict(checkpoint['ema_state_dict'])
        #optimizer_alpha.load_state_dict(checkpoint['optimizer'])
        #optimizer_beta.load_state_dict(checkpoint['optimizer'])
        #scheduler_alpha.load_state_dict(checkpoint['scheduler'])
        #scheduler_beta.load_state_dict(checkpoint['scheduler'])
    """
    #if args.amp:
    #    from apex import amp
    #    model, optimizer = amp.initialize(
    #        model, optimizer, opt_level=args.opt_level)

    #if args.local_rank != -1:
    #    model = torch.nn.parallel.DistributedDataParallel(
    #        model, device_ids=[args.local_rank],
    #        output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("Task = {}@{}".format(args.dataset, args.num_labeled))
    logger.info("Num Epochs = {}".format(args.epochs))
    logger.info("Batch size per GPU = {}".format(args.batch_size))
    logger.info(
        "Total train batch size = {}".format(args.batch_size * args.world_size))
    logger.info("Total optimization steps = {}".format(args.total_steps))

    #model_alpha.zero_grad()
    #model_beta.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader, test_src_loader,
          model_alpha, model_beta, optimizer_alpha, optimizer_beta, ema_model_alpha, ema_model_beta, scheduler_alpha, scheduler_beta)

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

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, test_src_loader,
          model_alpha, model_beta, optimizer_alpha, optimizer_beta, ema_model_alpha, ema_model_beta, scheduler_alpha, scheduler_beta):
    #if args.amp:
    #    from apex import amp
    global best_acc
    test_accs = []
    data_time = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)


    model_alpha.train()
    model_beta.train()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    sign_variable = torch.autograd.Variable(torch.FloatTensor([0]))
    for epoch in range(args.start_epoch, args.epochs):
        for batch_idx in range(args.eval_step):
            if batch_idx > 4:   # 4 for mnist
                break
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            #inputs_concat = torch.cat((inputs_x, inputs_x, inputs_u_w[:batch_size], inputs_u_s[:batch_size])).to(args.device)
            targets_x = targets_x.to(args.device)

            label_concat = torch.cat(
                (torch.FloatTensor([1, 0]).repeat(int(batch_size), 1),
                 torch.FloatTensor([0, 1]).repeat(int(batch_size * args.mu), 1),
                 torch.FloatTensor([0, 1]).repeat(int(batch_size * args.mu), 1)), dim=0)
            label_concat = label_concat.to(args.device)

            prob = model_beta(inputs, None, None, None, None)
            assert (F.softmax(prob.detach(), dim=1).cpu().numpy().all() >= 0 and F.softmax(prob.detach(),
                                                                                           dim=1).cpu().numpy().all() <= 1)
            loss_dis = bce_loss(prob, label_concat)
            prediction = F.softmax(prob, dim=1).detach()
            p_s = prediction[:, 0].reshape(-1, 1)
            p_t = prediction[:, 1].reshape(-1, 1)
            r = p_s / p_t
            # Separate source sample density ratios from target sample density ratios
            r_source = r[:batch_size].reshape(-1, 1)
            r_target = r[batch_size:].reshape(-1, 1)
            p_t_target = p_t[batch_size:]
            label_train_onehot = torch.zeros([batch_size, args.num_classes])
            for j in range(batch_size):
                label_train_onehot[j][targets_x[j].long()] = 1

            logits_x = model_alpha(inputs[:batch_size], label_train_onehot.cuda(), r_source.detach().cuda())
            r_target_w, r_target_s = r_target.chunk(2)
            logits_u_w = model_alpha(inputs[batch_size:batch_size*(args.mu+1)], torch.ones((batch_size*args.mu, args.num_classes)).cuda(), r_target_w.detach().cuda())

            #logits_u_w = logits_stl_to_cifar(logits_u_w)

            T = 1  # 1
            Lx = torch.sum(logits_x)
            pseudo_label = torch.softmax(logits_u_w.detach() / T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            mask = max_probs.detach().ge(args.threshold).float().cuda()

            label_train_onehot = torch.zeros([batch_size*args.mu, args.num_classes])
            for j in range(batch_size):
                label_train_onehot[j][targets_u[j].long()] = 1
            logits_u_s = model_alpha(inputs[batch_size*(args.mu+1):]*mask.reshape(-1, 1, 1, 1),
                                     label_train_onehot.cuda()*mask.reshape(-1, 1),
                                     r_target_s.detach().cuda()*mask.reshape(-1, 1))
            #logits_u_s = model_alpha(inputs[batch_size * (args.mu + 1):], label_train_onehot.cuda(), r_target_s.detach().cuda())

            #logits_u_s = logits_stl_to_cifar(logits_u_s)

            # Here we assume lambda_u is 1 (original ce loss: loss = Lx + args.lambda_u * Lu)
            Lu = torch.sum(logits_u_s)
            L_sum = Lx + args.lambda_u * Lu

            pred_target = F.softmax(torch.cat((logits_u_w, logits_u_s), dim=0), dim=1)
            prob_grad_r = model_beta(inputs[batch_size:], torch.cat((logits_u_w, logits_u_s), dim=0).detach(),
                                     pred_target.detach(), p_t_target.detach(), sign_variable)
            loss_r = torch.sum(prob_grad_r.mul(torch.zeros(prob_grad_r.shape).cuda()))

            if batch_idx < 50 and epoch == 0:   # 128 for mnist
                optimizer_beta.zero_grad()
                loss_dis.backward()
                optimizer_beta.step()

                optimizer_beta.zero_grad()
                loss_r.backward()
                optimizer_beta.step()
            scheduler_beta.step()

            if (batch_idx + 1) % 1 == 0:
                if args.amp:
                    with amp.scale_loss(Lx, optimizer_alpha) as scaled_loss_x:
                        optimizer_alpha.zero_grad()
                        scaled_loss_x.backward()
                        optimizer_alpha.step()
                    with amp.scale_loss(Lu, optimizer_alpha) as scaled_loss_s:
                        optimizer_alpha.zero_grad()
                        scaled_loss_s.backward()
                        optimizer_alpha.step()
                else:
                    optimizer_alpha.zero_grad()
                    #Lx.backward()
                    #optimizer_alpha.step()
                    #optimizer_alpha.zero_grad()
                    #Lu.backward()
                    L_sum.backward()
                    optimizer_alpha.step()

            scheduler_alpha.step()

            if args.use_ema:
                ema_model_alpha.update(model_alpha)
                ema_model_beta.update(model_beta)

            mask_probs.update(mask.mean().item())
            if (batch_idx + 1) % 128 == 0:
                print("Train epoch: {}, iter: {}/{} finished".format(
                    epoch + 1, batch_idx + 1, args.eval_step
                ))

        if args.use_ema:
            print("EMA model used")
            test_model_alpha = ema_model_alpha.ema
            test_model_beta = ema_model_beta.ema
        else:
            print("EMA model not used")
            test_model_alpha = model_alpha
            test_model_beta = model_beta
        #print("r_source sampled: ", r_source.reshape(-1, ).cpu().numpy()[:3])
        #print("r_target_w sampled: ", r_target_w.reshape(-1, ).cpu().numpy()[:3])
        #print("r_target_s sampled: ", r_target_s.reshape(-1, ).cpu().numpy()[:3])

        # Test
        if args.local_rank in [-1, 0]:
            #logger.info('Target test data')
            #test_loss, test_acc = test(args, test_loader, test_model_alpha, test_model_beta, epoch)
            #logger.info('Source test data')
            test_loss, test_acc = test(args, test_src_loader, test_model_alpha, test_model_beta, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            if args.use_ema:
                ema_to_save_alpha = ema_model_alpha.ema.module if hasattr(
                    ema_model_alpha.ema, "module") else ema_model_alpha.ema
                ema_to_save_beta = ema_model_beta.ema.module if hasattr(
                    ema_model_beta.ema, "module") else ema_model_beta.ema
            save_directory = "results/drst_models/"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            if is_best:
                torch.save(model_alpha.state_dict(), save_directory + "alpha_best.pth.tar")
                torch.save(model_beta.state_dict(), save_directory + "beta_best.pth.tar")
                if args.use_ema:
                    torch.save(ema_to_save_alpha.state_dict(), save_directory + "ema_alpha_best.pth.tar")
                    torch.save(ema_to_save_beta.state_dict(), save_directory + "ema_beta_best.pth.tar")
            #save_checkpoint({
            #    'epoch': epoch + 1,
            #    'state_dict': model_to_save.state_dict(),
            #    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            #    'acc': test_acc,
            #    'best_acc': best_acc,
            #    'optimizer': optimizer.state_dict(),
            #    'scheduler': scheduler.state_dict(),
            #}, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

def test(args, test_loader, model_alpha, model_beta, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    #if not args.no_progress:
    #    test_loader = tqdm(test_loader,
    #                       disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model_alpha.eval()
            model_beta.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            targets = targets.long()

            pred = F.softmax(model_beta(inputs, None, None, None, None).detach(), dim=1).to(args.device)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            #print(r_target[0][0])
            outputs = model_alpha(inputs, torch.ones((inputs.shape[0], args.num_classes)).cuda(),
                                     r_target.cuda()).detach()
            #outputs = model_alpha(inputs, torch.ones((inputs.shape[0], args.num_classes)).cuda(),
            #                      torch.ones((inputs.shape[0], args.num_classes)).cuda()).detach()

            #outputs = logits_stl_to_cifar(outputs)
            #print(outputs[0:5], targets[0:5])
            loss = F.cross_entropy(outputs, targets)
            #print(outputs[:5])
            #print(torch.argmax(outputs, dim=1))
            #print(targets)
            #print("")
            #print(r_target)
            #print(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            #print(prec1, prec5)
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_idx+1) % 50 == 0:
                print("Test iter {} completed".format(batch_idx+1))
            #if not args.no_progress:
            #    test_loader.set_description(
            #        "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
            #            batch=batch_idx + 1,
            #            iter=len(test_loader),
            #            data=data_time.avg,
            #            bt=batch_time.avg,
            #            loss=losses.avg,
            #            top1=top1.avg,
            #            top5=top5.avg,
            #        ))
        #if not args.no_progress:
        #    test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg

def label_stl_and_cifar(label):
    """
    Transfer label from STL10 to CIFAR10 or CIFAR10 to STL10 (symmetrical)
    CIFAR10:
        airplane,    automobile, bird,   cat, deer, dog,     frog,  horse,     ship, truck
    STL10:
        airplane,    bird,       car,    cat, deer, dog,     horse, monkey,    ship, truck
    :param label: the label index of CIFAR10 label. shape=(batch_size, )
    :return: transfromed STL10 label index
    """
    new_label = label.detach().clone()
    for i in range(new_label.shape[0]):
        if new_label[i] == 1:
            new_label[i] = 2
        elif new_label[i] == 2:
            new_label[i] = 1
        elif new_label[i] == 6:
            new_label[i] = 7
        elif new_label[i] == 7:
            new_label[i] = 6
    return new_label

def logits_stl_to_cifar(logits):
    """
    Transfer logits from STL10 to CIFAR10 or CIFAR10 to STL10 (symmetrical)
    :param logits: predicted output. shape=(batch_size, num_classes)
    :return: transformed logits
    """
    logit_tmp = logits[:, 1]
    logits[:, 1] = logits[:, 2]
    logits[:, 2] = logit_tmp

    logit_tmp = logits[:, 6]
    logits[:, 6] = logits[:, 7]
    logits[:, 7] = logit_tmp
    return logits

if __name__ == '__main__':
    main()

"""
Training settings: 
CIFAR-STL9: on STL9 test set:
                    source: 53.83
                    source+tgt: 59.61
                    DRL: 69.38
            
            on CIFAR9 test set: 91.60 / 95.17

Training based on source model (which only uses CIFAR9 data) gives much larger boost
In paper table: Fixmatch number comes from training with both source and target         
    
STL9-CIFAR: source + tgt: 61.13 (61.82, 62.58)
            DRL: 62.06
            
            on STL9 test set: 86.51 / 84.97  
                    
MNIST-SVHN: source: 30.78 / 26.50
            DRL: 30.96
            
            on MNIST test set: 99.43 / 99.46
"""