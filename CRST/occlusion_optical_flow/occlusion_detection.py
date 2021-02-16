import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

import torch
import torch.nn as nn

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def compute_optical_flow(model, im1, im2):
    padder = InputPadder(im1.shape)

    with torch.no_grad():
        im1, im2 = padder.pad(im1, im2)

        _, fwd_flow = model(im1, im2, iters=20, test_mode=True)
        # fwd_flow = fwd_flow[0]

        _, bwd_flow = model(im2, im1, iters=20, test_mode=True)
        # bwd_flow = bwd_flow[0]

        fwd_flow = padder.unpad(fwd_flow)
        bwd_flow = padder.unpad(bwd_flow)

    return fwd_flow, bwd_flow

def flow_warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask.data < 0.9999] = 0
    mask[mask.data > 0] = 1
    
    return output*mask

def L2_norm(x, dim=1, keepdim=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset, dim=dim, keepdim=keepdim)
    return l2_norm

def estimate_occlusion(fwd_flow, bwd_flow, flow_consistency_alpha=3, flow_consistency_beta=.05):
    # warp pyramid full flow
    fwd2bwd = flow_warp(fwd_flow, bwd_flow)

    # calculate flow consistency
    bwd_flow_diff = torch.abs(fwd2bwd + bwd_flow)

    # build flow consistency condition
    bwd_consist_bound = flow_consistency_beta * L2_norm(bwd_flow)
    bwd_consist_bound = torch.clamp(bwd_consist_bound, min=flow_consistency_alpha)
    
    # torch.maximum(bwd_consist_bound, flow_consistency_alpha)

    # build flow consistency mask
    noc_masks_src = L2_norm(bwd_flow_diff) < bwd_consist_bound
    return noc_masks_src.float()
    
    #  [tf.cast(tf.less(L2_norm(bwd_flow_diff[s]), 
    #                         bwd_consist_bound[s]), tf.float32)
    # noc_masks_tgt = [tf.cast(tf.less(L2_norm(fwd_flow_diff[s]),
    #                         fwd_consist_bound[s]), tf.float32)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # with torch.no_grad():
    #     images = glob.glob(os.path.join(args.path, '*.png')) + \
    #              glob.glob(os.path.join(args.path, '*.jpg'))
        
    #     images = sorted(images)
    #     for imfile1, imfile2 in zip(images[:-1], images[1:]):
    #         image1 = load_image(imfile1)
    #         image2 = load_image(imfile2)

    #         padder = InputPadder(image1.shape)
    #         image1, image2 = padder.pad(image1, image2)

    #         flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    #         # viz(image1, flow_up)
    im1_path = 'RAFT/demo-frames/frame_0016.png'
    im2_path = 'RAFT/demo-frames/frame_0019.png'
    im1 = load_image(im1_path)
    im2 = load_image(im2_path)
    fwd_flow, bwd_flow = compute_optical_flow(model, im1, im2)

    noc_mask = estimate_occlusion(fwd_flow, bwd_flow)
    occ_mask_im = (1 - noc_mask.squeeze().cpu().numpy()) * 255
    cv2.imwrite(os.path.basename(im1_path)[:-4] + '_occ.png', occ_mask_im.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
