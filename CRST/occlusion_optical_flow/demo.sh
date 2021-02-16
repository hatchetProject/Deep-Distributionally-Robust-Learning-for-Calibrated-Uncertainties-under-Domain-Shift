#!/bin/bash

export PYTHONPATH=RAFT:RAFT/core

CUDA_VISIBLE_DEVICES=0 python occlusion_detection.py --model=RAFT/models/raft-sintel.pth