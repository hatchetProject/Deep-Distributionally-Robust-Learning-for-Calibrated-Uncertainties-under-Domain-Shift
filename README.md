##DISTRIBUTIONALLY ROBUST LEARNING  FOR UNSUPERVISED DOMAIN ADAPTATION

###Introduction

We propose a  distributionally robust learning (DRL) method for unsupervised domain adaptation (UDA)  that scales to modern computer-vision benchmarks.  DRL can be naturally formulated as  a competitive two-player game between a predictor and an adversary that is allowed to    corrupt the labels, subject to certain constraints, and reduces to incorporating  a density ratio between the source and target domains (under the standard log loss).  This formulation motivates the use of two neural networks that are jointly trained--   a discriminative network between the  source and target domains  for density-ratio estimation, in addition to the standard classification network. The use of a density ratio in DRL prevents the model from being overconfident on target inputs far away from the source domain. Thus,  DRL   provides conservative confidence estimation in  the target domain, even when the  target labels are not available. This conservatism motivates the use of DRL in   self-training  for sample selection, and we term the approach distributionally robust self-training (DRST). In our experiments, DRST generates more calibrated  probabilities and  achieves state-of-the-art self-training accuracy on benchmark datasets. We demonstrate that DRST captures  shape features more effectively, and reduces the extent of distributional shift during self-training. 
###Usage

####Environment

* Python 3, Python 3.6
* PyTorch 0.4.0
* CUDA version 9.0 or lower

####File introduction

We provide the introduction to the folders and files in this section
* model_layers.py defines the foundations DRST models, and all other Python files can be directly run
* visda_exp.py contained the DRL and DRST training code for VisDA 2017 dataset

