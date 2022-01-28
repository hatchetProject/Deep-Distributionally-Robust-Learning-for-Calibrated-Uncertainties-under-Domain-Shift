## Distributionally Robust Learning for Calibrated Uncertainties under Domain Shift

### Introduction

We propose a framework for learning calibrated uncertainties under domain shifts, where the source (training) distribution differs from the target (test) distribution. We detect such domain shifts by using a binary domain classifier and manage to integrate it with the task network and train them jointly end-to-end. The binary domain classifier yields a density ratio that reflects the closeness of a target (test) sample to the source (training) distribution. It is used to adjust the uncertainty of prediction in the task network. This idea of using density ratio is based on the distributionally robust learning (DRL) framework, which accounts for the domain shift through adversarial risk minimization. We show that our method generates calibrated uncertainties that benefit many downstream tasks, e.g. unsupervised domain adaptation (UDA) and semi-supervised learning (SSL). In these tasks, methods like self-training and FixMatch use uncertainties to select confident pseudo-labels for re-training. Our experiments show that the introduction of DRL leads to significant improvements in cross-domain performance. We also demonstrate that the estimated density ratios show agreement with the human selection frequencies, suggesting a positive correlation with a proxy of human perceived uncertainties.

#### Environment

* Python 3.6
* PyTorch 0.4.0
* CUDA version 9.0
(This combination of environment can be obtained by using docker images)

#### File introduction

We provide the introduction to the folders and files in this section
* model_layers.py defines the foundations of DRL, DRST and DRSSL models;
* office_exp.py contains the code for running experiments on Office31 and Office-Home;
* visda_exp.py contained the DRST training code for VisDA 2017 dataset;
* tempertaure_scaling.py contains the code for implementing TS; 
* FixMatch-pytorch-master contains the SSL training code. train.py contains the code for achieving pretrained models, while train_var.py contains the code for DRSSL;
* imagenet_train.py contains the code for training on ImageNet using DRL;
* Other files are included for plotting and discussion.


### Code Usage


The training code is generally written in a manner easy for debugging and experimenting on different tasks. Here we provide the example scripts for specific tasks.

#### Office31 and Office-Home:

For Office31 and OfficeHome, the two datasets are trained in a quite similar way, thus the only difference is in the code are the data paths needed and the 

Take the one of the tasks in the Office31 dataset as example, the code can be run by directly using:

```
python office_exp.py --num_classes=31 --src=amazon --tgt=webcam
```

When changed to OfficeHome dataset, the users can change the number of classes into 65 and assign the **path_dict** new values with the intended path. Do not forget to write new paths for the saved
models and metrics to avoid overwrite.

#### VisDA 2017

For VisDA 2017, we adopt the same training criterion as CBST. The procedure is included in visda_exp.py. In the code, we include both the 
original CBST (train_and_val_cbst()) and our DRL training (train_and_val_rescue_var3()) procedure, so that the users can compare their differences.

```
python visda_exp.py
```

The specific hyperparameters in the code can be changed via the **CONFIG** parameter, which is included at the top of the file. We set the random seed to 5 for better performance. 

#### ImageNet 

ImageNet training process is directly modified from the official PyTorch training script. The users can directly run the script and achieve corresponding calibration scores (Brier score and ECE). However, it would take 
a long time since it also does the training process. Function ts_model() contains the score calculation. Note that the code is not written in a parallelism way so batch size is set to a small value and the users need to 
make sure they have abundant computation resources before running it. Code can be run by:

```
python imagenet_train.py
```

#### Note

For other code (not for training ones, the users may need to select the functions they need and set the specific path for saved models)
