import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def show_cam_on_image(img, mask, path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(path, np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def detect_wrong_predictions():
    from visda_exp import CONFIG, ImageClassdata, avh_score, accuracy_new, entropy
    from visda_exp import source_vis, beta_vis, alpha_vis, validate, AverageMeter
    import math
    from sklearn.metrics import brier_score_loss
    import torch.nn as nn
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val4mix': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    kwargs = {'num_workers': 1, 'pin_memory': False}
    visDA17_valset = ImageClassdata(txt_file=CONFIG["tgt_gt_list"], root_dir=CONFIG["tgt_root"],
                                    transform=data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(visDA17_valset, batch_size=CONFIG["batch_size"], shuffle=True,
                                             **kwargs)

    # DRST model
    #resume_path = "runs/best_model_comp13/"
    #print("=> Loading checkpoint '{}'".format(resume_path))
    #checkpoint_alpha = torch.load(resume_path + "alpha_best.pth.tar")
    #checkpoint_beta = torch.load(resume_path + "beta_best.pth.tar")
    resume_path = "runs/best_model_gradcam/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint_alpha = torch.load(resume_path + "alpha_epoch_12.pth.tar")
    checkpoint_beta = torch.load(resume_path + "beta_epoch_12.pth.tar")
    model_alpha = alpha_vis(12)
    model_beta = beta_vis()
    model_alpha.load_state_dict(checkpoint_alpha)
    model_beta.load_state_dict(checkpoint_beta)

    model_alpha = model_alpha.to(DEVICE)
    model_beta = model_beta.to(DEVICE)
    model_alpha.eval()
    model_beta.eval()
    """
    # ASG model
    model_asg = models.resnet101(pretrained=False)
    num_ftrs = model_asg.fc.in_features
    model_asg.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/res101_asg/res101_vista17_best.pth.tar"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path)
    state_pre = checkpoint["state_dict"]
    state = model_asg.state_dict()
    for key in state.keys():
        if key in state_pre.keys():
            state[key] = state_pre[key]
        elif key == "fc.0.weight":
            state[key] = state_pre["fc_new.0.weight"]
        elif key == "fc.0.bias":
            state[key] = state_pre["fc_new.0.bias"]
        elif key == "fc.2.weight":
            state[key] = state_pre["fc_new.2.weight"]
        elif key == "fc.2.bias":
            state[key] = state_pre["fc_new.2.bias"]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model_asg.load_state_dict(state, strict=True)
    model_asg = model_asg.to(DEVICE)
    model_asg.eval()
    """
    # CBST + ASG model
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    #resume_path = "runs/cbst_asg_model/"
    resume_path = "runs/cbst_model_gradcam/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    #checkpoint = torch.load(resume_path + "epoch_11checkpoint.pth.tar")
    checkpoint = torch.load(resume_path + "alpha_epoch_12.pth.tar")
    state = model.state_dict()
    for key in state.keys():
        if "model." + key in checkpoint.keys():
            state[key] = checkpoint["model." + key]
        else:
            print("Param {} not loaded".format(key))
            raise ValueError("Param not loaded completely")
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE)
    model.eval()
    """
    # CRST
    model_crst = models.resnet101(pretrained=False)
    num_ftrs = model_crst.fc.in_features
    model_crst.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 12),
    )
    resume_path = "runs/crst/"
    print("=> Loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path + "model_best.pth.tar")
    model_crst.load_state_dict(checkpoint["state_dict"])
    model_crst = model_crst.to(DEVICE)
    model_crst.eval()
    """
    check_name = []
    cbst_pred_wrong = []
    crst_pred_wrong = []
    asg_pred_wrong = []
    drst_pred_correct = []
    with torch.no_grad():
        for i, (input, label, input_name) in enumerate(val_loader):
            label = label.to(DEVICE)
            input = input.to(DEVICE)
            BATCH_SIZE = input.shape[0]
            pred = F.softmax(model_beta(input, None, None, None, None).detach(), dim=1).to(DEVICE)
            r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
            target_out = model_alpha(input, torch.ones((BATCH_SIZE, CONFIG["num_class"])).cuda(),
                                     r_target.cuda()).detach()
            drst_predict = F.softmax(target_out, dim=1)
            cbst_predict = F.softmax(model(input), dim=1)
            #crst_predict = F.softmax(model_crst(input), dim=1)
            #asg_predict = F.softmax(model_asg(input), dim=1)

            drst_predict = torch.argmax(drst_predict, dim=1)
            cbst_predict = torch.argmax(cbst_predict, dim=1)
            #crst_predict = torch.argmax(crst_predict, dim=1)
            #asg_predict = torch.argmax(asg_predict, dim=1)

            drst_predict = drst_predict.detach().cpu().numpy()
            cbst_predict = cbst_predict.detach().cpu().numpy()
            #crst_predict = crst_predict.detach().cpu().numpy()
            #asg_predict = asg_predict.detach().cpu().numpy()

            label = label.cpu().numpy()
            label = label.reshape((-1, ))
            drst_correct = np.where(drst_predict == label)[0]
            cbst_wrong = np.where(cbst_predict != label)[0]
            #crst_wrong = np.where(crst_predict != label)[0]
            #asg_wrong = np.where(asg_predict != label)[0]
            specific_class = [2, 10, 11]
            for idx in drst_correct:
                if idx in cbst_wrong: #and idx in crst_wrong:
                    if cbst_predict[idx] in specific_class and drst_predict[idx] in specific_class:
                    #if idx in asg_wrong:
                        check_name.append(input_name[idx])
                        cbst_pred_wrong.append(cbst_predict[idx])
                        #crst_pred_wrong.append(crst_predict[idx])
                        #asg_pred_wrong.append(asg_predict[idx])
                        drst_pred_correct.append(drst_predict[idx])
            if len(check_name) > 10:
                break

            if i % CONFIG["print_freq"] == 0:
                print("100 samples batches processed")

    print(check_name)
    print("CBST wrong prediction:", cbst_pred_wrong)
    #print("CRST wrong prediction:", crst_pred_wrong)
    #print("ASG wrong prediction:", asg_pred_wrong)
    print("DRST correct prediction: ", drst_pred_correct)
    return check_name, cbst_pred_wrong, crst_pred_wrong, asg_pred_wrong, drst_pred_correct

def save_gif(model, input, save_path, frame):
    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
    target_index = None
    mask = grad_cam(input, target_index)
    show_cam_on_image(img, mask, save_path + str(frame) + "_cam3.jpg")
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask * gb)
    cv2.imwrite(save_path + str(frame) + '_cam_gb3.jpg', cam_gb)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    import os
    if not os.path.exists("log/gradcam/gifs/"):
        os.mkdir("log/gradcam/gifs/")
    image_list, cbst_pred_wrong, crst_pred_wrong, asg_pred_wrong, drst_pred_correct = detect_wrong_predictions()
    for i in range(len(image_list)):
        print("Image {} processed".format(str(i)))
        image_name = image_list[i]
        image_path = "visda/validation/" + image_name
        directory = "log/gradcam/gifs/"+str(i) + "/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        directory_cbst = "log/gradcam/gifs/"+str(i) + "_cbst/"
        if not os.path.exists(directory_cbst):
            os.mkdir(directory_cbst)
        img = cv2.imread(image_path, 1)
        cv2.imwrite(directory + str(i) + "_orig.jpg", img)
        cv2.imwrite(directory_cbst + str(i) + "_orig.jpg", img)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)

        # Get gif frames
        for j in range(20):
            # DRST
            model = models.resnet101(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(512, 12),
            )
            resume_path = "runs/best_model_gradcam/alpha_epoch_"+str(j)+".pth.tar"
            checkpoint = torch.load(resume_path)
            state = model.state_dict()
            for key in state.keys():
                if "model." + key in checkpoint.keys():
                    state[key] = checkpoint["model." + key]
                elif key == "fc.2.weight":
                    state[key] = checkpoint["final_layer.weight"]
                elif key == "fc.2.bias":
                    state[key] = checkpoint["final_layer.bias"]
                else:
                    print("Param {} not loaded".format(key))
                    raise ValueError("Param not loaded completely")
            model.load_state_dict(state, strict=True)
            save_gif(model, input, directory, j)

            # CBST
            model = models.resnet101(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(512, 12),
            )
            resume_path = "runs/cbst_model_gradcam/"
            checkpoint = torch.load(resume_path + "alpha_epoch_"+str(j)+".pth.tar")
            state = model.state_dict()
            for key in state.keys():
                if "model." + key in checkpoint.keys():
                    state[key] = checkpoint["model." + key]
                else:
                    print("Param {} not loaded".format(key))
                    raise ValueError("Param not loaded completely")
            model.load_state_dict(state, strict=True)
            save_gif(model, input, directory_cbst, j)

        """
        ## Load model params
        # ASG model
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 12),
        )
        resume_path = "runs/res101_asg/res101_vista17_best.pth.tar"
        checkpoint = torch.load(resume_path)
        state_pre = checkpoint["state_dict"]
        state = model.state_dict()
        for key in state.keys():
            if key in state_pre.keys():
                state[key] = state_pre[key]
            elif key == "fc.0.weight":
                state[key] = state_pre["fc_new.0.weight"]
            elif key == "fc.0.bias":
                state[key] = state_pre["fc_new.0.bias"]
            elif key == "fc.2.weight":
                state[key] = state_pre["fc_new.2.weight"]
            elif key == "fc.2.bias":
                state[key] = state_pre["fc_new.2.bias"]
            else:
                print("Param {} not loaded".format(key))
                raise ValueError("Param not loaded completely")
        model.load_state_dict(state, strict=True)
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask, directory + str(i) + "_cam0.jpg")
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
        #cv2.imwrite(directory + 'gb0.jpg', gb)
        cv2.imwrite(directory + str(i) + '_cam_gb0.jpg', cam_gb)

        # ASG + CBST model
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 12),
        )
        resume_path = "runs/cbst_asg_model/"
        checkpoint = torch.load(resume_path + "epoch_11checkpoint.pth.tar")
        state = model.state_dict()
        for key in state.keys():
            if "model."+key in checkpoint.keys():
                state[key] = checkpoint["model."+key]
            else:
                print("Param {} not loaded".format(key))
                raise ValueError("Param not loaded completely")
        model.load_state_dict(state, strict=True)
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask, directory + str(i) + "_cam1.jpg")
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
        #cv2.imwrite(directory + 'gb1.jpg', gb)
        cv2.imwrite(directory + str(i) + '_cam_gb1.jpg', cam_gb)

        # ASG + CRST model
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 12),
        )
        resume_path = "runs/crst/"
        checkpoint = torch.load(resume_path + "model_best.pth.tar")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask, directory + str(i) + "_cam2.jpg")
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)
        # cv2.imwrite(directory + 'gb1.jpg', gb)
        cv2.imwrite(directory + str(i) + '_cam_gb2.jpg', cam_gb)

        # Our model
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 12),
        )
        resume_path = "runs/best_model_comp13/alpha_best.pth.tar"
        checkpoint = torch.load(resume_path)
        state = model.state_dict()
        for key in state.keys():
            if "model."+key in checkpoint.keys():
                state[key] = checkpoint["model."+key]
            elif key == "fc.2.weight":
                state[key] = checkpoint["final_layer.weight"]
            elif key == "fc.2.bias":
                state[key] = checkpoint["final_layer.bias"]
            else:
                print("Param {} not loaded".format(key))
                raise ValueError("Param not loaded completely")
        model.load_state_dict(state, strict=True)
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask, directory + str(i) + "_cam3.jpg")
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        #cv2.imwrite(directory + 'gb2.jpg', gb)
        cv2.imwrite(directory + str(i) + '_cam_gb3.jpg', cam_gb)
        """
    """
    saved 
    index = 4, others: motorcycle, ours: car
    index = 1, cbst, asg:bus, crst: car, ours: truck
    index=5, others: train, ours: truck
    index=6, others: horse, ours: car (intuitive: the image is a giraffe, but no such class)
    index=17, others: horse, ours:bicycle
    index=27, others: motorcycle, ours:skateboard 
    """