
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# This code is adapted from
# https://github.com/pd90506/AGI/blob/master/AGI_main.py

import json

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def pre_processing(obs, torch_device):
    # rescale imagenet, we do mornalization in the network, instead of preprocessing
    obs = obs / 255
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device)
    return obs_tensor

def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    grad_lab_norm = torch.norm(data_grad_lab,p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta
    # return perturbed_image, delta

def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    c_delta = 0 # cumulative delta
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # if attack is successful, then break
        if pred.item() == targeted.item():
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        loss = output[0,targeted.item()]

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        loss_lab = output[0, init_pred.item()]
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(image, epsilon, data_grad_adv, data_grad_lab)
        c_delta += delta
    
    return c_delta, perturbed_image


def test(model, device, data, epsilon, topk, selected_ids, max_iter):
    # Send the data and label to the device
    data = pre_processing(data, device)
    data = data.to(device)

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    top_ids = selected_ids # only for predefined ids
    # initialize the step_grad towards all target false classes
    step_grad = 0 
    # num_class = 1000 # number of total classes
    for l in top_ids:
        targeted = torch.tensor([l]).to(device) 
        if targeted.item() == init_pred.item():
            continue # we don't want to attack to the predicted class.

        delta, perturbed_image = pgd_step(data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    if (torch.is_tensor(step_grad)):
        adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk
    else:
        return 0, 0, 0

    adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk
    img = data.squeeze().detach().cpu().numpy()
    # perturbed_image = perturbed_image.squeeze().detach().cpu().numpy()
    example = (init_pred.item(), img, adv_ex)

    # Return prediction, original image, and heatmap
    return example

# set lowerbound and upperbound for figure
percentile = 80
upperbound = 99
# input
def plot_img(plt, example, class_names):
    pred, img, ex = example
    plt.title("Pred:{}".format(class_names[pred]))
    ex = np.transpose(img, (1,2,0))
    print(ex.shape)
    plt.imshow(ex)

# heatmap
def plot_hm(plt, example):
    pred, img, ex = example
    # plt.title("Pred: {}".format(pred))
    plt.title("Heatmap")
    ex = np.mean(ex, axis=0)
    q = np.percentile(ex, percentile)
    u = np.percentile(ex, upperbound)
    # q=0
    ex[ex<q] = q
    ex[ex>u] = u
    ex = (ex-q)/(u-q)
    plt.imshow(ex, cmap='gray')
    return ex

# input * heatmap
def plot_hm_img(plt, example):
    pred, img, ex = example
    plt.title("Input * heatmap")
    ex = np.expand_dims(np.mean(ex, axis=0), axis=0)
    q = np.percentile(ex, percentile)
    u = np.percentile(ex, upperbound)
    # q=0
    ex[ex<q] = q
    ex[ex>u] = u
    ex = (ex-q)/(u-q)
    ex = np.transpose(ex, (1,2,0))
    img = np.transpose(img, (1,2,0))

    img = img * ex
    plt.imshow(img)

# Boilerplate methods.
def LoadImage(file_path, resize, crop):
    im = Image.open(file_path)
    im = resize(im)
    im = crop(im)
    im = np.asarray(im)
    return im