from fpdf import FPDF
import os, sys, csv

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import AttributionMethods as attribution
import guidedIGBuilder as GIG_Builder
import AGI as AGI

from captum.attr import visualization as viz

import argparse

from matplotlib.colors import LinearSegmentedColormap

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
)

# invert standard ImageNet normalization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

default_cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, '#ffffff'), (0.25, '#0000ff'), (1, '#0000ff')], N = 256)   

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    if FLAGS.model == "R101":
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
    elif FLAGS.model == "R152":
        model = models.resnet152(weights = "ResNet152_Weights.IMAGENET1K_V2")
    elif FLAGS.model == "RESNXT":
        model = models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1")

    # img_hw determines how to transform input images for model needs
    img_hw = 224

    model = model.eval()
    model.to(device)

    # transform data into format needed for model
    transform = transforms.Compose([
        transforms.Resize((img_hw, img_hw)),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    # specify the transforms needed
    resize = transforms.Resize((img_hw, img_hw))
    crop = transforms.CenterCrop(img_hw)

    plt.rcParams['figure.dpi'] = 75
    plt.rcParams['savefig.dpi'] = 75
    plt.ioff()

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("supplementaryCode/class_maps/correctly_classified_R101.txt").astype(np.int64)

    images_used = 0

    pdf = FPDF(format = "letter", unit="in")
    row = 0
    pdf.add_page()

    attributions = torch.zeros(4, 3, img_hw, img_hw)

    # look at imagenet images in order from 1
    for image in sorted(os.listdir(FLAGS.imagenet)):
        if images_used == FLAGS.image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        if correctly_classified[image_num] == 0:
            continue

        img = Image.open(FLAGS.imagenet + "/" + image)
        img = transform(img)

        # only rgb images can be classified
        if img.shape != (3, img_hw, img_hw):
            continue

        # put the image in form needed for prediction for the ins/del method
        img_tensor = transform_normalize(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        # IG initialization
        target_class, class_name = attribution.getClass(img, model, device)   
        percentage, _ = attribution.getPrediction(img, model, device, target_class) 
        percentage = percentage * 100

        if len(class_name) > 13:
            continue

        if (class_name == "maillot 1" or class_name == "maillot 2" or class_name == "swimming trunks"):
            continue

        # Guided IG initialization
        gig_img = LoadImage(FLAGS.imagenet + "/" + image, resize, crop)
        gig_img = gig_img.astype(np.float32)
        call_model_args = {class_idx_str: target_class.item()}
        guided_ig = GIG_Builder.GuidedIG()
        baseline = np.zeros(gig_img.shape)

        # AGI initialization
        epsilon = 0.05
        topk = 1
        max_iter = 20
        agi_img = AGI.LoadImage(FLAGS.imagenet + "/" + image, resize, crop)
        agi_img = agi_img.astype(np.float32) 
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm_layer = AGI.Normalize(mean, std)
        modified_model = nn.Sequential(norm_layer, model).to(device)


        ####### run methods #######
        print("Generating Attributions for image: " + image)

        print("working on ig 50")
        attributions[0], _, _ = attribution.IGParallel(img, model, 50, 25, 1, 0, device, target_class)
        torch.cuda.empty_cache()

        print("working on lig 50")
        attributions[1], _, _ = attribution.IGParallel(img, model, 50, 25, .9, 0, device, target_class)
        torch.cuda.empty_cache()

        print("Working on IDG 50")
        attributions[3], _, _, _ = attribution.IDG(img, model, 50, 25, 0, device, target_class)
        torch.cuda.empty_cache()

        print("Working on gig 50")
        gig = guided_ig.GetMask(gig_img, model, device, call_model_function, call_model_args, x_baseline=baseline, x_steps=50, max_dist=1.0, fraction=0.5)
        torch.cuda.empty_cache()

        print("Working on agi")
        selected_ids = range(0, 999, int(1000 / topk)) # define the ids of the selected adversarial class
        _, _, agi_1 = AGI.test(modified_model, device, agi_img, epsilon, topk, selected_ids, max_iter)
        torch.cuda.empty_cache()

        ####### save images and add to pdf #######
        print("Total used: " + str(images_used + 1) + "/" + str(FLAGS.image_count))

        # add page to PDF when a new row is needed
        if (row == 10):
            pdf.add_page()
            row = 0

        # if on first row of a page, print the column titles
        if row == 0:
            titles = ["Input", "IG", "LIG", "GIG", "AGI", "AGI", "IDG"]
        else:
            titles = ["", "", "", "", "", "", ""]

        # make the temp image folder if it doesn't exist
        if not os.path.exists("temp_folder_attr"):
            os.makedirs("temp_folder_attr")

        ####### save original image #######
        fig, ax =  plt.subplots(1, 1)
        img_np = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))

        ax.set_ylabel(class_name, size = 30)
        ax.set_title(titles[0], size = 30)
        fig, ax = viz.visualize_image_attr(None, 
                            img_np, 
                            method="original_image",
                            plt_fig_axis = (fig, ax),
                            use_pyplot = False)
        plt.figure(fig)
        fig.savefig("temp_folder_attr/image" + str(images_used) + ".jpg", bbox_inches='tight')
        fig.clear()
        plt.close(fig)

        ####### save IG 50 #######
        grads = np.transpose(attributions[0].squeeze().detach().cpu().numpy(), (1,2,0))
        fig, ax =  plt.subplots(1, 1)
        ax.set_title(titles[1], size = 30)
        fig, ax = viz.visualize_image_attr(grads,
                            img_np,
                            method = 'heat_map',
                            plt_fig_axis = (fig, ax),
                            cmap = default_cmap,
                            sign = "absolute_value",
                            use_pyplot = False)
        plt.figure(fig)
        fig.savefig("temp_folder_attr/ig" + str(images_used) + ".jpg", bbox_inches='tight')
        fig.clear()
        plt.close(fig)

        ####### save LIG 50 #######
        grads = np.transpose(attributions[1].squeeze().detach().cpu().numpy(), (1,2,0))
        fig, ax =  plt.subplots(1, 1)
        ax.set_title(titles[2], size = 30)
        fig, ax = viz.visualize_image_attr(grads,
                            img_np,
                            method = 'heat_map',
                            plt_fig_axis = (fig, ax),
                            cmap = default_cmap,
                            sign = "absolute_value",
                            use_pyplot = False)
        plt.figure(fig)
        fig.savefig("temp_folder_attr/lig" + str(images_used) + ".jpg", bbox_inches='tight')
        fig.clear()
        plt.close(fig)


        ####### save GIG 50 #######
        grads = gig
        fig, ax =  plt.subplots(1, 1)
        ax.set_title(titles[3], size = 30)
        fig, ax = viz.visualize_image_attr(grads,
                            img_np,
                            method = 'heat_map',
                            plt_fig_axis = (fig, ax),
                            cmap = default_cmap,
                            sign = "absolute_value",
                            use_pyplot = False)
        plt.figure(fig)
        plt.savefig("temp_folder_attr/gig" + str(images_used) + ".jpg", bbox_inches='tight', transparent = "True", pad_inches = .1)
        fig.clear()
        plt.close(fig)

        ####### save AGI #######
        percentile = 80
        upperbound = 99
        hm = agi_1
        hm = np.mean(hm, axis=0)
        q = np.percentile(hm, percentile)
        u = np.percentile(hm, upperbound)
        # q=0
        hm[hm<q] = q
        hm[hm>u] = u
        hm = (hm-q)/(u-q)
        grads = np.reshape(hm, (img_hw, img_hw, 1))

        fig, ax =  plt.subplots(1, 1)
        ax.set_title(titles[5], size = 30)
        fig, ax = viz.visualize_image_attr(grads,
                            img_np,
                            method = 'heat_map',
                            plt_fig_axis = (fig, ax),
                            cmap = default_cmap,
                            sign = "absolute_value",
                            use_pyplot = False)
        plt.figure(fig)
        plt.savefig("temp_folder_attr/agi" + str(images_used) + ".jpg", bbox_inches='tight', transparent = "True", pad_inches = .1)
        fig.clear()
        plt.close(fig)

        ####### save IDG 50 #######
        grads = np.transpose(attributions[3].squeeze().detach().cpu().numpy(), (1,2,0))
        fig, ax =  plt.subplots(1, 1)
        ax.set_title(titles[6], size = 30)
        fig, ax = viz.visualize_image_attr(grads,
                            img_np,
                            method = 'heat_map',
                            plt_fig_axis = (fig, ax),
                            cmap = default_cmap,
                            sign = "absolute_value",
                            use_pyplot = False)
        plt.figure(fig)
        plt.savefig("temp_folder_attr/idg" + str(images_used) + ".jpg", bbox_inches='tight', transparent = "True", pad_inches = .1)
        fig.clear()
        plt.close(fig)

        # print pages with 1 x 1.5 x 1.5 x ~>1 margins (TxLxRxB)
        if row == 0:
            y = 1
            pdf.image("temp_folder_attr/image" +   str(images_used) + ".jpg", 1.500, y, .993, .993)
            pdf.image("temp_folder_attr/ig" +      str(images_used) + ".jpg", 2.493, y, .889, .993)
            pdf.image("temp_folder_attr/lig" +     str(images_used) + ".jpg", 3.382, y, .889, .993)
            pdf.image("temp_folder_attr/gig" +     str(images_used) + ".jpg", 4.271, y, .889, .993)
            pdf.image("temp_folder_attr/agi" +     str(images_used) + ".jpg", 5.160, y, .889, .993)
            pdf.image("temp_folder_attr/idg" +     str(images_used) + ".jpg", 6.049, y, .889, .993)
            y = 1.993
        else:
            pdf.image("temp_folder_attr/image" +   str(images_used) + ".jpg", 1.500, y, .993, .889)
            pdf.image("temp_folder_attr/ig" +      str(images_used) + ".jpg", 2.493, y, .889, .889)
            pdf.image("temp_folder_attr/lig" +     str(images_used) + ".jpg", 3.382, y, .889, .889)
            pdf.image("temp_folder_attr/gig" +     str(images_used) + ".jpg", 4.271, y, .889, .889)
            pdf.image("temp_folder_attr/agi" +     str(images_used) + ".jpg", 5.160, y, .889, .889)
            pdf.image("temp_folder_attr/idg" +     str(images_used) + ".jpg", 6.049, y, .889, .889)
            y += 0.889

        images_used += 1
        row += 1

    pdf.output(FLAGS.file_name + ".pdf", "F")
    pdf.close()

    # clear the folder that held the images
    dir = 'temp_folder_attr'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    return

# Boilerplate methods.
def LoadImage(file_path, resize, crop):
    im = Image.open(file_path)
    im = resize(im)
    im = crop(im)
    im = np.asarray(im)
    return im

def PreprocessImages(images):
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = normalize.forward(images)
    return images.requires_grad_(True)

# returns the class of an image 
def getClass(input, model, device):
    # calculate a prediction
    input = input.to(device)
    output = model(input)

    _, index = torch.max(output, 1)

    # open the class list so the detected class string can be returned for printing
    with open('supplementaryCode/class_maps/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    return index[0], classes[index[0]]

class_idx_str = 'class_idx_str'
def call_model_function(images, model, device, call_model_args = None, expected_keys = None):
    images = PreprocessImages(images)
    target_class_idx = call_model_args[class_idx_str]
    output = model(images.to(device))
    
    # capture the logit before the output is crushed by softmax
    logit = ((output[0])[target_class_idx]).detach().cpu().numpy().item()

    m = torch.nn.Softmax(dim=1)
    output = m(output)

    if GIG_Builder.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        grads = torch.movedim(grads, 1, 3)
        gradients = grads.detach().numpy()
        
        return {GIG_Builder.INPUT_OUTPUT_GRADIENTS: gradients}, logit

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Attribution Generation Script.')
    parser.add_argument('--image_count',
                        type = int, default = 50,
                        help='How many attributions to make.')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101, R152. RNXT')
    parser.add_argument('--cuda_num',
                        type = int, default = 0,
                        help = 'The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                    type = str, default = "imagenet",
                    help = 'The path to your 2012 imagenet vlaidation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    parser.add_argument('--file_name',
                    type = str, default = "IDG_supplementary",
                    help = 'The file name for the resulting pdf.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)