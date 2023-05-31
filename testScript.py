import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt

import os
import csv

import numpy as np

import XRAITestFunctions as XRAI
import RISETestFunctions as RISE

from captum.attr import IntegratedGradients
import AttributionMethods as attribution
import guidedIGBuilder as GIG_Builder
import AGI as AGI

import argparse

model = None

normalize = transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
)

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

def save_aggregate_PIC_curve(test_results, model_name, img_label, image_count, method):
    fig, ax = plt.subplots(figsize=(12, 6))

    if method == 0:
        title = "PIC - Aggregated SIC Over " + str(image_count) +  " Images"
        folder = "tests/" + model_name + "/SIC_ins/"
        type = "SIC_ins"
        y_label = "Average Classificaton Confidence"
    else:
        title = "PIC - Aggregated AIC Over " + str(image_count) +  " Images"
        folder = "tests/" + model_name + "/AIC_ins/"
        type = "AIC_ins"
        y_label = "Average Accuracy"

    # make the test folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_label = img_label + type

    # create/overwrite a csv
    file = open(folder + img_label + ".csv", "w")
    writer = csv.writer(file)

    aggregate = XRAI.aggregate_individual_pic_results(test_results[0], "mean")
    XRAI.show_curve_xy(aggregate.curve_x, aggregate.curve_y, title = title, label = '' + test_results[1] + " mean " + type, color = "purple", ax = ax)
    ax.set_ylabel(y_label) 

    # save the plot image
    fig.savefig(folder + img_label + ".jpg")

    # save the plot data to csv
    writer.writerow([test_results[1] + " mean " + type])
    writer.writerow(["x_vals"] + list(map(str, aggregate.curve_x.tolist())))
    writer.writerow(["y_vals"] + list(map(str, aggregate.curve_y.tolist())))
    writer.writerow(["AUC", str(aggregate.auc)])

    file.close()

    return fig, aggregate

def save_RISE_curve(test_results, model_name, img_label, image_count, method):
    fig, _ = plt.subplots(figsize = (12, 6))

    if method == 0:
        title = "Aggregated Deletion Curve Over " + str(image_count) +  " Images"
        folder = "tests/" + model_name + "/RISE_del/"
        x_label = "Pixels Deleted"
        type = "deletion"
        img_label = img_label + "RISE_del"
    else:
        title = "Aggregated Insertion Curve Over " + str(image_count) +  " Images"
        folder = "tests/" + model_name + "/RISE_ins/"
        x_label = "Pixels Inserted"
        type = "insertion"
        img_label = img_label + "RISE_ins"

    # make the test folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # create/overwrite a csv
    file = open(folder + img_label + ".csv", "w")
    writer = csv.writer(file)

    steps = test_results[0]
    y_vals = test_results[1]
    auc = test_results[2]
    test_name = test_results[3]

    x_vals = np.arange(steps + 1)

    plt.plot(x_vals, y_vals, label = '' + test_name + " mean " + type + ", AUC = " + '{0:.3g}'.format(auc), color = "purple")
    plt.fill_between(x_vals, 0, y_vals, color = "purple", alpha = 0.4)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Average Classificaton Confidence")
    plt.legend(loc="upper left")

    # save the plot image
    fig.savefig(folder + img_label + ".jpg")

    # save the plot data to csv
    writer.writerow([test_name + " mean " + type])
    writer.writerow(["x_vals"] + list(map(str, x_vals.tolist())))
    writer.writerow(["y_vals"] + list(map(str, y_vals.tolist())))
    writer.writerow(["AUC", str(auc)])

    file.close()

    return fig

# runs an attribution method w 3 baselines over imageCount images and calculates the mean PIC
def run_and_save_tests(img_hw, random_mask, saliency_thresholds, transform_list, image_count, function, function_steps, batch_size, model, model_name, deletion, insertion, device, imagenet):
    # attr_func_steps
    if function == "AGI":
        img_label = function + "_"
    else:
        img_label = function + "_" + str(function_steps) + "_steps_"

    if function == "AGI":
        # set up new AGI model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm_layer = AGI.Normalize(mean, std)
        modified_model = nn.Sequential(norm_layer, model).to(device)

    # num imgs used for testing
    img_label = img_label + str(image_count) + "_images_"

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("supplementaryCode/class_maps/correctly_classified_" + model_name + ".txt").astype(np.int64)

    # This holds the run data for a set of images
    # used to aggreagate runs after all images are done
    sic_runs = []
    aic_runs = []

    images_used = 0

    images_per_class = int(np.ceil(image_count / 1000))
    classes_used = [0] * 1000

    # look at test images in order from 1
    for image in sorted(os.listdir(imagenet)):    
        if images_used == image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        img = Image.open(imagenet + "/" + image)
        img = transform_list[0](img)

        # put the image in form needed for prediction for the ins/del method
        img_tensor = transform_normalize(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        # only rgb images can be classified
        if img.shape != (3, img_hw, img_hw):
            continue

        target_class, _ = attribution.getClass(img, model, device)

        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        print(model_name + " Function " + function + ", image: " + image)

        ########  IG  ########
        if (function == "IG"):
            integrated_gradients = IntegratedGradients(model)
            saliency_map = integrated_gradients.attribute(img_tensor.to(device), 0, target = target_class, n_steps = function_steps, internal_batch_size=batch_size)
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))

            # image for AIC/SIC test
            img_test = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))
        ########  LIG  ########
        elif (function == "LIG"):
            saliency_map, _, _ = attribution.IGParallel(img, model, function_steps, batch_size, .9, 0, device, target_class)
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))

            # image for AIC/SIC test
            img_test = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))
        ########  IDG  ########
        elif (function == "IDG"):
            saliency_map, _, _, _ = attribution.IDG(img, model, function_steps, batch_size, 0, device, target_class)
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))

            # image for AIC/SIC test
            img_test = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))
        ########  GIG  ########
        elif (function == "GIG"):
            # Load the image for GIG
            im_orig = LoadImage(imagenet + "/" + image, transform_list[1], transform_list[2])
            img_tensor = PreprocessImages([im_orig])
            
            # get the class
            prediction_class, _ = getClassGIG(img_tensor, model, device)
            prediction_class = prediction_class.item()
            call_model_args = {class_idx_str: prediction_class}

            # make sal map
            im = im_orig.astype(np.float32)
            guided_ig = GIG_Builder.GuidedIG()
            baseline = np.zeros(im.shape)
            saliency_map = guided_ig.GetMask(im, model, device, call_model_function, call_model_args, x_baseline=baseline, x_steps=function_steps, max_dist=1.0, fraction=0.5)

            # image for AIC/SIC test
            img_test = im / 255
        ########  AGI  ########
        elif (function == "AGI"):
            epsilon = 0.05
            max_iter = 20
            topk = 1
            # define the ids of the selected adversarial class
            selected_ids = range(0, 999, int(1000 / topk)) 

            agi_img = AGI.LoadImage(imagenet + "/" + image, transform_list[1], transform_list[2])
            agi_img = agi_img.astype(np.float32) 

            # Run test
            example = AGI.test(modified_model, device, agi_img, epsilon, topk, selected_ids, max_iter)
            AGI_map = example[2]

            if type(AGI_map) is not np.ndarray:
                print("AGI failure, skipping image")
                classes_used[target_class] -= 1
                continue

            saliency_map = np.transpose(AGI_map, (1, 2, 0))

            # image for AIC/SIC test
            img_test = np.transpose(img.squeeze().detach().numpy(), (1, 2, 0))

            img_tensor = torch.tensor(np.transpose(agi_img / 255, (2, 0, 1)))
            img_tensor = torch.unsqueeze(img_tensor, 0)
            img_tensor = transform_normalize(img_tensor)

        # use abs val of attribution map pixels for testing
        saliency_map = np.abs(np.sum(saliency_map, axis = 2))

        # SIC ins computation
        print("SIC ins in progress")
        sic_score = XRAI.compute_pic_metric(img_test, saliency_map, random_mask, saliency_thresholds, 0, model, device)

        # AIC ins computation
        print("AIC ins in progress")
        aic_score = XRAI.compute_pic_metric(img_test, saliency_map, random_mask, saliency_thresholds, 1, model, device)
        
        # if the current image didn't fail the PIC tests use its result
        if sic_score != 0 and aic_score != 0:
            sic_runs.append(sic_score)
            aic_runs.append(aic_score)
        # if it did fail, skip to next loop so that ins and del tests are not used as well
        else:
            print("image: " + image + " thrown out due to 0 score")
            classes_used[target_class] -= 1
            continue

        # ins and del computation
        print("Insertion and Deletion in progress")
        
        if images_used == 0:
            n_steps, _, del_sum = deletion.single_run(img_tensor, saliency_map, 0, device)
            
            _, _, ins_sum = insertion.single_run(img_tensor, saliency_map, 0, device)
        else:
            _, _, del_sum_temp = deletion.single_run(img_tensor, saliency_map, 0, device)
            del_sum += del_sum_temp

            _, _, ins_sum_temp = insertion.single_run(img_tensor, saliency_map, 0, device)
            ins_sum += ins_sum_temp

        # when all tests have passed the number of images used can go up by 1
        images_used += 1

        print("Total used: " + str(images_used) + " / " + str(image_count))

    # average the curves across the images
    del_sum = del_sum / (images_used + 1)
    ins_sum = ins_sum / (images_used + 1)

    # get area under the curve
    del_auc = RISE.auc(del_sum)
    ins_auc = RISE.auc(ins_sum)

    # add all 4 tests to arrays so their data can be saved
    del_data = (n_steps, del_sum, del_auc, function)
    ins_data = (n_steps, ins_sum, ins_auc, function)

    # save the ins/del plots as images, and the data to text files
    save_RISE_curve(del_data, model_name, img_label, images_used, 0)
    save_RISE_curve(ins_data, model_name, img_label, images_used, 1)

    sic_data = (sic_runs, function)
    aic_data = (aic_runs, function)

    # save the insertion PIC plots as images, and the data to text files
    save_aggregate_PIC_curve(sic_data, model_name, img_label, images_used, 0)
    save_aggregate_PIC_curve(aic_data, model_name, img_label, images_used, 1)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # img_hw determines how to transform innput images for model needs
    if FLAGS.model == "R101":
        model = models.resnet101(weights = "ResNet101_Weights.IMAGENET1K_V2")
        img_hw = 224
        batch_size = 50
    elif FLAGS.model == "R152":
        model = models.resnet152(weights = "ResNet152_Weights.IMAGENET1K_V2")
        img_hw = 224
        batch_size = 25
    elif FLAGS.model == "RESNXT":
        model = models.resnext101_64x4d(weights = "ResNeXt101_64X4D_Weights.IMAGENET1K_V1")
        img_hw = 224
        batch_size = 25

    function_steps = 50

    model = model.eval()
    model.to(device)

    # specify the transforms needed
    resize = transforms.Resize((img_hw, img_hw))
    crop = transforms.CenterCrop(img_hw)

    transform_IG = transforms.Compose([
        transforms.Resize((img_hw, img_hw)),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    transform_list = (transform_IG, resize, crop)

    # initialize XRAI blur kernel
    random_mask = XRAI.generate_random_mask(img_hw, img_hw, .01)
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    # initialize RISE blur kernel
    klen = 11
    ksig = 5
    kern = RISE.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)

    # RISE ins and del test clases
    insertion = RISE.CausalMetric(model, img_hw * img_hw, 'ins', img_hw, substrate_fn = blur)
    deletion = RISE.CausalMetric(model, img_hw * img_hw, 'del', img_hw, substrate_fn = torch.zeros_like)

    run_and_save_tests(img_hw, random_mask, saliency_thresholds, transform_list, FLAGS.image_count, FLAGS.function, function_steps, batch_size, model, FLAGS.model, deletion, insertion, device, FLAGS.imagenet)

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
def getClassGIG(input, model, device):
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
    parser = argparse.ArgumentParser('Attribution Test Script.')
    parser.add_argument('--function',
                        type = str, default = "IG",
                        help = 'Name of the attribution method to use: IG, LIG, GIG, AGI, or IDG.')
    parser.add_argument('--image_count',
                        type = int, default = 5000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101, R152. RNXT')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                type = str, default = "imagenet",
                help = 'The path to your 2012 imagenet vlaidation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)