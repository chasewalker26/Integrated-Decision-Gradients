import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import csv
import argparse
import numpy as np
from PIL import Image
from captum.attr import IntegratedGradients
import os
os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.test_methods import PICTestFunctions as PIC
from util.test_methods import RISETestFunctions as RISE
from util.attribution_methods import saliencyMethods as attribution
from util.attribution_methods import GIGBuilder as GIG_Builder
from util.attribution_methods import AGI as AGI

model = None

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# runs an attribution method w 3 baselines over imageCount images and calculates the mean PIC
def run_and_save_tests(img_hw, random_mask, saliency_thresholds, transform_list, image_count, function, function_steps, batch_size, model, model_name, deletion, insertion, device, image_path):
    # attr_func_steps
    if function == "AGI":
        img_label = function + "_"
    else:
        img_label = function + "_" + str(function_steps) + "_steps_"

    # num imgs used for testing
    img_label = img_label + str(image_count) + "_images_"

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    num_classes = 1000
    images_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    fields = ["attr", "SIC", "AIC", "Ins", "Del"]
    scores = [function, 0, 0, 0, 0]

    images = sorted(os.listdir(image_path))
    images_used = 0

    # look at test images in order from 1
    for image in images:    
        if images_used == image_count:
            print("method finished")
            break

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        img = Image.open(image_path + "/" + image)
        trans_img = transform_list[0](img)

        # put the image in form needed for prediction for the ins/del method
        img_tensor = transform_normalize(trans_img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(img_tensor, model, device)

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
        ########  LIG  ########
        elif (function == "LIG"):
            saliency_map = attribution.IG(img_tensor, model, function_steps, batch_size, .9, 0, device, target_class)
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
        ########  IDG  ########
        elif (function == "IDG"):
            saliency_map = attribution.IDG(img_tensor, model, function_steps, batch_size, 0, device, target_class)
            saliency_map = np.transpose(saliency_map.squeeze().detach().cpu().numpy(), (1, 2, 0))
        ########  GIG  ########
        elif (function == "GIG"):
            call_model_args = {'class_idx_str': target_class.item()}
            guided_ig = GIG_Builder.GuidedIG()
            baseline = torch.zeros_like(img_tensor)
            gig = guided_ig.GetMask(img_tensor, model, device, GIG_Builder.call_model_function, call_model_args, x_baseline=baseline, x_steps=50, max_dist=1.0, fraction=0.5)
            saliency_map = np.transpose(gig.squeeze().detach().cpu().numpy(), (1, 2, 0))
        ########  AGI  ########
        elif (function == "AGI"):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            norm_layer = AGI.Normalize(mean, std)
            modified_model = nn.Sequential(norm_layer, model).to(device)
            epsilon = 0.05
            max_iter = 20
            topk = 1
            selected_ids = range(0, 999, int(1000 / topk)) 
            agi_img = AGI.LoadImage(image_path + "/" + image, transform_list[1], transform_list[2])
            agi_img = agi_img.astype(np.float32) 
            example = AGI.test(modified_model, device, agi_img, epsilon, topk, selected_ids, max_iter)
            AGI_map = example[2]
            if type(AGI_map) is not np.ndarray:
                print("AGI failure, skipping image")
                classes_used[target_class] -= 1
                continue
            saliency_map = np.transpose(AGI_map, (1, 2, 0))

        # use abs val of attribution map pixels for testing
        saliency_map = np.abs(np.sum(saliency_map, axis = 2))

        # make sure attribution is valid
        if np.sum(saliency_map.reshape(1, 1, img_hw ** 2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue

        # Get attribution scores
        ins_del_img = img_tensor
        PIC_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))

        sic_score = PIC.compute_pic_metric(PIC_img, saliency_map, random_mask, saliency_thresholds, 0, model, device, transform_list[2])
        aic_score = PIC.compute_pic_metric(PIC_img, saliency_map, random_mask, saliency_thresholds, 1, model, device, transform_list[2])
        
        # if the current image didn't fail the PIC tests use its result
        if sic_score == 0 or aic_score == 0:
            print("image: " + image + " thrown out due to 0 score")
            classes_used[target_class] -= 1
            continue

        # capture PIC scores
        scores[1] += sic_score.auc
        scores[2] += aic_score.auc

        # ins and del computation
        _, ins_sum = insertion.single_run(ins_del_img, saliency_map, device, batch_size)
        _, del_sum = deletion.single_run(ins_del_img, saliency_map, device, batch_size)
        scores[3] += RISE.auc(ins_sum)
        scores[4] += RISE.auc(del_sum)

        # when all tests have passed the number of images used can go up by 1
        images_used += 1

        print("Total used: " + str(images_used) + " / " + str(image_count))

    for i in range(1, len(scores)):
        scores[i] /= images_used
        scores[i] = round(scores[i], 3)

    # make the test folder if it doesn't exist
    folder = "../test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_label = function + "_" + str(image_count) + "_images"
    with open(folder + img_label + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerow(scores)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    # img_hw determines how to transform input images for model needs
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

    # initialize PIC blur kernel
    random_mask = PIC.generate_random_mask(img_hw, img_hw, .01)
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


if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Attribution Test Script.')
    parser.add_argument('--function',
                        type = str, default = "IG",
                        help = 'Name of the attribution method to test: IG, LIG, GIG, AGI, IDG.')
    parser.add_argument('--image_count',
                        type = int, default = 5000,
                        help='How many images to test with.')
    parser.add_argument('--model',
                        type = str,
                        default = "R101",
                        help='Classifier to use: R101, R152, or RESNXT')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                type = str, default = "ImageNet",
                help = 'The relative path to your 2012 imagenet validation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)