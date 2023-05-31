import torch
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt
from captum.attr import visualization as viz

from matplotlib.colors import LinearSegmentedColormap
default_cmap = LinearSegmentedColormap.from_list('custom blue',  [(0, '#ffffff'), (0.25, '#0000ff'), (1, '#0000ff')], N = 256)   

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

# returns the softmax classification value of an image for the highest predicted class or a target class
def getPrediction(input, model, device, target_class):
    input = transform_normalize(input)
    input = torch.unsqueeze(input, 0)

    # calculate a prediction
    input = input.to(device)
    output = model(input)

    if target_class == -1:
        _, index = torch.max(output, 1)
        percentage = ((torch.nn.functional.softmax(output, dim = 1)[0])[index[0]]).detach().cpu().numpy()
        logit = ((output[0])[index[0]]).detach().cpu().numpy()
        return percentage, logit
    else:
        percentage = ((torch.nn.functional.softmax(output, dim = 1)[0])[target_class]).detach().cpu().numpy()
        logit = ((output[0])[target_class]).detach().cpu().numpy()
        return percentage, logit
        
# returns the class of an image 
def getClass(input, model, device):
    input = transform_normalize(input)
    input = torch.unsqueeze(input, 0)

    # calculate a prediction
    input = input.to(device)
    output = model(input)

    _, index = torch.max(output, 1)

    # open the class list so the detected class string can be returned for printing
    with open('supplementaryCode/class_maps/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    return index[0], classes[index[0]]

# returns the gradients from the model for an input
def getGradientsParallel(images, baseline, alphas, model, device, target_class):
    images.requires_grad = True

    images, baseline, alphas = images.to(device), baseline.to(device), alphas.to(device)

    baseline_diff = torch.sub(images, baseline)

    inputs = torch.add(baseline, torch.mul(alphas, baseline_diff))

    # calculate logit derivative
    output = model(inputs)

    scores = output[:, target_class]

    gradients = torch.autograd.grad(scores, inputs, grad_outputs = torch.ones_like(scores))[0]

    return gradients, scores

def IGParallel(trans_img, model, steps, batch_size, alpha_star, baseline, device, target_class):
    if (steps % batch_size != 0):
        print("steps must be evenly divisible by batch size: " + str(batch_size) + "!")
        return 0, 0, 0, 0

    loops = int(steps / batch_size)

    # generate alpha values as 4D
    alphas = torch.linspace(0, 1, steps, requires_grad = True)
    alphas = alphas.reshape(steps, 1, 1, 1)

    # array to store the gradient at each step
    gradients = torch.zeros((steps, trans_img.shape[0], trans_img.shape[1], trans_img.shape[2])).to(device)
    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)

    # batch the input image
    unmodified_input_img = transform_normalize(trans_img)
    unmodified_input_img = torch.unsqueeze(unmodified_input_img, 0)

    images = unmodified_input_img.repeat(steps, 1, 1, 1)

    # create baseline and get difference from input image
    baseline = torch.full(images.shape, baseline, dtype = torch.float)
    
    # run batched input
    for i in range(loops):
        torch.cuda.empty_cache()
        start = i * batch_size
        end = (i + 1) * batch_size

        gradients[start : end], logits[start : end] = getGradientsParallel(images[start : end], baseline[start : end], alphas[start : end], model, device, target_class)
        torch.cuda.empty_cache()

    max_perc = torch.max(logits)
    cutoff_perc = max_perc * alpha_star

    # IG: sum all the gradients
    if alpha_star == 1:
        grads = gradients.mean(dim = 0)
    # LeftIG: sum the gradients up to the cutoff point and no later
    else:
        # find where cutoff point is
        cutoff_step = torch.where(logits > cutoff_perc)[0][0]
        grads = (gradients[0:cutoff_step]).mean(dim = 0)

    # multiply sum by (original image - baseline)
    baseline_diff = torch.sub(unmodified_input_img, baseline[0]).to(device)
    grads = torch.multiply(grads, baseline_diff.to(device))

    return grads, logits.detach().cpu().numpy(), alphas.detach().cpu().numpy()


# returns the weighted gradients from the model for batched input
def getWeightedGradientsParallel(images, baseline, alphas, model, device, target_class):
    images.requires_grad = True

    images, baseline, alphas = images.to(device), baseline.to(device), alphas.to(device)

    baseline_diff = torch.sub(images, baseline)

    inputs = torch.add(baseline, torch.mul(alphas, baseline_diff))

    # calculate logit derivative
    output = model(torch.cat((torch.add(baseline, torch.mul(alphas, baseline_diff)), inputs)))

    scores = output[:, target_class]

    # scores is the length of 2x the batch size, its first and last half are 
    # duplicated, so we only return the first half
    half = int(scores.shape[0] / 2)
    end = scores.shape[0]
    
    gradients = torch.autograd.grad(scores[half : end + 1], inputs, grad_outputs = torch.ones_like(scores[half : end + 1]), retain_graph = True)[0]

    slopes = torch.autograd.grad(scores[half : end + 1], alphas, grad_outputs = torch.ones_like(scores[half : end + 1]))[0]

    # weight the gradient
    gradients = torch.mul(gradients, slopes)

    return gradients, scores[0 : half], slopes[:, 0, 0, 0]


# returns the logit outputs for a batch of images
def getPredictionParallel(input, model, device, target_class):
    # calculate a prediction
    input = input.to(device)
    output = model(input)
    logit = (output[:, target_class])

    return logit

def getSlopes(trans_img, model, steps, batch_size, baseline, device, target_class):
    if (steps % batch_size != 0):
        print("steps must be evenly divisible by batch size: " + str(batch_size) + "!")
        return 0, 0, 0, 0

    loops = int(steps / batch_size)

    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)

    # batch the input image
    images = trans_img.repeat(steps, 1, 1, 1)
    images = transform_normalize(images)

    baseline = torch.full(images.shape, baseline, dtype = torch.float)

    # generate alpha values as 4D
    alphas = torch.linspace(0, 1, steps)
    alphas = alphas.reshape(steps, 1, 1, 1)

    images, baseline, alphas = images.to(device), baseline.to(device), alphas.to(device)

    baseline_diff = torch.sub(images, baseline)
    inputs = torch.add(baseline, torch.mul(alphas, baseline_diff))

    # run batched input
    for i in range(loops):
        torch.cuda.empty_cache()
        start = i * batch_size 
        end = (i + 1) * batch_size

        logits[start : end] = getPredictionParallel(inputs[start : end], model, device, target_class)
        torch.cuda.empty_cache()
    
    # calculate logit slopes
    slopes = torch.zeros(steps).to(device)
    x_diff = float(alphas.squeeze()[1] - alphas.squeeze()[0])

    slopes[0] = 0

    # calculate all slopes
    for i in range(0, steps - 1):
        y_diff = logits[i + 1] - logits[i]
        slopes[i + 1] = y_diff / x_diff

    return slopes, x_diff

# does an initial point to point slope calculation using a psuedo IG run with steps_hyper steps
# returns the alpha values to be used as well as the spacing of the alpha values
def getAlphaParameters(trans_img, model, steps, steps_hyper, hyper_batch, baseline, device, target_class):
    slopes, step_size = getSlopes(trans_img, model, steps_hyper, hyper_batch, baseline, device, target_class)

    # normalize slopes 0 to 1 to eliminate negatives and preserve magnitude
    slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes))
    # reset the first slope to zero after normalization because it is impossible to be nonzero
    slopes_0_1_norm[0] = 0

    # normalize the slope values so that they sum to 1.0 and preserve magnitude
    slopes_sum_1_norm = slopes_0_1_norm / torch.sum(slopes_0_1_norm)

    # obtain the samples at each alpha step as a float based on the slope (steps/alpha)
    sample_placements_float = torch.mul(slopes_sum_1_norm, steps)
    # truncate the result to int values to clean up decimals, this leaves unused steps (samples)
    sample_placements_int = sample_placements_float.type(torch.int)
    # find how many unused steps are left
    remaining_to_fill = steps - torch.sum(sample_placements_int)

    # find the values which were not truncated to 0 (float values >= 1) 
    # by the int casting and make them 0 in the float array
    non_zeros = torch.where(sample_placements_int != 0)[0]
    sample_placements_float[non_zeros] = -1

    # Find the indicies of the remaining spots to fill from the float array (the zero values) sorted high to low
    remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims = [0])
    # Fill all of these spots in the int array with 1, this gives the final distribution of steps
    sample_placements_int[remaining_hi_lo[0 : remaining_to_fill]] = 1

    # an array that tracks indivdual steps between alpha values
    # this is important to counteract the non-uniform alpha spacing of this method
    alpha_substep_size = torch.zeros(steps)
    # the index at which a range of samples begins, it is a function of num_samples in loop
    alpha_start_index = 0
    # the value at which a range of samples starts, it is a function of step_size
    alpha_start_value = 0
    # holds new alpha values to be created
    alphas = torch.zeros(steps)    

    # generate the new alpha values
    for num_samples in sample_placements_int:        
        if num_samples == 0:
            continue

        # Linearly divide the samples into the required alpha range
        alphas[alpha_start_index: (alpha_start_index + num_samples)] = torch.linspace(alpha_start_value, alpha_start_value + step_size, num_samples + 1)[0 : num_samples]

        # track the step size of the alpha divisions
        alpha_substep_size[alpha_start_index: (alpha_start_index + num_samples)] = (step_size / num_samples)

        alpha_start_index += num_samples
        alpha_start_value += step_size

    return alphas, alpha_substep_size

def IDG(trans_img, model, steps, batch_size, baseline, device, target_class):
    if (batch_size == 0 or steps % batch_size != 0):
        print("steps must be evenly divisible by batch size!")
        return 0, 0, 0, 0

    loops = int(steps / batch_size)

    steps_hyper = steps

    alphas, alpha_substep_size = getAlphaParameters(trans_img, model, steps, steps_hyper, batch_size, baseline, device, target_class)

    alpha_substep_size = alpha_substep_size.to(device)
    alphas.requires_grad = True

    # Transform alpha values to 4D
    alphas = alphas.reshape(steps, 1, 1, 1)
    alpha_substep_size = alpha_substep_size.reshape(steps, 1, 1, 1)

    # array to store the gradient at each step
    gradients = torch.zeros((steps, trans_img.shape[0], trans_img.shape[1], trans_img.shape[2])).to(device)
    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)
    # array to store the slope at each step
    slopes = torch.zeros(steps).to(device)

    # batch the input image
    unmodified_input_img = transform_normalize(trans_img)
    unmodified_input_img = torch.unsqueeze(unmodified_input_img, 0)

    images = unmodified_input_img.repeat(steps, 1, 1, 1)

    # create baseline and get difference from input image
    baseline = torch.full(images.shape, baseline, dtype = torch.float)
    
    # run batched input
    for i in range(loops):
        torch.cuda.empty_cache()
        start = i * batch_size
        end = (i + 1) * batch_size

        gradients[start : end], logits[start : end], slopes[start : end] = getWeightedGradientsParallel(images[start : end], baseline[start : end], alphas[start : end], model, device, target_class)
        torch.cuda.empty_cache()

    # multiply weighted gradients by the alpha value spacings
    # this makes up for the non-uniform sampling
    gradients = torch.multiply(gradients, alpha_substep_size)

    slopes_mul_substep = torch.multiply(slopes, alpha_substep_size[:, 0, 0, 0])

    # integral approximation
    grads = gradients.mean(dim = 0)

    # multiply sum by (original image - baseline)
    baseline_diff = torch.sub(unmodified_input_img, baseline[0]).to(device)
    grads = torch.multiply(grads, baseline_diff.to(device))

    return grads, alphas[:, 0, 0, 0], logits, slopes_mul_substep
