import torch

# This file implements IG (LIG) and IDG
# For the attributions, the input is a normalized tensor image (1, 3, 224, 224)

def IG(input, model, steps, batch_size, alpha_star, baseline, device, target_class):
    if (steps % batch_size != 0):
        print("steps must be evenly divisible by batch size: " + str(batch_size) + "!")
        return 0, 0, 0, 0

    loops = int(steps / batch_size)

    # generate alpha values as 4D
    alphas = torch.linspace(0, 1, steps, requires_grad = True)

    alphas = alphas.reshape(steps, 1, 1, 1).to(device)

    # array to store the gradient at each step
    gradients = torch.zeros((steps, input.shape[1], input.shape[2], input.shape[3])).to(device)
    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)

    if torch.is_tensor(baseline):
        baseline = baseline
    else:
        baseline = torch.full(input.shape, baseline, dtype = torch.float)

    input = input.to(device)
    baseline = baseline.to(device)
    baseline_diff = torch.sub(input, baseline)

    # run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_imgs = torch.add(baseline, torch.mul(alphas[start : end], baseline_diff))

        gradients[start : end], logits[start : end] = getGradientsParallel(interp_imgs, model, target_class)

    max_perc = torch.max(logits)
    cutoff_perc = max_perc * alpha_star

    # IG: sum all the gradients
    if alpha_star == 1:
        grads = gradients.mean(dim = 0)
    # LeftIG: sum the gradients up to the cutoff point and no later
    else:
        # find where cutoff point is
        cutoff_steps = torch.where(logits > cutoff_perc)[0]

        if len(cutoff_steps) != 0:
            cutoff_step = torch.where(logits > cutoff_perc)[0][0]
        else:
            cutoff_step = 1

        # avoid rare case where no attribution is returned
        if cutoff_step == 0:
            cutoff_step = 1

        grads = (gradients[0 : cutoff_step]).mean(dim = 0)

    # multiply sum by (original image - baseline)
    grads = torch.multiply(grads, baseline_diff[0].unsqueeze(0))
    
    return grads.squeeze()

def IDG(input, model, steps, batch_size, baseline, device, target_class):
    if (batch_size == 0 or steps % batch_size != 0):
        print("steps must be evenly divisible by batch size!")
        return 0, 0, 0

    loops = int(steps / batch_size)
    
    # baseline given as image
    if torch.is_tensor(baseline):
        baseline = baseline
    # make baseline image from a float [0, 1]
    else:
        baseline = torch.full(input.shape, baseline, dtype = torch.float)

    input = input.to(device)
    baseline = baseline.to(device)
    baseline_diff = torch.sub(input, baseline)

    # find IG slopes
    slopes, step_size = getSlopes(baseline, baseline_diff, model, steps, batch_size, device, target_class)
    # find new alpha spacing
    alphas, alpha_substep_size = getAlphaParameters(slopes, steps, step_size)

    alphas.requires_grad = True
    alphas = alphas.reshape(steps, 1, 1, 1).to(device)
    alpha_substep_size = alpha_substep_size.reshape(steps, 1, 1, 1).to(device)

    # array to store the gradient at each step
    gradients = torch.zeros((steps, input.shape[1], input.shape[2], input.shape[3])).to(device)
    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)
    # array to store the slope at each step
    slopes = torch.zeros(steps).to(device)
    
    # run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_imgs = torch.add(baseline, torch.mul(alphas[start : end], baseline_diff))
        
        gradients[start : end], logits[start : end] = getGradientsParallel(interp_imgs, model, target_class)

    # calculate logit slopes
    slopes = torch.zeros(steps).to(device)
    slopes[0] = 0

    # calculate all slopes
    for i in range(0, steps - 1):
        slopes[i + 1] = (logits[i + 1] - logits[i]) / (alphas[i + 1] - alphas[i])

    gradients = torch.multiply(gradients, slopes.reshape(steps, 1, 1, 1))

    # multiply weighted gradients by the alpha value spacings
    # this makes up for the non-uniform sampling
    gradients = torch.multiply(gradients, alpha_substep_size)

    # integral approximation
    grads = gradients.mean(dim = 0)
    # multiply sum by (original image - baseline)
    grads = torch.multiply(grads, baseline_diff[0].unsqueeze(0))

    return grads.squeeze()

# returns the gradients from the model for an input
def getGradientsParallel(inputs, model, target_class):
    output = model(inputs)
    scores = output[:, target_class]

    gradients = torch.autograd.grad(scores, inputs, grad_outputs = torch.ones_like(scores))[0]

    return gradients.detach().squeeze(), scores.detach().squeeze()

# returns the logit outputs for a batch of images
def getPredictionParallel(inputs, model, target_class):
    # calculate a prediction
    output = model(inputs)

    scores = output[:, target_class].detach()

    return scores.squeeze()

def getSlopes(baseline, baseline_diff, model, steps, batch_size, device, target_class):
    if (steps % batch_size != 0):
        print("steps must be evenly divisible by batch size: " + str(batch_size) + "!")
        return 0, 0

    loops = int(steps / batch_size)

    # generate alpha values as 4D
    alphas = torch.linspace(0, 1, steps)
    alphas = alphas.reshape(steps, 1, 1, 1).to(device)

    # array to store the logit at each step
    logits = torch.zeros(steps).to(device)

    # run batched input
    for i in range(loops):
        start = i * batch_size
        end = (i + 1) * batch_size

        interp_imgs = torch.add(baseline, torch.mul(alphas[start : end], baseline_diff))
        
        logits[start : end] = getPredictionParallel(interp_imgs, model, target_class)

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
def getAlphaParameters(slopes, steps, step_size):
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
    # by the int casting and make them -1 in the float array
    non_zeros = torch.where(sample_placements_int != 0)[0]
    sample_placements_float[non_zeros] = -1

    # Find the indicies of the remaining spots to fill from the float array (the zero values) sorted high to low
    remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims = [0])
    # Fill all of these spots in the int array with 1, this gives the final distribution of steps
    sample_placements_int[remaining_hi_lo[0 : remaining_to_fill]] = 1

    # holds new alpha values to be created
    alphas = torch.zeros(steps)    
    # an array that tracks indivdual steps between alpha values
    # this is important to counteract the non-uniform alpha spacing of this method
    alpha_substep_size = torch.zeros(steps)

    # the index at which a range of samples begins, it is a function of num_samples in loop
    alpha_start_index = 0
    # the value at which a range of samples starts, it is a function of step_size
    alpha_start_value = 0

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
