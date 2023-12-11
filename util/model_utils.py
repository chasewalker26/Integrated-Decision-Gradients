import torch

# returns the softmax classification value of an image for the highest predicted class or a target class
def getPrediction(input, model, device, target_class):
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
    # calculate a prediction
    input = input.to(device)
    output = model(input)

    _, index = torch.max(output, 1)

    # # open the class list so the detected class string can be returned for printing
    # with open('class_maps/ImageNet/imagenet_classes.txt') as f:
    #     classes = [line.strip() for line in f.readlines()]
    # return index[0], classes[index[0]]

    return index[0]

# returns the gradients from the model for an input
def getGradients(input, model, device, target_class):
    input = input.to(device)

    # track gradients
    input.requires_grad = True

    # calculate predictions
    output = model(input)

    # get output for the target class
    score = output[0][target_class]

    # get gradients of the target class w.r.t. the input image
    gradients = torch.autograd.grad(score, input)[0][0]

    # turn off grad tracking
    input.requires_grad = False

    return gradients