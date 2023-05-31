import torch
from torchvision import transforms
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# this code is adapted from 
# https://github.com/eclique/RISE/blob/master/evaluation.py
# driver code is adapted from 
# https://github.com/eclique/RISE/blob/master/Evaluation.ipynb

n_classes = 1000

# invert standard ImageNet normalization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""

    # create nxn zeros
    inp = np.zeros((klen, klen))

    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1

    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k

    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, HW, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW: image size in pixels given as h*y e.g. 224*224.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.HW = HW
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, saliency_map, verbose, device, save_to = None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.to(device))
        top, index = torch.max(pred, 1)

        n_steps = (self.HW + self.step - 1) // self.step

        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)

        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(saliency_map.reshape(-1, self.HW), axis = 1), axis = -1)

        with open('supplementaryCode/class_maps/imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        for i in range(n_steps + 1):
            pred = self.model(start.to(device))
            pr, cl = torch.topk(pred, 2)

            # confidence of prediciton in range 0 - 1
            percentage = torch.nn.functional.softmax(pred, dim = 1)[0]
            confidence = percentage[index[0]].item()

            if verbose == 2:
                print('{}: {:.3f}'.format(classes[cl[0][0]], float(pr[0][0])))
                print('{}: {:.3f}'.format(classes[cl[0][1]], float(pr[0][1])))

            scores[i] = confidence

            if i < n_steps:
                coords = salient_order[:, self.step * i : self.step * (i + 1)]
                start.detach().cpu().numpy().reshape(1, 3, self.HW)[0, :, coords] = finish.detach().cpu().numpy().reshape(1, 3, self.HW)[0, :, coords]

        return n_steps, classes[index], scores