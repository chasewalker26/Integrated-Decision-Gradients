import torch
from torchvision import transforms
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# this code is adapted from 
# https://github.com/eclique/RISE/blob/master/evaluation.py
# driver code is adapted from 
# https://github.com/eclique/RISE/blob/master/Evaluation.ipynb

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


    def single_run(self, img_tensor, saliency_map, device, max_batch_size = 50):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            device: gpu or cpu.
        Return:
            n_steps (int): the number of steps used over the test.
            scores (nd.array): Array containing scores at every step.
        """

        n_steps = (self.HW + self.step - 1) // self.step

        batch_size = n_steps if n_steps < max_batch_size else max_batch_size

        if batch_size > n_steps:
            print("Batch size cannot be greater than number of steps: " + str(n_steps))
            return 0, []

        original_pred = self.model(img_tensor.to(device))
        _, index = torch.max(original_pred, 1)
        target_class = index[0]
        percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
        original_pred = percentage[target_class].item()

        scores = np.zeros(n_steps + 1)

        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)

            black_pred = self.model(finish.to(device))
            percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
            black_pred = percentage[target_class].item()

            scores[0] = original_pred

        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

            blur_pred = self.model(start.to(device))
            percentage = torch.nn.functional.softmax(blur_pred, dim = 1)[0]
            blur_pred = percentage[target_class].item()

            scores[0] = blur_pred
    
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(saliency_map.reshape(-1, self.HW), axis = 1), axis = -1)

        density = np.zeros(n_steps + 1)

        if self.mode == "del":
            density[0] = 1
        elif self.mode == "ins":
            density[0] = 0

        total_steps = 1
        num_batches = int((n_steps) / batch_size)
        leftover = (n_steps) % batch_size

        batches = np.full(num_batches + 1, batch_size)
        batches[-1] = leftover

        for batch in batches:
            images = torch.zeros((batch, start.shape[1], start.shape[2], start.shape[3]))

            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                coords = salient_order[:, self.step * (total_steps - 1) : self.step * (total_steps)]
                start.cpu().numpy().reshape(1, 3, self.HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, self.HW)[0, :, coords]
                images[i] = start

                total_steps += 1

            # get predictions from image batch
            output = self.model(images.to(device)).detach()
            percentage = torch.nn.functional.softmax(output, dim = 1)
            scores[total_steps - batch : total_steps] = percentage[:, target_class].cpu().numpy()

        return n_steps, scores