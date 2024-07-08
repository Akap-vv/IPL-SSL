import numpy as np
import torch
import random
from torch.nn import functional as F
def DataTransform(sample,factor=4):
    # weak_aug = scaling(sample)
    # strong_aug = jitter(permutation(sample))
    # weak_aug = sample
    weak_aug = random_mask(sample,mask_K=2)
    strong_aug = random_mask(permutation(sample,max_segments=factor),mask_K=factor)
    return weak_aug, strong_aug

def DataTransform_784(sample,factor=4):
    # weak_aug = scaling(sample)
    # strong_aug = jitter(permutation(sample))
    weak_aug = sample
    # weak_aug = random_mask(sample, mask_K=4)
    strong_aug = random_mask_784(permutation_784(sample,max_segments=factor),mask_K=factor)
    return weak_aug, strong_aug

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=0., scale=sigma, size=x.shape)
    # import pdb;pdb.set_trace()
    return x + factor

def scaling(x, ratio=0.001,sigma=1.1):
    # factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], 1))
    # import pdb;pdb.set_trace()
    noise = torch.rand(x.shape[0], x.shape[1]) * ratio
    return x + noise
    # return x * factor
# def scaling(x, sigma=1.1):
#     # https://arxiv.org/pdf/1706.00527.pdf
#     x = x.cpu().numpy()
#     factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[1], x.shape[2]))
#     ai = []
#     for i in range(x.shape[1]):
#         xi = x[:, i, :]
#         ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
#     return np.concatenate((ai), axis=1)


def permutation(x, max_segments=4, seg_mode="random"):
    len = x.shape[1]
    orig_steps = np.arange(len)
    # x = x.cpu().numpy()
    # num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    num_segs = np.full(x.shape[0], max_segments)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(len- 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            random.shuffle(splits)
            # import pdb;pdb.set_trace()
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
        # import pdb;pdb.set_trace()
    return torch.from_numpy(ret)
def permutation_784(x, max_segments=4, seg_mode="random"):
    x = x.reshape(-1,28,28)
    orig_steps = np.arange(28)
    # x = x.cpu().numpy()
    # num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    num_segs = np.full(x.shape[0], max_segments)
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(28 - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            random.shuffle(splits)
            # import pdb;pdb.set_trace()
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
        # import pdb;pdb.set_trace()
    return torch.from_numpy(ret).reshape(-1,784)

def random_mask(x, mask_K=4):
    input_length = x.shape[1]
    mask_temp = torch.randint(0,high=input_length, size=(x.shape[0], mask_K))
    mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
    mask_4 = mask_temp_one_hot[:, 0, :].int().float()
    if (mask_temp_one_hot.size(1) >= 2):
        for i in range(1, mask_temp_one_hot.size(1)):
            mask_4 = mask_4 * (mask_temp_one_hot[:, i, :].int().float())
    # import pdb;pdb.set_trace()
    x_masked = x * mask_4
    # import pdb;pdb.set_trace()
    return x_masked

def random_mask_784(x, mask_K=4):
    input_length = 28
    mask_temp = torch.randint(0,high=input_length, size=(x.shape[0], mask_K))
    mask_temp_one_hot = (F.one_hot(mask_temp, num_classes=input_length) == 0)
    mask_4 = mask_temp_one_hot[:, 0, :].int().float()
    if (mask_temp_one_hot.size(1) >= 2):
        for i in range(1, mask_temp_one_hot.size(1)):
            mask_4 = mask_4 * (mask_temp_one_hot[:, i, :].int().float())
    # import pdb;pdb.set_trace()
    mask_4_3dim = mask_4.unsqueeze(2).expand(-1, -1, 28)
    # import pdb;pdb.set_trace()
    x_masked = x.reshape(-1,28,28) * mask_4_3dim
    # import pdb;pdb.set_trace()
    return x_masked.reshape(-1,784)