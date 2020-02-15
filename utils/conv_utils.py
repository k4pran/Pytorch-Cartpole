import torch
import torch.nn.functional as fn


def as_convolved_img(img, filter, padding):
    return fn.conv2d(img.unsqueeze(0).unsqueeze(0), filter.unsqueeze(0), padding=2)
