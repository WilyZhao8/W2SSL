# -*- coding: utf-8 -*-
# @Time    : 2021-12-11 13:49
# @Author  : Wily
# @File    : loader.py
# @Software: PyCharm

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class CropTransform:
    def __init__(self, transform1,transform2, cross = 0.3):
        self.transform1 = transform1
        self.transform2 = transform2

        self.c = cross

    def __call__(self, x):
        h,w = x.size
        ch = self.c * h
        cw = self.c * w
        return [self.transform1(x.crop((0,           0,          h//2+ch,    w//2+cw))),
                self.transform1(x.crop((0,           w//2-cw,    h//2+ch,    w))),
                self.transform1(x.crop((h//2-ch,     0,          h,          w//2+cw))),
                self.transform1(x.crop((h//2-ch,     w//2-cw,    h,          w))),
                self.transform2(x.crop((0, 0, h // 2 + ch, w // 2 + cw))),
                self.transform2(x.crop((0, w // 2 - cw, h // 2 + ch, w))),
                self.transform2(x.crop((h // 2 - ch, 0, h, w // 2 + cw))),
                self.transform2(x.crop((h // 2 - ch, w // 2 - cw, h, w)))]



