'''
The python script for data analysis of this specific inpainting task.

Author: WANG Xiangzhi
Date: 21-Dec-2023
'''

# Since the non-deep method, namely, Alexander Telea's inpainting algorithm, doesn't require training
# The code here only provide the inference & evaluation methods with will follow the deep method


import numpy as np
from NonDeepMethodBaseline import pyheal
from skimage.metrics import structural_similarity as ssim

# Infer the image with the non-deep method
def infer_image_non_deep_method(img, mask):
    pyheal_img = img.copy()
    pyheal.inpaint(pyheal_img, mask.astype(bool, copy=True), 5)
    return pyheal_img


# Compute l1 loss
def compute_l1_loss(predicted, target):
    return np.mean(np.abs(predicted - target))

# Compute MSE loss
def compute_MSE_loss(predicted, target):
    return np.mean(np.square(predicted - target))

# Compute PSNR
def compute_PSNR(predicted, target):
    return 10 * np.log10(1 / compute_MSE_loss(predicted, target))

# Compute SSIM
def compute_SSIM(predicted, target):
    return ssim(predicted, target, multichannel=True, channel_axis=2)


# Evaluate images
def evaluation(predicted, target):
    l1_loss = compute_l1_loss(predicted, target)
    MSE_loss = compute_MSE_loss(predicted, target)
    PSNR = compute_PSNR(predicted, target)
    SSIM = compute_SSIM(predicted, target)

    return l1_loss, MSE_loss, PSNR, SSIM


