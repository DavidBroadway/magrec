
import torch
import numpy as np
import matplotlib.pyplot as plt



def mask_vert_dir(image, threshold, plot = False):

    image = np.abs(image)
    wieght_mask = np.ones(image.size())
    img_size = image.shape 


    for idx in range(img_size[1]):
        for idx2 in range(img_size[0]):
            if image[-idx2, idx] > threshold:
                break
            wieght_mask[-idx2, idx] = 0 

        for idx2 in range(img_size[0]):
            if image[idx2,idx] > threshold:
                break
            wieght_mask[idx2,idx] = 0 

    wieght_mask = torch.from_numpy(wieght_mask).to(torch.float32)
    if plot:
        plot_mask(image, wieght_mask)
    return wieght_mask


def mask_hor_dir(image, threshold, plot = False):

    image = np.abs(image)
    wieght_mask = np.ones(image.size())
    img_size = image.shape 


    for idx in range(img_size[0]):
        for idx2 in range(img_size[1]):
            if image[idx, -idx2] > threshold:
                break
            wieght_mask[idx, -idx2] = 0 

        for idx2 in range(img_size[1]):
            if image[idx,idx2] > threshold:
                break
            wieght_mask[idx,idx2] = 0 

    wieght_mask = torch.from_numpy(wieght_mask).to(torch.float32)
    if plot:
        plot_mask(image, wieght_mask)
    return wieght_mask


def plot_mask(image, wieght_mask):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(16)

    plt.subplot(1,3,2)
    plt.imshow(image)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(wieght_mask, vmin = 0, vmax = 1)
    plt.colorbar()

