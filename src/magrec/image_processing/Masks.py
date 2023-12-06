
import torch
import numpy as np
import matplotlib.pyplot as plt



def mask_vert_dir(image, threshold, plot = False):

    image_abs = np.abs(image)
    wieght_mask = np.ones(image.size())
    img_size = image.shape 


    for idx in range(img_size[1]):
        for idx2 in range(img_size[0]):
            if image_abs[-idx2, idx] > threshold:
                break
            wieght_mask[-idx2, idx] = 0 

        for idx2 in range(img_size[0]):
            if image_abs[idx2,idx] > threshold:
                break
            wieght_mask[idx2,idx] = 0 

    wieght_mask = torch.from_numpy(wieght_mask).to(torch.float32)
    if plot:
        plot_mask(image, wieght_mask)
    return wieght_mask


def mask_hor_dir(image, threshold, plot = False):

    image_abs = np.abs(image)
    wieght_mask = np.ones(image.size())
    img_size = image_abs.shape 


    for idx in range(img_size[0]):
        for idx2 in range(img_size[1]):
            if image_abs[idx, -idx2] > threshold:
                break
            wieght_mask[idx, -idx2] = 0 

        for idx2 in range(img_size[1]):
            if image_abs[idx,idx2] > threshold:
                break
            wieght_mask[idx,idx2] = 0 

    wieght_mask = torch.from_numpy(wieght_mask).to(torch.float32)
    if plot:
        plot_mask(image, wieght_mask)
    return wieght_mask


def plot_mask(image, wieght_mask):
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    norm_img = image - np.nanmean(image)
    c_range = np.nanmax(np.abs(norm_img))


    plt.subplot(1,3,1)
    plt.imshow(norm_img, cmap='bwr', vmin = -c_range, vmax = c_range)
    plt.title('B image')
    # make the colour bar the same size as the image
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('B (T)')


    plt.subplot(1,3,2)
    plt.imshow(norm_img*wieght_mask, cmap='bwr', vmin = -c_range, vmax = c_range)
    plt.title('Masked B image')
    # make the colour bar the same size as the image
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('B (T)')


    plt.subplot(1,3,3)
    plt.imshow(wieght_mask, vmin = 0, vmax = 1)
    plt.title('Mask')
    # make the colour bar the same size as the image
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Wieght value')

    # increase the horizontal space between subplots
    plt.tight_layout(pad=0.5)



