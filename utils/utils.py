import random
import numpy as np
from data.data_augmentation import elastic_transform, gaussian_blur, gaussian_noise, random_crop
from skimage.transform import resize
import pdb

def force_inside_img(x, patch_size, img_shape):
    x_low = int(x - patch_size / 2)
    x_up = int(x + patch_size / 2)
    if x_low < 0:
        x_up -= x_low
        x_low = 0
    elif x_up > img_shape[2]:
        # x_low -= (x_up - img_shape[2])
        x_low = max(0, x_low - (x_up - img_shape[2]))
        x_up = img_shape[2]
    return x_low, x_up


def extract_random_patch(img, mask, weight, i, subset, empty_interval=5, patch_size=128):
    flag_empty = False

    # padding
	    # if min(img.shape) < patch_size:
	    #     pdb.set_trace()
	    #     z = max(0, patch_size-img.shape[0])
	    #     y = max(0, patch_size-img.shape[1])
	    #     x = max(0, patch_size-img.shape[2])
	    #     z_min = math.floor(z/2)
	    #     z_max = math.ceil(z/2)
	    #     y_min = math.floor(y/2)
	    #     y_max = math.ceil(y/2)
	    #     x_min = math.floor(x/2)
	    #     x_max = math.ceil(x/2)
	    #     img = np.pad(img, ((z_min, z_max), (y_min, y_max), (x_min, x_max)), 'constant', constant_values=(4, 6))


    # list available vertebrae
    verts = np.unique(mask)
    chosen_vert = verts[random.randint(1, len(verts) - 1)]

    # create corresponde instance memory and ground truth
    ins_memory = np.copy(mask)
    ins_memory[ins_memory <= chosen_vert] = 0
    ins_memory[ins_memory > 0] = 1

    gt = np.copy(mask)
    gt[gt != chosen_vert] = 0
    gt[gt > 0] = 1

    # send empty mask sample in certain frequency
    if i % empty_interval == 0:
        patch_center = [np.random.randint(0, s) for s in img.shape]
        x = patch_center[2]
        y = patch_center[1]
        z = patch_center[0]

        # for instance memory
        gt = np.copy(mask)
        flag_empty = True
    else:
        indices = np.nonzero(mask == chosen_vert)
        lower = [np.min(i) for i in indices]
        upper = [np.max(i) for i in indices]
        # random center of patch
        x = random.randint(lower[2], upper[2])
        y = random.randint(lower[1], upper[1])
        z = random.randint(lower[0], upper[0])

    # force random patches' range within the image
    x_low, x_up = force_inside_img(x, patch_size, img.shape)
    y_low, y_up = force_inside_img(y, patch_size, img.shape)
    z_low, z_up = force_inside_img(z, patch_size, img.shape)

    # crop the patch
    img_patch = img[z_low:z_up, y_low:y_up, x_low:x_up]
    ins_patch = ins_memory[z_low:z_up, y_low:y_up, x_low:x_up]
    gt_patch = gt[z_low:z_up, y_low:y_up, x_low:x_up]
    weight_patch = weight[z_low:z_up, y_low:y_up, x_low:x_up]
    # print(img.shape, img_patch.shape)
    #  if the label is empty mask
    if flag_empty:
        ins_patch = np.copy(gt_patch)
        ins_patch[ins_patch > 0] = 1
        gt_patch = np.zeros_like(ins_patch)
        weight_patch = np.ones_like(ins_patch)

    out_shape = (128, 128, 128)
    img_patch = resize(img_patch, out_shape, order=1, preserve_range=True)
    ins_patch = resize(ins_patch, out_shape, order=0, preserve_range=True)
    gt_patch = resize(gt_patch, out_shape, order=0, preserve_range=True)
    weight_patch = resize(weight_patch, out_shape, order=1, preserve_range=True)
    
    # Randomly on-the-fly Data Augmentation
    # 50% chance elastic deformation
    if subset == 'train':
        if np.random.rand() > 0.5:
            img_patch, gt_patch, ins_patch, weight_patch = elastic_transform(img_patch, gt_patch, ins_patch,
                                                                             weight_patch, alpha=20, sigma=5)
        # 50% chance gaussian blur
        if np.random.rand() > 0.5:
            img_patch = gaussian_blur(img_patch)
        # 50% chance gaussian noise
        if np.random.rand() > 0.5:
            img_patch = gaussian_noise(img_patch)

        # 50% random crop along z-axis
        if np.random.rand() > 0.5:
            img_patch, ins_patch, gt_patch, weight_patch = random_crop(img_patch, ins_patch, gt_patch
                                                                       , weight_patch)

    # decide label of completeness(partial or complete)
    vol = np.count_nonzero(gt == 1)
    sample_vol = np.count_nonzero(gt_patch == 1)
    c_label = 0 if float(sample_vol / (vol + 0.0001)) < 0.98 else 1

    img_patch = np.expand_dims(img_patch, axis=0)
    ins_patch = np.expand_dims(ins_patch, axis=0)
    gt_patch = np.expand_dims(gt_patch, axis=0)
    weight_patch = np.expand_dims(weight_patch, axis=0)
    c_label = np.expand_dims(c_label, axis=0)

    return img_patch, ins_patch, gt_patch, weight_patch, c_label
