
# coding: utf-8

# In[1]:


import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import struct
import tifffile as tiff


# In[2]:


def open_hdr(hdr_path):
    fla_path = hdr_path[:-3] + "fla"
    with open(hdr_path) as f:
        f_lines = f.readlines()
        for line in f_lines:
            if line[:7] == "samples":
                samples = int(line.split()[-1])
            if line[:5] == "lines":
                lines = int(line.split()[-1])
            if line[:5] == "bands":
                bands = int(line.split()[-1])
    array_data = np.fromfile(fla_path, dtype=np.uint16).reshape([bands, lines, samples])
    return array_data.astype(np.float32)


# In[3]:


def open_tiff(filename):
    return np.transpose(tiff.imread(filename), (2, 0, 1))

def open_hdr_or_tiff(filename):
    if filename[-3:] == 'tif':
        return open_tiff(filename)
    else:
        return open_hdr(filename)


# In[4]:


def normalize(filenames, resultpath, zero_mean=False, means=None):
    """
    Reads image file in filenames dividing them by 2**16.
    And then subtract each band's average value from the images.
    The resulting arrays will be concatenated to 1 npz file.
    If means is give, then this function subtract the means from the images.
    
    The resulting array is saved in resultpath file, returning means.
    """
    
    filenames = sorted(filenames)
    images = np.array([open_hdr_or_tiff(f)/65535 for f in filenames]) # (n_images, n_bands, width, height)
    print(images.shape)
    n_channels = images.shape[1]
    
    if means is None: # calculate bandwise means.
        means = np.mean(images, axis=(0, 2, 3))
        
    if zero_mean:
        images = images - means.reshape([1, n_channels, 1, 1])
    
    np.save(resultpath, images, allow_pickle=False)
    return means


# In[7]:


# Preprocessing for Track 1

os.makedirs('PIRMt1/normalized/', exist_ok=True)

t1_train_hr = sorted(glob.glob('PIRMt1/training_hr/*.hdr'))
t1_train_lr = sorted(glob.glob('PIRMt1/training_lr/*lr3.hdr'))
t1_val_lr = sorted(glob.glob('PIRMt1/validation_lr/*lr3.hdr'))
t1_val_hr = sorted(glob.glob('PIRMt1/validation_hr/*.hdr'))
t1_test_lr = sorted(glob.glob('PIRMt1/testing_lr/*lr3.hdr'))

# We conduct mean subtraction on training samples.
_ = normalize(t1_train_hr, 'PIRMt1/normalized/train_hr.npy', zero_mean=False)
_ = normalize(t1_val_hr, 'PIRMt1/normalized/val_hr.npy', zero_mean=False)
lr_means = normalize(t1_train_lr, 'PIRMt1/normalized/train_lr3.npy', zero_mean=True)
_ = normalize(t1_val_lr, 'PIRMt1/normalized/val_lr3.npy', zero_mean=True, means=lr_means)
_ = normalize(t1_test_lr, 'PIRMt1/normalized/testing_lr3_data.npy', zero_mean=True, means=lr_means)
np.save('PIRMt1/normalized/lr_means.npy', lr_means)


# In[6]:


# Preprocessing for Track 2

os.makedirs('PIRMt2/normalized/', exist_ok=True)

t2_train_spec_hr = sorted(glob.glob('PIRMt2/training_hr/*.hdr'))
t2_train_color_reg_hr = sorted(glob.glob('PIRMt2/training_hr/*hr_registered.tif'))
t2_train_color_unreg_hr = sorted(glob.glob('PIRMt2/training_hr/*hr_unregistered.tif'))
t2_train_spec_lr = sorted(glob.glob('PIRMt2/training_lr/*_lr3.hdr'))
t2_train_color_reg_lr = sorted(glob.glob('PIRMt2/training_lr/*lr3_registered.tif'))
t2_train_color_unreg_lr = sorted(glob.glob('PIRMt2/training_lr/*lr3_unregistered.tif'))

_ = normalize(t2_train_spec_hr, 'PIRMt2/normalized/train_spec_hr.npy', zero_mean=False)
# color_reg_hr_means = normalize(t2_train_color_reg_hr, 'PIRMt2/normalized/train_color_reg_hr.npy', zero_mean=False)
# color_unreg_hr_means = normalize(t2_train_color_unreg_hr, 'PIRMt2/normalized/train_color_unreg_hr.npy', zero_mean=False)
spec_lr_means = normalize(t2_train_spec_lr, 'PIRMt2/normalized/train_spec_lr.npy', zero_mean=True)
color_reg_lr_means = normalize(t2_train_color_reg_lr, 'PIRMt2/normalized/train_color_reg_lr.npy', zero_mean=True)
color_unreg_lr_means = normalize(t2_train_color_unreg_lr, 'PIRMt2/normalized/train_color_unreg_lr.npy', zero_mean=True)

#np.save('PIRMt2/normalized/spec_hr_means.npy', spec_hr_means)
#np.save('PIRMt2/normalized/color_reg_hr_means.npy', color_reg_hr_means)
#np.save('PIRMt2/normalized/color_unreg_hr_means.npy', color_unreg_hr_means)
np.save('PIRMt2/normalized/spec_lr_means.npy', spec_lr_means)
np.save('PIRMt2/normalized/color_reg_lr_means.npy', color_reg_lr_means)
np.save('PIRMt2/normalized/color_unreg_lr_means.npy', color_unreg_lr_means)


# In[5]:


t2_val_spec_lr = sorted(glob.glob('PIRMt2/validation_lr/*_lr3.hdr'))
t2_val_color_reg_lr = sorted(glob.glob('PIRMt2/validation_lr/*lr3_registered.tif'))
t2_val_color_unreg_lr = sorted(glob.glob('PIRMt2/validation_lr/*lr3_unregistered.tif'))
t2_val_spec_hr = sorted(glob.glob('PIRMt2/validation_hr/*.hdr'))

t2_test_spec_lr = sorted(glob.glob('PIRMt2/testing_lr/*_lr3.hdr'))
t2_test_color_reg_lr = sorted(glob.glob('PIRMt2/testing_lr/*lr3_registered.tif'))
t2_test_color_unreg_lr = sorted(glob.glob('PIRMt2/testing_lr/*lr3_unregistered.tif'))


# In[7]:


_ = normalize(t2_val_spec_lr, 'PIRMt2/normalized/val_spec_lr.npy', means=spec_lr_means, zero_mean=True)
_ = normalize(t2_val_spec_hr, 'PIRMt2/normalized/val_spec_hr.npy', zero_mean=False)

_ = normalize(t2_val_color_reg_lr, 'PIRMt2/normalized/val_color_reg_lr.npy', means=color_reg_lr_means, zero_mean=True)
_ = normalize(t2_val_color_unreg_lr, 'PIRMt2/normalized/val_color_unreg_lr.npy', means=color_unreg_lr_means, zero_mean=True)

_ = normalize(t2_test_spec_lr, 'PIRMt2/normalized/testing_spec_lr.npy', means=spec_lr_means, zero_mean=True)
_ = normalize(t2_test_color_reg_lr, 'PIRMt2/normalized/testing_color_reg_lr.npy', means=color_reg_lr_means, zero_mean=True)
_ = normalize(t2_test_color_unreg_lr, 'PIRMt2/normalized/testing_color_unreg_lr.npy', means=color_unreg_lr_means, zero_mean=True)


# In[8]:


def upscale_spec(targets):
    results = []
    
    for im in targets:
        im_result = []
        for band in im:
            im_result.append(cv2.resize(band, (band.shape[1]*2, band.shape[0]*2), interpolation=cv2.INTER_CUBIC))
        results.append(im_result)
    return results


# In[9]:


def stack_spec_and_color(filenames_spec, filenames_color, resultpath, zero_mean=True, means=None):
    """
    Reads image file in filenames dividing them by 2**16.
    And then subtract each band's average value from the images.
    The resulting arrays will be concatenated to 1 npz file.
    If means is give, then this function subtract the means from the images.
    
    The resulting array is saved in resultpath file, returning means.
    """
    
    filenames_spec, filenames_color = sorted(filenames_spec), sorted(filenames_color)
    specs = np.array([open_hdr_or_tiff(f)/65535 for f in filenames_spec]) # (n_images, n_bands, width, height)
    specs = upscale_spec(specs)
    colors = np.array([open_hdr_or_tiff(f)/65535 for f in filenames_color])
    
    images = np.concatenate([specs, colors], axis=1)
    
    if means is None: # calculate bandwise means.
        means = np.mean(images, axis=(0, 2, 3))
        
    if zero_mean:
        images = images - means.reshape([1, len(means), 1, 1])
    
    np.save(resultpath, images, allow_pickle=False)
    return means


# In[10]:


# Preprocessing for Track 2

os.makedirs('PIRMt2/normalized/', exist_ok=True)

t2_train_spec_hr = sorted(glob.glob('PIRMt2/training_hr/*.hdr'))
t2_train_color_reg_hr = sorted(glob.glob('PIRMt2/training_hr/*hr_registered.tif'))
t2_train_color_unreg_hr = sorted(glob.glob('PIRMt2/training_hr/*hr_unregistered.tif'))
t2_train_spec_lr = sorted(glob.glob('PIRMt2/training_lr/*_lr3.hdr'))
t2_train_color_reg_lr = sorted(glob.glob('PIRMt2/training_lr/*lr3_registered.tif'))
t2_train_color_unreg_lr = sorted(glob.glob('PIRMt2/training_lr/*lr3_unregistered.tif'))

stack_means = stack_spec_and_color(t2_train_spec_lr, t2_train_color_reg_lr,'PIRMt2/normalized/train_stack_reg.npy', zero_mean=True)
_ = stack_spec_and_color(t2_val_spec_lr, t2_val_color_reg_lr,'PIRMt2/normalized/val_stack_reg.npy', zero_mean=True, means=stack_means)
_ = stack_spec_and_color(t2_test_spec_lr, t2_test_color_reg_lr,'PIRMt2/normalized/test_stack_reg.npy', zero_mean=True, means=stack_means)


stack_means = stack_spec_and_color(t2_train_spec_lr, t2_train_color_unreg_lr,'PIRMt2/normalized/train_stack_unreg.npy', zero_mean=True)
_ = stack_spec_and_color(t2_val_spec_lr, t2_val_color_unreg_lr,'PIRMt2/normalized/val_stack_unreg.npy', zero_mean=True, means=stack_means)
_ = stack_spec_and_color(t2_test_spec_lr, t2_test_color_unreg_lr,'PIRMt2/normalized/test_stack_unreg.npy', zero_mean=True, means=stack_means)
_ = stack_spec_and_color(t2_test_spec_lr, t2_test_color_unreg_lr,'PIRMt2/normalized/test_stack_unreg.npy', zero_mean=True, means=stack_means)

