import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from nilearn.image import resample_img
from scipy.ndimage import zoom, distance_transform_edt
from skimage import measure
import os


cwd = os.getcwd()



patient_test = glob.glob(cwd + "\database\\training\*\*_frame01.nii")
patient_test_gt = glob.glob(cwd + "\database\\training\*\*_frame01_gt.nii")



def normalize_and_resample(imgs, normalize=True, target_voxel_size = [1.25, 1.25, 10]):
    '''
    Normalize and resample a list of images.

    This function takes a list of image paths as input and performs normalization and resampling on each image
    to match the recommended voxel size specified by the target_voxel_size parameter. The images are loaded
    using nibabel, normalized using mean and standard deviation, and resampled using the zoom function from
    scipy. The normalized and resampled Nifti1 images are then returned as a list.

    Parameters:
    - imgs (list): A list of image paths.
    - target_voxel_size (list): The recommended voxel size to resample the images to. Default is [1.25, 1.25, 10] :
      these values are provided by the paper.

    Returns:
    - ret (list): A list of normalized and resampled Nifti1 images.
    '''

    # create an empty list to store normalized images
    ret = []

    # loop through each image
    for image_path in imgs:
        # load image using nibabel
        image = nib.load(image_path)

        # get image data as a numpy array
        image_array = image.get_fdata()

        if(normalize):
            # normalize the image data using mean and standard deviation
            mean = np.mean(image_array)
            std = np.std(image_array)
            image_array = (image_array - mean) / std

        # get current voxel size
        current_voxel_size = image.header.get_zooms()

        # calculate the resampling factor
        resampling_factor = np.divide(current_voxel_size, target_voxel_size)

        # resample the image data
        image_array = zoom(image_array, resampling_factor, order=1)

        # create a new nibabel NIFTI1Image with the normalized image data
        image_array = nib.Nifti1Image(image_array, image.affine, image.header)

        # append the normalized image to the list
        ret.append(image_array)

    return ret



def convert_to_list(imgs):
    # create an empty list to store images
    ret = []

    # loop through each image
    for image_path in imgs:
        # load image using nibabel
        image = nib.load(image_path)

        # get image data as a numpy array
        image_array = image.get_fdata()

        # create a new nibabel NIFTI1Image with the normalized image data
        image_array = nib.Nifti1Image(image_array, image.affine, image.header)

        # append the image to the list
        ret.append(image_array)

    return ret

patient_test_normres = normalize_and_resample(patient_test)
patient_test = convert_to_list(patient_test)


fig, ax = plt.subplots(1, 2)

ax[0].imshow(patient_test_normres[2].get_fdata()[:,:,1], cmap='gray', label="After normalization and resampling")
ax[0].legend()

ax[1].imshow(patient_test[2].get_fdata()[:,:,1], cmap='gray', label="Before normalization and resampling")
ax[1].legend()

plt.show()





