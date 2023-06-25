import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import math
import numpy as np
import monai
from typing import Union

df = pd.read_csv('database/metadata.csv')
base_path = "."

def visualize_2D(data_2D_dict, ax_=None, title=None):
    '''
    :param data_dict: {"image": 2D, "label": 2D}
    '''
    image_2D = data_2D_dict["image"].squeeze()
    label_2D = data_2D_dict["label"].squeeze()

    if ax_ is None:
        fig, ax = plt.subplots()
    else:
        ax = ax_

    # color map for labels: 1-Red 2-Green 3-Blue
    cmap = ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)])
    ax.imshow(image_2D, cmap='gray')
    ax.imshow(label_2D, cmap=cmap)
    ax.axis('off')
    if title is not None:
        ax.set_title(title)

    if ax_ is None:
        plt.show()


def visualize_3D(data_dict, save_file_name=None):
    '''
    :param data_dict: {"image": (H,W,D), "label": 3D}
    '''
    image = data_dict["image"].squeeze()
    label = data_dict["label"].squeeze()

    d = image.shape[2]
    num_cols = 5
    num_rows = math.ceil(d / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    for z in range(d):
        row = z // num_cols
        col = z % num_cols
        title = f"Slice {z + 1}"
        if num_rows>1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        visualize_2D({"image": image[:, :, z],"label": label[:, :, z]},
                     ax_=ax,
                     title=title)

    # turn off unused subplots
    for z in range(d, num_rows * num_cols):
        row = z // num_cols
        col = z % num_cols
        axs[row, col].axis('off')

    if save_file_name is not None:
        plt.savefig(save_file_name)

    plt.tight_layout()
    plt.show()


def load_file_dict(patient_id, mode: str):
    if mode not in ['ED','ES']:
        raise ValueError("mode must be either ED or ES")

    num_img = "%03d" % patient_id

    # find from metadata which is ED frame and which is ES frame
    n_ED = df.loc[df['id'] == patient_id, 'ED'].iloc[0]
    n_ES = df.loc[df['id'] == patient_id, 'ES'].iloc[0]

    patient_mode = ['training', 'testing'][patient_id > 100]
    patient_dir = os.path.join(base_path, 'database', patient_mode, f'patient{num_img}')

    if mode == 'ED':
        file_dict = {"image": os.path.join(patient_dir, f"patient{num_img}_frame{n_ED:02d}.nii.gz"),
                     "label": os.path.join(patient_dir, f"patient{num_img}_frame{n_ED:02d}_gt.nii.gz")}

    else:
        file_dict = {"image": os.path.join(patient_dir, f"patient{num_img}_frame{n_ES:02d}.nii.gz"),
                     "label": os.path.join(patient_dir, f"patient{num_img}_frame{n_ES:02d}_gt.nii.gz")}

    return file_dict

def load_data_dict(patient_id, mode: str):
    file_dict = load_file_dict(patient_id, mode)
    data_dict = monai.transforms.LoadImageD(("image", "label"), image_only=True)(file_dict)
    return data_dict

#%%
# tmp = load_file_dict(101,'ED')
# tmp