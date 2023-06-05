import sys
base_path = "/home/infres/xchen-21/IMA206_project"
sys.path.append(base_path)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import torch
import monai
from unet import UNet
from data_processing import *

#%%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_UNET_PATH = os.path.join(base_path,r'saved_models/unet_random_crop.pt')
VISUALIZE_ID = 1 # select a patient id to visualize segmentation results
df = pd.read_csv('database/metadata.csv')
#%%
def visualize_data(data_dict, save_file_name=None):
    '''
    visualize MDR data slice by slice

    :param data_dict: loaded data dictionary
    :param save_file_name: save visualize file to directory
    '''
    image = data_dict["image"].squeeze()
    label = data_dict["label"].squeeze()

    # color map for labels: 1-Red 2-Green 3-Blue
    cmap = ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)])

    plt.figure(figsize=(10, 8))
    d = image.shape[2]
    for z in range(d):
        plt.subplot(int(np.sqrt(d))+1, int(np.sqrt(d))+1, z+1)
        plt.imshow(image[:, :, z], cmap='gray')
        plt.imshow(label[:, :, z], cmap=cmap)
        plt.axis('off')
        plt.title('Slice {}'.format(z + 1))

    if save_file_name is not None:
        plt.savefig(save_file_name)

    plt.show()



#%% visualize some training data, with gt label
num = 101
num_img = "%03d" % num

# find from metadata which is ED frame and which is ES frame
n_ED = df.loc[df['id'] == num, 'ED'].iloc[0]
n_ES = df.loc[df['id'] == num, 'ES'].iloc[0]

patient_mode = ['training', 'testing'][num > 100]
patient_dir = os.path.join(base_path, 'database', patient_mode, f'patient{num_img}')

# visualize ED frame
file_dict = {"image": os.path.join(patient_dir, f"patient{num_img}_frame{n_ED:02d}.nii.gz"),
             "label": os.path.join(patient_dir, f"patient{num_img}_frame{n_ED:02d}_gt.nii.gz")}
data_dict = monai.transforms.LoadImageD(("image", "label"))(file_dict)
visualize_data(data_dict)

# visualize ES frame
file_dict = {"image": os.path.join(patient_dir, f"patient{num_img}_frame{n_ES:02d}.nii.gz"),
             "label": os.path.join(patient_dir, f"patient{num_img}_frame{n_ES:02d}_gt.nii.gz")}
data_dict = monai.transforms.LoadImageD(("image", "label"))(file_dict)
visualize_data(data_dict)





#%% load model weights
model = UNet(n_channels=1, n_classes=4).to(DEVICE)
state_dict = torch.load(SAVE_UNET_PATH)
model.load_state_dict(state_dict)
print(f'Weights loaded from {SAVE_UNET_PATH}')
model.eval()


#%% run model on a test sample
val_batch = transform_val(file_dict)


# plt.imshow(val_batch["input"].detach().cpu().numpy()[0, :, :, 5],cmap='gray')
# plt.imshow(val_batch["input"].detach().cpu().numpy()[1, :, :, 5],cmap=ListedColormap([(1, 1, 1, 0), (1, 0, 0, 0.5)]))
# plt.imshow(val_batch["input"].detach().cpu().numpy()[2, :, :, 5],cmap=ListedColormap([(1, 1, 1, 0), (0, 1, 0, 0.5)]))
# # plt.imshow(val_batch["label"].detach().cpu().numpy()[0, :, :, 5])
# plt.title('input channel all')
# plt.savefig(os.path.join(base_path,'images/input_channel_all.png'))
# plt.axis('off')
# plt.show()

#% visualize model output
outputs_val = monai.inferers.sliding_window_inference(
    inputs=val_batch["input"].squeeze(0).permute(3, 0, 1, 2).to(DEVICE),
    predictor=model,
    roi_size=(128, 128),
    sw_batch_size=32)
outputs_val = outputs_val.permute(1, 2, 3, 0)  # (1, H, W, D) a probability distribution for LV class (not normalized)

plt.imshow(outputs_val.detach().cpu().numpy()[0, :, :, 5])
plt.title('unet output')
plt.savefig(os.path.join(base_path,'images/unet_output.png'))
plt.colorbar()
plt.show()


#% visualize predicted mask
outputs_val = postprocess(outputs_val)  # (1, H, W, D) a 0-1 mask for LV class
plt.imshow(outputs_val.detach().cpu().numpy()[0, :, :, 5])
plt.title('predicted mask')
plt.savefig(os.path.join(base_path,'images/predicted_mask.png'))
plt.colorbar()
plt.show()


#% metric on prediction of validation sample
y_pred = outputs_val.unsqueeze(0).cpu()  # (1, 1, H, W, D)
y = val_batch["label"].unsqueeze(0)

dice_metric = monai.metrics.DiceMetric()
dsc = dice_metric(y_pred, y)

hd_metric = monai.metrics.HausdorffDistanceMetric()
hd = hd_metric(y_pred, y)
print(f'metric on sample: DSC {dsc.mean().numpy():.3f}, HD {hd.mean().numpy():.3f}')


#%% visualize ground truth and predicted results on MDR image
result = {"image": val_batch["input"][0, :, :, :].squeeze(),
          "label": outputs_val.squeeze().cpu()}
print("Predicted mask of class 3")
visualize_data(result)

gt = {"image": val_batch["input"][0, :, :, :].squeeze(),
      "label": val_batch["label"].squeeze()}
print("Ground truth of class 3")
visualize_data(gt)