import sys
base_path = "/home/infres/xchen-21/IMA206_project"
sys.path.append(base_path)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from unet import UNet
import pandas as pd
from data_processing import *

#%%
VAL_SIZE = 15 # set at 0 if train on the entire train set with no validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-2
EPOCH = 50
SAVE_UNET_PATH = os.path.join(base_path, r'saved_models/unet_tmp.pt')
df = pd.read_csv('database/metadata.csv')

#%%
# randomly split validation set
val_ids = set(random.sample(range(1, 101), VAL_SIZE))

file_dict_train = []
file_dict_val = []

#%%
for ptid_ in range(1, 101):
    ptid = "%03d" % ptid_

    n_ED = df.loc[df['id'] == ptid_, 'ED'].iloc[0]
    n_ES = df.loc[df['id'] == ptid_, 'ES'].iloc[0]

    patient_mode = ['training', 'testing'][ptid_ > 100]
    patient_dir = os.path.join(base_path, 'database', patient_mode, f'patient{ptid}')

    file_dict_ED = {"image": os.path.join(patient_dir, f"patient{ptid}_frame{n_ED:02d}.nii.gz"),
                    "label": os.path.join(patient_dir, f"patient{ptid}_frame{n_ED:02d}_gt.nii.gz")}

    file_dict_ES = {"image": os.path.join(patient_dir, f"patient{ptid}_frame{n_ES:02d}.nii.gz"),
                    "label": os.path.join(patient_dir, f"patient{ptid}_frame{n_ES:02d}_gt.nii.gz")}

    if ptid_ in val_ids:
        file_dict_val.append(file_dict_ED)
        file_dict_val.append(file_dict_ES)
    else:
        file_dict_train.append(file_dict_ED)
        file_dict_train.append(file_dict_ES)

#%% preprocess input
dataset_train = monai.data.Dataset(data=file_dict_train, transform=transform_train)
dataloader_train = monai.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
print(f'Train set contains {len(dataset_train)} elements')


dataset_val = monai.data.Dataset(data=file_dict_val, transform=transform_val)
dataloader_val = monai.data.DataLoader(dataset_val, batch_size=1, shuffle=False)
print(f'Validation set contains {len(dataset_val)} elements')


#%% define model, loss function and optimizer
model = UNet(n_channels=1, n_classes=4).to(DEVICE)
print(model)

loss_function = monai.losses.DiceLoss(softmax=True,to_onehot_y=True, batch=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% train model
model.train()

training_losses = []
for epoch in range(EPOCH):
    epoch_loss = 0

    for batch_data in dataloader_train:
        input = batch_data["image"]
        label = batch_data["label"]

        optimizer.zero_grad()

        outputs = model(input.to(DEVICE))
        loss = loss_function(outputs, label.to(DEVICE))
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    training_losses.append(epoch_loss / len(dataloader_train))
    print(f'\nepoch {epoch}/{EPOCH} training loss: {epoch_loss / len(dataloader_train)}')

#%%  plot train loss
plt.plot(training_losses)
plt.title('training losses')
plt.xlabel('n_epoch')
# plt.savefig(os.path.join(base_path,'images/train_loss.png'))
plt.show()

#%% save model
tmp_path = os.path.join(base_path,'saved_models/unet_random_crop.pt')
torch.save(model.state_dict(), tmp_path)
print(f'Trained model weights saved at {tmp_path}')

#%% reload from saved weights (optional)
# state_dict = torch.load(SAVE_UNET_PATH)
# model.load_state_dict(state_dict)
# print(f'Weights loaded from {SAVE_UNET_PATH}')
#%%

#%% evaluate metrics over validation set


if VAL_SIZE > 0:
    print("Evaluating DSC and HD metrics over validation set")

    model.eval()

    # store HD and DSC by class
    hds = np.zeros((VAL_SIZE*2, 3))
    dscs = np.zeros((VAL_SIZE*2, 3))

    for i,val_batch in enumerate(dataloader_val):
        val_batch_input = val_batch["image"].squeeze(0).permute(3, 0, 1, 2).to(DEVICE)
        with torch.no_grad():
            outputs_val = monai.inferers.sliding_window_inference(
                inputs=val_batch_input,
                predictor=model,
                roi_size=(128, 128),
                sw_batch_size=32)
        outputs_val = outputs_val.permute(1, 2, 3, 0) # (4, H, W, D)
        outputs_val = postprocess(outputs_val)  # (4, H, W, D) a 0-1 mask for 4 classes

        y_pred = outputs_val.unsqueeze(0).cpu()  # (1, 4, H, W, D)
        y_gt = monai.networks.utils.one_hot(val_batch["label"],num_classes=4)

        dice_metric = monai.metrics.DiceMetric()
        # exclude background
        dsc = dice_metric(y_pred[:, 1:, :, :, :], y_gt[:, 1:, :, :, :])

        hd_metric = monai.metrics.HausdorffDistanceMetric()
        hd = hd_metric(y_pred, y_gt)

        hds[i] = hd.numpy().reshape(-1)
        dscs[i] = dsc.numpy().reshape(-1)

    # TODO: show avg by class
    print(f'Average metrics over validation set: DSC {dscs.mean():.3f}, HD {hds.mean():.3f}')