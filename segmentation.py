# import sys
# base_path = "/home/infres/xchen-21/IMA205_challenge"
# sys.path.append(base_path)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from tqdm import tqdm
# import os
# import nibabel as nib
# from data_processing import *
# from unet import *
#
#
# #%%
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SAVE_UNET_PATH = os.path.join(base_path, r'saved_models/unet_weight.pt')
#
#
# #%% load model weights
# model = UNet(n_channels=3, n_classes=1).to(DEVICE)
# state_dict = torch.load(SAVE_UNET_PATH)
# model.load_state_dict(state_dict)
# print(f'Weights loaded from {SAVE_UNET_PATH}')
# model.eval()
#
#
# #%%
# test_ids = range(101, 151)
#
# print(f'predicting LV labels on test set')
# for ptid_ in tqdm(test_ids):
#     ptid = "%03d" % ptid_
#     pt_dir = f'{base_path}/data/Test/{ptid}'
#
#     for mode in ['ES', 'ED']:
#         img_path = os.path.join(pt_dir, f'{ptid}_{mode}.nii')
#         label_path = os.path.join(pt_dir, f'{ptid}_{mode}_seg.nii')
#
#         # path to save predicted results
#         pred_full_label_path = os.path.join(pt_dir, f'{ptid}_{mode}_seg_pred2.nii')
#
#         # load test batch
#         file_dict_test = {'image': img_path, 'label': label_path}
#         test_batch = transform_test(file_dict_test)
#
#         # do segmentation of LV
#         with torch.no_grad():
#             outputs_test = monai.inferers.sliding_window_inference(
#                 inputs=test_batch["input"].squeeze(0).permute(3, 0, 1, 2).to(DEVICE),
#                 predictor=model,
#                 roi_size=(128, 128),
#                 sw_batch_size=32)
#
#         outputs_test = outputs_test.permute(1, 2, 3, 0)
#         outputs_test = postprocess(outputs_test)
#
#         # show predicted LV mask and ground truth M mask
#         plt.imshow(outputs_test.detach().cpu().numpy()[0, :, :, 5] + \
#                    2*(test_batch['input'].squeeze()[2, :, :, 5]))
#         plt.colorbar()
#         plt.show()
#
#
#         # deduce fully segmented labels
#         part_label_img = nib.load(label_path)
#         part_label = np.asanyarray(part_label_img.dataobj)
#         pred_full_label = part_label.copy()
#
#         # only change initially background area to LV
#         pred_full_label[(part_label == 0) & (outputs_test.squeeze() == 1)] = 3
#         full_label_img = nib.Nifti1Image(pred_full_label, part_label_img.affine, part_label_img.header)
#         nib.save(full_label_img, pred_full_label_path)
#
#
#
#
