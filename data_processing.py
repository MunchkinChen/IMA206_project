import torch
import monai


transform_train = monai.transforms.Compose([
    monai.transforms.LoadImageD(("image", "label")),
    # (232,256,10)
    monai.transforms.AddChannelD(("image", "label")),
    # (1,232,256,10)
    monai.transforms.RandSpatialCropD(keys=("image", "label"), roi_size=(128, 128, 1),
                                      random_center=True, random_size=False),
    # (1,128,128,1)
    monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
    monai.transforms.SqueezeDimd(keys=("image", "label"), dim=-1),
    # (1,128,128)
    monai.transforms.ToTensorD(("image", "label")),
])



transform_val = monai.transforms.Compose([
    monai.transforms.LoadImageD(("image", "label")),
    monai.transforms.AddChannelD(("image", "label")),
    monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
    monai.transforms.ToTensorD(("image", "label")),
])
#
# transform_test = monai.transforms.Compose([
#     monai.transforms.LoadImageD(("image", "label")),
#     monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
#     LeaveOutLV(inference=True),
#     monai.transforms.ToTensorD(("input")),
# ])



postprocess = monai.transforms.Compose([
    monai.transforms.AsDiscrete(argmax=True, to_onehot=4, threshold_values=False),
    monai.transforms.KeepLargestConnectedComponent(applied_labels=(1, 2, 3), independent=False, connectivity=None)
])











