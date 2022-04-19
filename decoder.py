import torch

import torch.nn.functional as F
import numpy as np
from lib.hevc_feature_decoder import HevcFeatureReader
import torchvision.transforms as T

def decode_compressed_video(filename):

    reader = HevcFeatureReader(filename, nb_frames=None, n_parallel=4)
    num_frames = reader.getFrameNums()
    decode_order = reader.getDecodeOrder()
    width, height = reader.getShape()

    print(
        "[HEVC feature decoder]: Decode {}x{} video with {} frames.".format(
            width, height, num_frames
        )
    )

    rgb = []
    frame_type = []
    quadtree_stru = []
    mv_x_L0 = []
    mv_y_L0 = []
    mv_x_L1 = []
    mv_y_L1 = []
    ref_off_L0 = []
    ref_off_L1 = []
    residual = []
    i = 0
    for feature in reader.nextFrame():
        i += 1
        frame_type.append(feature[0])
        quadtree_stru.append(feature[1])
        rgb.append(feature[2])
        mv_x_L0.append(feature[3])
        mv_y_L0.append(feature[4])
        mv_x_L1.append(feature[5])
        mv_y_L1.append(feature[6])
        ref_off_L0.append(feature[7])
        ref_off_L1.append(feature[8])
        residual.append(feature[10])


    reader.close()
    frame_type = np.array(frame_type)

    # [0,1,2] 0:I-frame, 1:P-frame, 2:B-frame

    covos_features = {
        "nb_frames": num_frames,
        "frame_type": frame_type,
        "key_idx": np.where(frame_type != 2)[0],
        "non_key_idx": np.where(frame_type == 2)[0],
        "width": width,
        "height": height,
        "rgb": np.array(rgb),
        "residual": np.array(residual),
        "mv_x_L0": np.array(mv_x_L0),
        "mv_y_L0": np.array(mv_y_L0),
        "mv_x_L1": np.array(mv_x_L1),
        "mv_y_L1": np.array(mv_y_L1),
        "L0_ref": np.array(ref_off_L0),
        "L1_ref": np.array(ref_off_L1),
        "decode_order": decode_order,
    }

    return covos_features


def size_transform(cvf, gt_shape, dset):
    im_normalization = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    if dset == 'yt2018':
        im_transform = T.Compose(
            [
                im_normalization,
                T.Resize(480, interpolation=T.InterpolationMode.BICUBIC),
            ]
        )
        res_transform = T.Compose(
            [
                T.Resize(480, interpolation=T.InterpolationMode.BICUBIC),
            ]
        )
    else:
        im_transform = T.Compose(
            [   
                im_normalization,
                T.Resize(gt_shape),
            ]
        )
        res_transform = T.Compose(
            [
                T.Resize(gt_shape),
            ]
        )

    # B,T,C,H,W
    cvf["rgb_tensor"] = im_transform(
        torch.from_numpy(cvf["rgb"]).permute((0, 3, 1, 2)).contiguous().float().div(255)
    ).cuda()

    cvf["residual"] = res_transform(
        (torch.from_numpy(cvf["residual"]).permute((0, 3, 1, 2)).contiguous().float()).div(255)
    )
    cvf["residual"] = cvf["residual"]-0.5

    mv_shape = cvf["rgb_tensor"].shape[-2:]

    x_rescale = 0.25 * (mv_shape[0] / cvf["height"])
    y_rescale = 0.25 * (mv_shape[1] / cvf["width"])
    cvf["mv_x_L0"] = (
        torch.from_numpy(cvf["mv_x_L0"]).float().unsqueeze(0) * x_rescale
    )
    cvf["mv_y_L0"] = (
        torch.from_numpy(cvf["mv_y_L0"]).float().unsqueeze(0) * y_rescale
    )
    cvf["mv_x_L1"] = (
        torch.from_numpy(cvf["mv_x_L1"]).float().unsqueeze(0) * x_rescale
    )
    cvf["mv_y_L1"] = (
        torch.from_numpy(cvf["mv_y_L1"]).float().unsqueeze(0) * y_rescale
    )
    cvf["L0_ref"] = torch.from_numpy(cvf["L0_ref"]).float().unsqueeze(0)
    cvf["L1_ref"] = torch.from_numpy(cvf["L1_ref"]).float().unsqueeze(0)
    cvf["mv_x_L0"] = (
        F.interpolate(
            cvf["mv_x_L0"],
            size=(mv_shape),
            mode="nearest",
        )
        .float()
        .squeeze(0)
    )
    cvf["mv_y_L0"] = (
        F.interpolate(
            cvf["mv_y_L0"],
            size=(mv_shape),
            mode="nearest",
        )
        .float()
        .squeeze(0)
    )
    cvf["mv_x_L1"] = (
        F.interpolate(
            cvf["mv_x_L1"],
            size=(mv_shape),
            mode="nearest",
        )
        .float()
        .squeeze(0)
    )
    cvf["mv_y_L1"] = (
        F.interpolate(
            cvf["mv_y_L1"],
            size=(mv_shape),
            mode="nearest",
        )
        .float()
        .squeeze(0)
    )
    cvf["L0_ref"] = (
        F.interpolate(
            cvf["L0_ref"],
            size=(mv_shape),
            mode="nearest",
        )
        .int()
        .squeeze(0)
    )
    cvf["L1_ref"] = (
        F.interpolate(
            cvf["L1_ref"],
            size=(mv_shape),
            mode="nearest",
        )
        .int()
        .squeeze(0)
    )
    return cvf


def mv_pad(cvf, pad_array):

    cvf["mv_x_L0"] = F.pad(cvf["mv_x_L0"], pad_array)
    cvf["mv_y_L0"] = F.pad(cvf["mv_y_L0"], pad_array)
    cvf["mv_x_L1"] = F.pad(cvf["mv_x_L1"], pad_array)
    cvf["mv_y_L1"] = F.pad(cvf["mv_y_L1"], pad_array)
    cvf["L0_ref"] = F.pad(cvf["L0_ref"], pad_array).float()
    cvf["L1_ref"] = F.pad(cvf["L1_ref"], pad_array).float()

    cvf["rgb_tensor"] = F.pad(cvf["rgb_tensor"], pad_array)
    cvf["residual"] = F.pad(cvf["residual"], pad_array)

    cvf["residual"] = (
        F.interpolate(
            cvf["residual"],
            scale_factor=0.25,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )
        .float()
        .squeeze(0)
    )
    cvf["residual"] = torch.abs(cvf["residual"])

    cvf["residual"] = torch.sum(cvf["residual"], dim=1, keepdim=True)
    cvf["residual"] = cvf["residual"]>0.15
    cvf["residual"] = cvf["residual"].cuda()
    cvf["mv_x_L0"] = (
        F.interpolate(
            cvf["mv_x_L0"].unsqueeze(0),
            scale_factor=0.25,
            mode="nearest",
            recompute_scale_factor=False,
        )
        .float()
        .squeeze(0)
    ).cuda()
    cvf["mv_y_L0"] = (
        F.interpolate(
            cvf["mv_y_L0"].unsqueeze(0),
            scale_factor=0.25,
            mode="nearest",
            recompute_scale_factor=False,
        )
        .float()
        .squeeze(0)
    ).cuda()
    cvf["mv_x_L1"] = (
        F.interpolate(
            cvf["mv_x_L1"].unsqueeze(0),
            scale_factor=0.25,
            mode="nearest",
            recompute_scale_factor=False,
        )
        .float()
        .squeeze(0)
    ).cuda()
    cvf["mv_y_L1"] = (
        F.interpolate(
            cvf["mv_y_L1"].unsqueeze(0),
            scale_factor=0.25,
            mode="nearest",
            recompute_scale_factor=False,
        )
        .float()
        .squeeze(0)
    ).cuda()
    cvf["L0_ref"] = (
        F.interpolate(
            cvf["L0_ref"].unsqueeze(0),
            scale_factor=0.25,
            mode="nearest",
            recompute_scale_factor=False,
        )
        .int()
        .squeeze(0)
    ).cuda()
    cvf["L1_ref"] = (
        F.interpolate(
            cvf["L1_ref"].unsqueeze(0),
            scale_factor=0.25,
            mode="nearest",
            recompute_scale_factor=False,
        )
        .int()
        .squeeze(0)
    ).cuda()


    cvf["mv_x_L0"] *= 0.25
    cvf["mv_y_L0"] *= 0.25
    cvf["mv_x_L1"] *= 0.25
    cvf["mv_y_L1"] *= 0.25

    return cvf
