import os
import json
import torch
import argparse
import numpy as np

from utils import *
from segmentor import *
from propagator import Propagator
from path_config import path_config
from decoder import *
from model import RGBEncoder


def main(args):
    torch.set_grad_enabled(False)
    base_segmentor = get_segmentor(args.base_segmentor)(args.cfg_dict)
    feature_extractor = RGBEncoder("weights/covos_light_encoder.pth").cuda()
    feature_extractor.eval()
    
    propagator = (
        Propagator(
            model_path="weights/covos_propagator.pth"
        )
        .cuda()
        .eval()
    )

    segment_t_all, prop_t_all, light_encoder_t_all= 0, 0, 0
    total_frames = 0
    total_keyframes = 0
    warmup_t = 3
    data_path = os.path.join(path_config.data_path(), "DAVIS")

    assert args.dset in ["dv2016", "dv2017"]
    sample = Image.open(data_path + "/Annotations/480p/blackswan/00000.png")
    palette = sample.getpalette()
    video_folder = os.path.join(data_path, "HEVCVideos")

    if args.dset == "dv2016":
        val_list = [
            line.rstrip("\n")
            for line in open(os.path.join(data_path, "ImageSets/2016/val.txt"))
        ]
        base_segmentor.build_dataset(single_object=True)
    else:
        val_list = [
            line.rstrip("\n")
            for line in open(os.path.join(data_path, "ImageSets/2017/val.txt"))
        ]
        base_segmentor.build_dataset(single_object=False)

    save_path = os.path.join(
        os.path.join(args.save_path, args.dset)
    )

    for i, v in enumerate(val_list[0:warmup_t]+val_list):
        print('Evaluating {}/{} video: {}'.format(i, len(val_list[0:warmup_t]+val_list)-1, v))
        video = os.path.join(video_folder, v + ".mp4")
        label = os.path.join(data_path, "Annotations/480p/", v)
        mask_results, info = process(
            video,
            base_segmentor,
            propagator,
            label,
            args.dset,
            feature_extractor
        )
        
        if i < warmup_t:
            pass
        else:
            num_frames = info["nb_frames"]
            total_frames += num_frames
            total_keyframes += len(info['key_idx'])
            segment_t_all += info["seg_t"]
            prop_t_all += info["prop_t"]
            light_encoder_t_all += info["light_encoder_t"]

        save_format_result(palette, mask_results, v, 0, save_path)
        torch.cuda.empty_cache()
    
    base_t = segment_t_all/total_keyframes
    prop_t = prop_t_all/(total_frames-total_keyframes)

    print("*"*150)
    print(f"Base Model FPS: {1/base_t:.2f}")
    print(f"Propagation FPS: {1/prop_t:.2f}")
    print(f"Keyframe Ratio: {total_keyframes / total_frames:.2f}")
    print(f"Overall FPS: {total_frames/(segment_t_all + prop_t_all + light_encoder_t_all) :.2f}")

    



def get_mask_shape(mask_file_path):
    mask_file = os.path.join(mask_file_path, "00000.png")
    masks = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    return masks.shape[-2:]

def process(
    video, segmentor, propagator, mask_folder, dset, feature_extractor,
):
    gt_shape = get_mask_shape(mask_folder)

    cvf = decode_compressed_video(video)  # cvf: compressed video features

    cvf = size_transform(cvf, gt_shape, dset)

    key_perc = cvf["key_idx"].size / cvf["nb_frames"] * 100
    print(f"Keyframe percentage: {key_perc:.2f}%")

    # get prediction and low level feature from segmentor

    key_masks, key_prob4, key_feat, pad, seg_t, light_encoder_t = segmentor.inference(
        cvf["rgb"], cvf["rgb_tensor"], cvf["key_idx"], mask_folder, feature_extractor
    )
    cvf = mv_pad(cvf, pad)

    # propagate prediction.
    all_mask, prop_t = propagator.propagate(
        key_masks,
        key_prob4,
        key_feat,
        cvf,
        pad,
        gt_shape,
        feature_extractor,
    )

    info = {
        "nb_frames": cvf["nb_frames"],
        "seg_t": seg_t,
        "prop_t": prop_t,
        "light_encoder_t": light_encoder_t,
        "key_idx": cvf["key_idx"].tolist()
    }
    return all_mask, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoVOS pipeline")

    parser.add_argument("--dset", type=str, choices=["yt2018", "dv2017", "dv2016"])
    parser.add_argument(
        "--base_segmentor",
        type=str,
        help="base vos method used for keyframe segmentation. If you want to try with your own base segmentor, you should implement coresponding method in base_segmenter.py",
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_dict",
        action=StoreDictKeyPair,
        nargs="+",
        metavar="KEY=VAL",
        help="config for your base segmentor.",
    )
    parser.add_argument("--save_path", type=str, help="path for segmentation results")
    args = parser.parse_args()
    main(args)
