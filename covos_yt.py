import os
import json
import torch
import argparse
import numpy as np

from utils import *
from segmentor import *
from propagator_yt import Propagator
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
    data_path = os.path.join(path_config.data_path(), 'YouTube-VOS')
    assert args.dset in ['yt2018']
    
    base_segmentor.build_dataset()
    
    
    save_path = os.path.join(
        os.path.join(args.save_path, args.dset)
        )

    palette = Image.open(
        data_path + '/valid/Annotations/0062f687f1/00000.png').getpalette()
    video_folder = os.path.join(data_path, 'HEVCVideos')
    meta = json.load(
        open(os.path.join(data_path, 'valid/meta.json')))
    val_list = list(meta['videos'])

    val_list = val_list[:30]
    for i, v in enumerate(val_list[0:warmup_t]+val_list):
        print('Evaluating {}/{} video: {}'.format(i,
                                                    len(val_list[0:warmup_t]+val_list)-1, v))
        if os.path.exists(os.path.join(save_path, v)) and warmup_t!=0:
           print('Prediction already exist.')
           continue
        first_frame = sorted(os.listdir(os.path.join(
            data_path, 'all_frames/valid_all_frames/JPEGImages', v)))[0][:-4]
        video = os.path.join(
            video_folder, v + '_' + first_frame + '.mp4')
        label = os.path.join(
            data_path, 'valid/Annotations/', v)
        
        mask_results, info = process(
            video,
            base_segmentor,
            propagator,
            label,
            args.dset,
            feature_extractor,
            base_index=first_frame,
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
  
        save_format_result(palette, mask_results, v, first_frame, save_path)
        torch.cuda.empty_cache()

    base_t = segment_t_all/total_keyframes
    prop_t = prop_t_all/(total_frames-total_keyframes)

    print("*"*150)
    print(f"Base Model FPS: {1/base_t:.2f}")
    print(f"Propagation FPS: {1/prop_t:.2f}")
    print(f"Keyframe Ratio: {total_keyframes / total_frames:.2f}")
    print(f"Overall FPS: {total_frames/(segment_t_all + prop_t_all + light_encoder_t_all) :.2f}")
    generate_submit_samples(os.path.join(data_path, "valid/JPEGImages"), save_path)

def process(
    video, segmentor, propagator, mask_folder, dset, feature_extractor, base_index=0
):

    cvf  = decode_compressed_video(
        video
    )  # cvf: compressed video features

    cvf = size_transform(cvf, None, dset)

    # get prediction and low level feature from segmentor

    key_idxs, key_masks, key_prob4, key_feat, size, label_backward, num_obj, pad, seg_t, light_encoder_t = segmentor.inference(
        cvf["rgb"], cvf["rgb_tensor"], cvf["key_idx"], mask_folder, feature_extractor=feature_extractor, base_index=base_index
    )
    cvf = mv_pad(cvf, pad)

    key_perc = len(key_idxs) / cvf["nb_frames"] * 100
    print(f"Keyframe percentage: {key_perc:.2f}%")

    # propagate prediction.
    all_mask, prop_t = propagator.propagate(
        key_idxs,
        key_masks,
        key_prob4,
        key_feat,
        size,
        label_backward,
        num_obj, 
        pad,
        cvf,
        feature_extractor,
    )

    info = {
        "nb_frames": cvf["nb_frames"],
        "seg_t": seg_t,
        "prop_t": prop_t,
        "key_idx": key_idxs.tolist(),
        "light_encoder_t":light_encoder_t
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
