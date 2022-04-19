import os
import cv2
import time
import json
import numpy as np
import argparse

from PIL import Image
from torch import tensor
from pathlib import Path
from lib.segmentors import *
from lib.raw_mv_warp_func_bilinear import raw_mv_warp
from lib.raw_mv_warp_func_rgb import raw_mv_warp_rgb
from lib.raw_mv_warp_func_mask import raw_mv_warp_mask

from lib.hevc_feature_decoder import HevcFeatureReader


def main(args):
    base_segmentor = get_segmentor(args.base_segmentor)(args.cfg_dict)
    base_segmentor.build_segmentor()

    if args.dset in ['dv2016', 'dv2017']:
        sample = Image.open(args.dset_path + '/Annotations/480p/blackswan/00000.png')
        palette = sample.getpalette()
        shape = sample.size
        video_folder = os.path.join(args.dset_path, 'HEVCVideos')

        if args.dset == 'dv2016':
            list_path = 'ImageSets/2016/{}.txt'.format(args.split)
            base_segmentor.build_dataset(resolution='full', single_object=True)
        else:
            list_path = 'ImageSets/2017/{}.txt'.format(args.split)
            base_segmentor.build_dataset(resolution='full', single_object=False)

        video_list = [line.rstrip('\n') for line in open(os.path.join(args.dset_path, list_path))]
        
        for i, video_name in enumerate(video_list):
            print('Generating {}/{} video: {}'.format(i, len(video_list)-1, video_name))
            semi_superived_label = os.path.join(args.dset_path, 'Annotations/480p/', video_name)
            warped_Es_result, segmented_result = segment_and_warp(os.path.join(video_folder, video_name + '.mp4'), 
            base_segmentor, semi_superived_label)
            if args.base_segmentor=='davis_oracle':
                split_prefix = 'from_gt_'
            else:
                split_prefix = ''
            warped_Es_save_path = os.path.join(args.dset_path, 'HEVCfeatures', split_prefix+args.split, args.dset,  video_name, 'warped_Es_results') #as npz
            warped_save_path = os.path.join(args.dset_path, 'HEVCfeatures', split_prefix+args.split, args.dset,  video_name, 'warped_results') #as png
            segmented_save_path = os.path.join(args.dset_path, 'HEVCfeatures', split_prefix+args.split, args.dset, video_name, 'segmented_results') #as png

            save_format_result(palette, 0, warped_Es_result, segmented_result, warped_Es_save_path, warped_save_path, segmented_save_path)
        

        

    if args.dset in ['yt2018']:
        base_segmentor.build_dataset()
        palette = Image.open(args.dset_path + '/valid/Annotations/0062f687f1/00000.png').getpalette()
        video_folder = os.path.join(args.dset_path, 'HEVCVideos')
        meta = json.load(open(os.path.join(args.dset_path, args.split, 'meta.json')))
        
        video_list = list(meta['videos'])

        for i, video_name in enumerate(video_list):
            print('Evaluating {}/{} video: {}'.format(i, len(video_list)-1, video_name))

            first_frame = sorted(os.listdir(os.path.join(args.dset_path, 'all_frames/{}_all_frames/JPEGImages'.format(args.split), video_name)))[0][:-4]
            video_path = os.path.join(video_folder, video_name + '_' + first_frame + '.mp4')
            semi_superived_label = os.path.join(args.dset_path, '{}/Annotations/'.format(args.split), video_name)

            warped_Es_result, segmented_result = segment_and_warp(video_path, base_segmentor, semi_superived_label, base_index=first_frame)

            warped_Es_save_path = os.path.join(args.dset_path, 'HEVCfeatures', args.split, args.dset,  video_name, 'warped_Es_results') #as npz
            warped_save_path = os.path.join(args.dset_path, 'HEVCfeatures', args.split, args.dset,  video_name, 'warped_results') #as png
            segmented_save_path = os.path.join(args.dset_path, 'HEVCfeatures', args.split, args.dset, video_name, 'segmented_results') #as 
            save_format_result(palette, first_frame, warped_Es_result, segmented_result, warped_Es_save_path, warped_save_path, segmented_save_path)
        


        
def save_format_result(palette, first_frame_index, warped_Es_result, segmented_result, warped_Es_save_path, warped_save_path, segmented_save_path):
    os.makedirs(warped_Es_save_path, exist_ok=True)
    os.makedirs(warped_save_path, exist_ok=True)
    os.makedirs(segmented_save_path, exist_ok=True)

    if warped_Es_result.shape[1]==2:
        warped_result = (warped_Es_result[:, 1:2] > 0.5).squeeze().astype(np.uint8)
        
    elif warped_Es_result.shape[1]>2: 
        warped_result = np.argmax(warped_Es_result, axis=1).astype(np.uint8)

    for f in range(len(warped_Es_save_path)):
        output = warped_Es_save_path[f]
        np.save(os.path.join(warped_Es_save_path, '{:05d}'.format(f+int(first_frame_index))), output)

    for f in range(len(warped_result)):
        output = warped_result[f]
        img = Image.fromarray(output)
        img.putpalette(palette)
        img.save(os.path.join(warped_save_path, '{:05d}.png'.format(f+int(first_frame_index))))
    
    for f in range(len(segmented_result)):
        output = segmented_result[f]
        img = Image.fromarray(output)
        img.putpalette(palette)
        img.save(os.path.join(segmented_save_path, '{:05d}.png'.format(f+int(first_frame_index))))

def read_compressed_features(input_mp4_name):

    timeStarted = time.time()    
    reader = HevcFeatureReader(input_mp4_name, nb_frames=None, n_parallel = 8)
    num_frames = reader.getFrameNums()
    decode_order = reader.getDecodeOrder()
    width, height = reader.getShape()

    rgb = []
    frame_type = [] 
    quadtree_stru = [] 
    mv_x_L0 = []
    mv_y_L0 = []
    mv_x_L1 = []
    mv_y_L1 = []
    ref_off_L0 = []
    ref_off_L1 = []
    bit_density = []

    for feature in reader.nextFrame():
        frame_type.append(feature[0])
        quadtree_stru.append(feature[1])
        rgb.append(feature[2])
        mv_x_L0.append(feature[3])
        mv_y_L0.append(feature[4])
        mv_x_L1.append(feature[5])
        mv_y_L1.append(feature[6])
        ref_off_L0.append(feature[7])
        ref_off_L1.append(feature[8])
        bit_density.append(feature[9])

    decode_time = (time.time() - timeStarted)/num_frames
    print("[HEVC feature decoder]: Decode {}x{} video ({} frames) at FPS {:.2f}.".format(width, height, num_frames, 1/decode_time))


    covos_features = {
    'nb_frames': num_frames,
    'width': width,
    'height': height,
    'rgb' : np.array(rgb),
    'frame_type' : np.array(frame_type),    # [0,1,2] 0:I-frame, 1:P-frame, 2:B-frame
    'quadtree_stru' : np.array(quadtree_stru),
    'mv_x_L0' : np.array(mv_x_L0),
    'mv_y_L0' : np.array(mv_y_L0),
    'mv_x_L1' : np.array(mv_x_L1),
    'mv_y_L1' : np.array(mv_y_L1),
    'ref_off_L0' : np.array(ref_off_L0),
    'ref_off_L1' : np.array(ref_off_L1),
    'bit_density' : np.array(bit_density),
    'decode_order' : decode_order,
    }

    

    return covos_features


def mask_Es_warp_rawMV(masks_Es, covos_features, predected_frame_index):
    warp_time = AverageMeter()
    #masks_Es = np.ascontiguousarray(masks_Es)

    for pos in tqdm.tqdm(covos_features['decode_order'], desc='Motion Vector Warping'):
        if pos not in predected_frame_index:
            start = time.time() 
            reconstructed = raw_mv_warp(masks_Es, covos_features['mv_x_L0'][pos], covos_features['mv_y_L0'][pos], covos_features['mv_x_L1'][pos], covos_features['mv_y_L1'][pos], covos_features['ref_off_L0'][pos], covos_features['ref_off_L1'][pos], pos)
            masks_Es[pos]=reconstructed
            warp_time.update(time.time()-start)
    warp_fps = 1/warp_time.avg
    #warp_fps =1
    print("Warp video at FPS {:.2f}.".format(warp_fps))

    return masks_Es


def segment_and_warp(input_mp4_name, segmentor, first_frame_label, base_index=None, log_ffmpeg_baseline_decode_time=False, enable_residual_repair_model=False, warp_estimation = True, warp_prediction = False, test_rgb_warp=False, only_generate_meta_json = False):

    covos_features = read_compressed_features(input_mp4_name)

    key_frame_indexes = np.where(covos_features['frame_type']!=2)[0]
    all_frame_indexed = np.arange(covos_features['nb_frames'])

    mask_width, mask_height = Image.open(next(Path(first_frame_label).glob('*.png'))).size

    _, key_warped_Es_result, predected_frame_index, object_ids, _ = segmentor.inference(covos_features['rgb'], key_frame_indexes, first_frame_label, base_index = base_index)

    segmented_result, _, _, object_ids, _ = segmentor.inference(covos_features['rgb'], all_frame_indexed, first_frame_label, mem_freq=5, base_index = base_index)



    key_frames_masks_Es = F.interpolate(key_warped_Es_result, (covos_features['height']>>2, covos_features['width']>>2), mode='bilinear', align_corners=False).permute(0,2,3,1)
    masks_Es= key_frames_masks_Es.new_zeros((covos_features['nb_frames'],)+key_frames_masks_Es.shape[1:])
    masks_Es[predected_frame_index]=key_frames_masks_Es

    warped_Es_result = mask_Es_warp_rawMV(masks_Es.cpu().numpy(), covos_features, predected_frame_index)
    warped_Es_result = F.interpolate(torch.from_numpy(warped_Es_result).permute(0,3,1,2), (mask_height,mask_width), mode='bilinear', align_corners=False).numpy()

    return warped_Es_result, segmented_result
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class StoreDictKeyPair(argparse.Action):
     def __init__(self, option_strings, dest, nargs=None, **kwargs):
         self._nargs = nargs
         super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values:
             k,v = kv.split("=")
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoVOS pipeline')

    parser.add_argument('--single_video_inference', action='store_true', help='enable single video inference')
    parser.add_argument('--single_video_path', type=str, help='path for the video to be decoded and segmented')
    parser.add_argument('--first_label', type=str, help='path for the first frame label, required by semi-supervised vos')

    parser.add_argument('--dset', type=str, choices=["yt2018", "dv2017", "dv2016"])
    parser.add_argument('--split', type=str, default = 'train')
    parser.add_argument('--dset_path', type=str, help='path for the video to be decoded and segmented')

    parser.add_argument('--base_segmentor', type=str, help='base vos method used for keyframe segmentation. If you want to try with your own base segmentor, you should implement coresponding method in base_segmenter.py')
    parser.add_argument("--cfg", dest="cfg_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL")

    args = parser.parse_args()
    main(args)