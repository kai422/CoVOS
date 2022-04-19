from skvideo.utils import first
from lib.hevc_feature_decoder_res import HevcFeatureReader
import time
import numpy as np
import cv2
import os
import PIL.Image as Image
import argparse
import json
import tqdm

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
    residual = []
    i=0
    for feature in reader.nextFrame():
        i+=1
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
        residual.append(feature[10])



    decode_time = (time.time() - timeStarted)/num_frames
    print("[HEVC feature decoder]: Decode {}x{} video at FPS {:.2f}.".format(width, height, 1/decode_time))

    reader.close()



    covos_features = {
    'nb_frames': num_frames,
    'width': width,
    'height': height,
    'rgb' : np.array(rgb),
    'residual' : np.array(residual),
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CoVOS pipeline')

    parser.add_argument('--dset', type=str, choices=["yt2018", "dv2017", "dv2016"])
    parser.add_argument('--split', type=str, default = 'train')
    parser.add_argument('--dset_path', type=str, help='path for the video to be decoded and segmented')
    parser.add_argument('--save_path', type=str, help='path for segmentation results')

    args = parser.parse_args()

    if args.dset in ['yt2018']:
        save_path = args.save_path 
        video_folder = os.path.join(args.dset_path, 'HEVCVideos')
        meta = json.load(open(os.path.join(args.dset_path, args.split, 'meta.json')))
        
        video_list = list(meta['videos'])

        for i, video_name in enumerate(video_list):
            print('Generating {}/{} video: {}'.format(i, len(video_list)-1, video_name))
            #if os.path.exists(os.path.join(save_path, video_name)):
            #    print('video folder already exist.')
            #    continue
            first_frame = sorted(os.listdir(os.path.join(args.dset_path, 'all_frames/{}_all_frames/JPEGImages'.format(args.split), video_name)))[0][:-4]
            video_path = os.path.join(video_folder, video_name + '_' + first_frame + '.mp4')

            c = read_compressed_features(video_path)

    

            os.makedirs(os.path.join(save_path, video_name, 'rgb'), exist_ok = True)
            os.makedirs(os.path.join(save_path, video_name, 'residual'), exist_ok = True)
            os.makedirs(os.path.join(save_path, video_name, 'bits'), exist_ok = True)
            os.makedirs(os.path.join(save_path, video_name, 'quadtrees'), exist_ok = True)

            for f in range(c['nb_frames']):
                frame_type = c['frame_type'][f]
                #rgb = c['rgb'][f]
                residual = c['residual'][f]
                #bit_density = c['bit_density'][f]
                #quadtree_stru = c['quadtree_stru'][f]
                #cv2.imwrite(os.path.join(save_path, video_name, 'rgb', '{:05d}_{}.jpg'.format(f+int(first_frame), frame_type)), rgb)
                cv2.imwrite(os.path.join(save_path, video_name, 'residual', '{:05d}_{}.jpg'.format(f+int(first_frame), frame_type)), residual)
                #cv2.imwrite(os.path.join(save_path, video_name, 'bits', '{:05d}_{}.jpg'.format(f+int(first_frame), frame_type)), bit_density)
                #np.save(os.path.join(save_path, video_name, 'quadtrees', '{:05d}_{}.npy'.format(f+int(first_frame), frame_type)), quadtree_stru)


    if args.dset in ['dv2016', 'dv2017']:
        save_path = args.save_path 
        first_frame = 0
        video_folder = os.path.join(args.dset_path, 'HEVCVideos')
        if args.dset == 'dv2016':
            list_path = 'ImageSets/2016/{}.txt'.format(args.split)
        else:
            list_path = 'ImageSets/2017/{}.txt'.format(args.split)

        video_list = [line.rstrip('\n') for line in open(os.path.join(args.dset_path, list_path))]

        for i, video_name in enumerate(video_list):
            print('Generating {}/{} video: {}'.format(i, len(video_list)-1, video_name))
            video_path = os.path.join(video_folder, video_name + '.mp4')
            #if os.path.exists(os.path.join(save_path, video_name)):
            #    print('video folder already exist.')
            #    continue
            c = read_compressed_features(video_path)

            os.makedirs(os.path.join(save_path, video_name, 'rgb'), exist_ok = True)
            os.makedirs(os.path.join(save_path, video_name, 'residual'), exist_ok = True)
            os.makedirs(os.path.join(save_path, video_name, 'bits'), exist_ok = True)
            os.makedirs(os.path.join(save_path, video_name, 'quadtrees'), exist_ok = True)

            for f in range(c['nb_frames']):
                frame_type = c['frame_type'][f]
                rgb = c['rgb'][f]
                residual = c['residual'][f]
                bit_density = c['bit_density'][f]
                quadtree_stru = c['quadtree_stru'][f]
                cv2.imwrite(os.path.join(save_path, video_name, 'rgb', '{:05d}_{}.jpg'.format(f+int(first_frame), frame_type)), rgb)
                cv2.imwrite(os.path.join(save_path, video_name, 'residual', '{:05d}_{}.jpg'.format(f+int(first_frame), frame_type)), residual)
                cv2.imwrite(os.path.join(save_path, video_name, 'bits', '{:05d}_{}.jpg'.format(f+int(first_frame), frame_type)), bit_density)
                np.save(os.path.join(save_path, video_name, 'quadtrees', '{:05d}_{}.npy'.format(f+int(first_frame), frame_type)), quadtree_stru)


