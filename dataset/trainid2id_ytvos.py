#generate ytvos laber conversion dictionary

import os
import json
import numpy as np
from PIL import Image
import os.path as path

dset_path = '/mnt/ssd/allusers/kai422/dataset_VOS/YT_VOS/ytvos2018'
meta = json.load(open(os.path.join(dset_path, 'valid/meta.json')))
val_list = list(meta['videos'])

for i, video_name in enumerate(val_list):
    print('Evaluating {}/{} video: {}'.format(i, len(val_list)-1, video_name))
    vid_gt_path = os.path.join(dset_path, 'valid/Annotations/', video_name)
    image_dir = path.join(dset_path, 'all_frames', 'valid_all_frames', 'JPEGImages', video_name)

    info = {}
    info['num_objects'] = 0
    first_mask = os.listdir(vid_gt_path)[0]
    _mask = np.array(Image.open(os.path.join(vid_gt_path, first_mask)).convert("P"))
    size = np.shape(_mask)
    gt_obj = {} # Frames with labelled objects

    images = []
    masks = []
    this_label_history = None
    frames = sorted(os.listdir(image_dir))

    for i, f in enumerate(frames):
        mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
        if os.path.exists(mask_file):
            masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
            this_labels = np.unique(masks[-1])
            this_labels = this_labels[this_labels!=0]
            if np.all(this_labels==this_label_history):
                pass
            else:
                gt_obj[i] = this_labels
                this_label_history=this_labels
                if i not in key_frame_indexes:
                    next_key_frame = key_frame_indexes[np.argmax(key_frame_indexes > i)]
                    key_frame_indexes = np.sort(np.append(key_frame_indexes, np.arange(i,next_key_frame)))
        else:
            # Mask not exists -> nothing in it
            masks.append(np.zeros(size))



    masks = np.stack(masks, 0)


    # Construct the forward and backward mapping table for labels
    labels = np.unique(masks).astype(np.uint8)
    labels = labels[labels!=0]
    info['label_convert'] = {}
    info['label_backward'] = {}
    idx = 1
    for l in labels:
        info['label_convert'][l] = idx
        info['label_backward'][idx] = l
        idx += 1
    masks = torch.from_numpy(self.all_to_onehot(masks, labels)).float()

    # Resize to 480p
    masks = self.mask_transform(masks)
    masks = masks.unsqueeze(2)


    rgb = images.unsqueeze(0)
    msk = masks

    k = len(labels)

    torch.cuda.synchronize()
    timeStarted = time.time()

    # Frames with labels, but they are not exhaustively labeled
    frames_with_gt = sorted(list(gt_obj.keys()))

    processor = InferenceCoreYV(self.prop_model, rgb, num_objects=k, mem_freq=mem_freq)
    # min_idx tells us the starting point of propagation
    # Propagating before there are labels is not useful
    min_idx = 99999
    for i, frame_idx in enumerate(frames_with_gt):
        min_idx = min(frame_idx, min_idx)
        # Note that there might be more than one label per frame
        obj_idx = gt_obj[frame_idx].tolist()
        # Map the possibly non-continuous labels into a continuous scheme

        obj_idx = [info['label_convert'][o] for o in obj_idx]

        # Append the background label
        with_bg_msk = torch.cat([
            1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
            msk[:,frame_idx],
        ], 0).cuda()

        # We perform propagation from the current frame to the next frame with label
        if i == len(frames_with_gt) - 1:
            processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
        else:
            processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

    # Do unpad -> upsample to original size (we made it 480p)
    out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
    Es = []
    for ti in range(processor.t):
        prob = processor.prob[:,ti]

        if processor.pad[2]+processor.pad[3] > 0:
            prob = prob[:,:,processor.pad[2]:-processor.pad[3],:]
        if processor.pad[0]+processor.pad[1] > 0:
            prob = prob[:,:,:,processor.pad[0]:-processor.pad[1]]

        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
        Es.append(prob)
        out_masks[ti] = torch.argmax(prob, dim=0)
    
    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)
    Es = torch.stack(Es).cpu().squeeze()

    # # Remap the indices to the original domain
    # idx_masks = np.zeros_like(out_masks)
    # for i in range(1, k+1):
    #     backward_idx = info['label_backward'][i].item()
    #     idx_masks[out_masks==i] = backward_idx

    torch.cuda.synchronize()
    segment_time = (time.time() - timeStarted)/num_frames



    print("Segment {}x{} video at FPS {:.2f}.".format(size[1], size[0], 1/segment_time))

    return out_masks, Es, key_frame_indexes, info['label_backward'], 1/segment_time
