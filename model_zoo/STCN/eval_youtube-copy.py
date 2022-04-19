"""
YouTubeVOS has a label structure that is more complicated than DAVIS 
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set

Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

Validation/test set.
"""


import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from util.tensor_util import unpad
from inference_core_yv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--yv_path', default='../YouTube')

parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')

parser.add_argument('--output')
parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

yv_path = args.yv_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Load the json if we have to
if not args.output_all:
    with open(path.join(yv_path, args.split, 'meta.json')) as f:
        meta = json.load(f)['videos']

# Setup Dataset
test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

# Start eval

rgb = data['rgb']
msk = data['gt'][0]
info = data['info']
name = info['name'][0]
num_objects = len(info['labels'][0])
gt_obj = info['gt_obj']
size = info['size']

# Frames with labels, but they are not exhaustively labeled
frames_with_gt = sorted(list(gt_obj.keys()))

processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                            mem_every=args.mem_every, include_last=args.include_last, 
                            req_frames=None)
# min_idx tells us the starting point of propagation
# Propagating before there are labels is not useful


# Do unpad -> upsample to original size (we made it 480p)
out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

for ti in range(processor.t):
    prob = unpad(processor.prob[:,ti], processor.pad)
    prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
    out_masks[ti] = torch.argmax(prob, dim=0)

out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

# Remap the indices to the original domain
idx_masks = np.zeros_like(out_masks)
for i in range(1, num_objects+1):
    backward_idx = info['label_backward'][i].item()
    idx_masks[out_masks==i] = backward_idx

del rgb
del msk
del processor
