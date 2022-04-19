import os
import numpy as np
import tqdm
import shutil
import argparse
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image



def generate_submit_samples(valid_dataset_path, save_path):
    save_sumbit_path = save_path + "_submit/Annotations"
    for path in tqdm.tqdm(
        Path(valid_dataset_path).glob("**/*.jpg"), desc="Copy Submit File"
    ):
        lable_name = "/".join(str(path).rsplit("/")[-2:])[:-4] + ".png"
        src = os.path.join(save_path, lable_name)
        dst = os.path.join(save_sumbit_path, lable_name)
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        shutil.copyfile(src, dst)
    print("Save submit files to ", save_sumbit_path)


def save_format_result(palette, mask_results, video_name, first_frame_index, save_path):
    if not os.path.exists(os.path.join(save_path, video_name)):
        os.makedirs(os.path.join(save_path, video_name))

    for f in range(len(mask_results)):
        output = mask_results[f].astype(np.uint8)
        img = Image.fromarray(output)
        img.putpalette(palette)
        img.save(
            os.path.join(
                save_path, video_name, "{:05d}.png".format(f + int(first_frame_index))
            )
        )


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
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)



def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
        
    return Ms

def aggregate_wbg_channel(prob, keep_bg=True):

    new_prob = torch.cat([torch.prod(1 - prob, dim=1, keepdim=True), prob], 1).clamp(
        1e-7, 1 - 1e-7
    )
    logits = torch.log((new_prob / (1 - new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=1)
    else:
        return F.softmax(logits, dim=1)[:, 1:]