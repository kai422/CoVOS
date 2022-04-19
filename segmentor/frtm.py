import os
import torch
from torch.utils.data import Dataset
from model_zoo.FRTM.evaluate import Parameters
from segmentor.segmentor import SEGMENTOR_REGISTRY, Segmentor
import numpy as np
from torchvision.transforms import functional_pil
from PIL import Image
from collections import defaultdict
import torch.nn.functional as F
import time
import json
from torchvision.transforms import InterpolationMode
from torchvision import transforms

def pad_divide_by(in_img, d):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array

@SEGMENTOR_REGISTRY.register()
class FRTM(Segmentor):
    def __init__(self, cfg):
        self.model = "rn101_all.pth"
        model_path = os.path.join("model_zoo/FRTM/weights", self.model)
        weights = torch.load(model_path, map_location="cpu")["model"]
        p = Parameters(weights)
        self.tracker = p.get_model()
        print("Initializing FRTM segmentor.")


    def build_dataset(self, resolution="full", single_object=True):
        self.resolution = resolution
        self.single_object = single_object

    def inference(
        self,
        all_rgb,
        all_rgb_tensor,
        key_idx,
        msk_folder,
        feature_extractor,
        **kwargs
    ):
        torch.autograd.set_grad_enabled(True)
        all_rgb_tensor=all_rgb_tensor[key_idx]
        sequence = FileSequence(
            all_rgb,
            key_idx,
            msk_folder,
            base_index=0,
            merge_objects=self.single_object,
        )
        outputs, seq_fps = self.tracker.run_sequence(sequence, speedrun=False)

        out_masks = torch.stack(outputs)
        
        outputs_pad, pad = pad_divide_by(out_masks, 4)
        outputs4 = F.interpolate(
                outputs_pad,
                (outputs_pad.shape[2] // 4, outputs_pad.shape[3] // 4),
                mode="nearest",
            ).long()
        out_pred = F.one_hot(outputs4).squeeze(1).permute(0, 3, 1, 2)
        out_pred = out_pred[:, 1:]      # exclude background
        print("Segment video at FPS {:.2f}.".format(seq_fps))

        low_feat = []
        with torch.no_grad():
            torch.cuda.synchronize()
            OverheadTimeStarted = time.time()


            for image in all_rgb_tensor:
                low_feat.append(feature_extractor(image.cuda().unsqueeze(0)))
            
            torch.cuda.synchronize()
            overhead_time = time.time() - OverheadTimeStarted

        del sequence
        return (
            out_masks,
            out_pred,
            torch.cat(low_feat,dim=0),
            pad,
            (1/seq_fps)*len(key_idx),
            overhead_time,
        )

class FileSequence(Dataset):
    """Inference-only dataset. A sequence backed by jpeg images and start label pngs."""

    def __init__(
        self,
        rgb_all_frames,
        keyframe_video_index,
        anno_path,
        base_index=0,
        merge_objects=False,
        all_annotations=False,
    ):
        self.dset_path = anno_path.rsplit("/", 3)[0]
        self.seq = anno_path.rsplit("/", 1)[1]

        self.name = self.seq

        self.anno_path = anno_path
        f0 = "00000"  # In DAVIS, all objects appear in the first frame
        label = Image.open(os.path.join(self.anno_path, (f0 + ".png")))
        obj_ids = np.unique(np.array(label)).tolist()
        self.size = label.size
        self.start_frames = {obj_id: f0 for obj_id in sorted(obj_ids) if obj_id != 0}
        self.obj_ids = list(self.start_frames.keys()) if not merge_objects else [1]
        
        self.start_frames = dict(
            transpose_dict(self.start_frames)
        )  # key: frame, value: list of object ids

        keyframe_name_index = keyframe_video_index
        for start_frame in self.start_frames:
            start_frame = int(start_frame)
            if start_frame not in keyframe_name_index:
                next_key_frame = keyframe_name_index[
                    np.argmax(keyframe_name_index > start_frame)
                ]
                keyframe_name_index = np.sort(
                    np.append(
                        keyframe_name_index, np.arange(start_frame, next_key_frame)
                    )
                )

        self.keyframe_name_index = keyframe_name_index

        self.selected_frames_video_index = keyframe_name_index

        self.images = rgb_all_frames[self.selected_frames_video_index]

        self.merge_objects = merge_objects

    def __len__(self):
        return len(self.images)

    def imread(self, filename):
        im = np.array(Image.open(filename))
        im = np.ascontiguousarray(np.atleast_3d(im).transpose(2, 0, 1))
        im = torch.from_numpy(im)
        return im

    def __getitem__(self, item):
        im = self.images[item]
        # im = Image.fromarray(np.ascontiguousarray(np.atleast_3d(im).transpose(2, 0, 1)))
        im = Image.fromarray(im)

        lb = []
        f = "{:05d}".format(self.keyframe_name_index[item])
        obj_ids = self.start_frames.get(f, [])
        # for start frames of objects start not from 0, we predict this frame and consequted frame until next IP frame. these frames do not need to be warped.
        if len(obj_ids) > 0:
            lb = self.imread(self.anno_path + "/" + f + ".png")
            if self.merge_objects:
                lb = (lb != 0).byte()
                obj_ids = [1]
            else:
                # Suppress labels of objects not in their start frame (primarily for YouTubeVOS)
                suppressed_obj_ids = list(
                    set(lb.unique().tolist()) - set([0] + obj_ids)
                )
                for obj_id in suppressed_obj_ids:
                    lb[lb == obj_id] = 0

        im = np.array(
            functional_pil.resize(im, (self.size[1], self.size[0]), Image.BILINEAR)
        )

        im = torch.from_numpy(im).permute(2, 0, 1)

        return im, lb, obj_ids

    def get_predicted_frame_index(self):
        return self.selected_frames_video_index


@SEGMENTOR_REGISTRY.register()
class FRTM_YT(Segmentor):
    def __init__(self, cfg):
        self.model = "rn101_ytvos.pth"
        self.dset = "yt2018val"
        self.name = "FRTM_YT"
        model_path = os.path.join("model_zoo/FRTM/weights", self.model)
        weights = torch.load(model_path, map_location="cpu")["model"]
        p = Parameters(weights)
        self.tracker = p.get_model()

    def build_dataset(self, resolution="full", single_object=True):
        self.resolution = resolution
        self.single_object = single_object
        self.mask_transform = transforms.Compose([
                transforms.Resize(480, interpolation=InterpolationMode.NEAREST),
            ])

    def inference(
        self,
        all_rgb,
        all_rgb_tensor,
        key_idx,
        msk_folder,
        feature_extractor,
        base_index,
        **kwargs
    ):
        torch.autograd.set_grad_enabled(True)

        sequence = FileSequence_YT(
            all_rgb, key_idx, msk_folder, base_index
        )
        outputs, seq_fps = self.tracker.run_sequence(sequence, speedrun=False)
        key_idx = sequence.selected_frames_video_index

        all_rgb_tensor=all_rgb_tensor[key_idx]
        out_masks = torch.stack(outputs)
        out_masks_480p = self.mask_transform(out_masks)
        
        outputs_pad, pad = pad_divide_by(out_masks_480p, 16)
        all_rgb_tensor, pad = pad_divide_by(all_rgb_tensor, 16)
        outputs4 = F.interpolate(
                outputs_pad,
                (outputs_pad.shape[2] // 4, outputs_pad.shape[3] // 4),
                mode="nearest",
            ).long()
        out_pred = F.one_hot(outputs4).squeeze(1).permute(0, 3, 1, 2)
        out_pred = out_pred[:, 1:]      # exclude background
        print("Segment video at FPS {:.2f}.".format(seq_fps))

        low_feat = []

        # extract padded prediction
        # padded image for feature extraction
        # label convert = None
        with torch.no_grad():
            torch.cuda.synchronize()
            OverheadTimeStarted = time.time()

            for image in all_rgb_tensor:
                low_feat.append(feature_extractor(image.cuda().unsqueeze(0)))
            
            torch.cuda.synchronize()
            overhead_time = time.time() - OverheadTimeStarted

        del sequence
        torch.cuda.empty_cache()
        return (
            key_idx,
            out_masks,
            out_pred,
            torch.cat(low_feat,dim=0),
            out_masks.shape[-2:],
            None,
            None,
            pad,
            (1/seq_fps)*len(key_idx),
            overhead_time,
        )


class FileSequence_YT(Dataset):
    """Inference-only dataset. A sequence backed by jpeg images and start label pngs."""

    def __init__(
        self,
        rgb_all_frames,
        keyframe_video_index,
        anno_path,
        base_index,
        merge_objects=False,
        all_annotations=False,
    ):

        self.base_index = base_index
        self.dset_path = anno_path.rsplit("/", 3)[0]
        self.seq = anno_path.rsplit("/", 1)[1]
        self.meta = json.load(open(self.dset_path + "/valid/meta.json"))["videos"]
        start_frames = {
            int(obj_id): v["frames"][0]
            for obj_id, v in self.meta[self.seq]["objects"].items()
        }

        self.name = self.seq

        self.anno_path = anno_path
        self.start_frames = dict(
            transpose_dict(start_frames)
        )  # key: frame, value: list of object ids
        keyframe_name_index = int(base_index) + keyframe_video_index
        for start_frame in self.start_frames:
            start_frame = int(start_frame)
            if start_frame not in keyframe_name_index:
                next_key_frame = keyframe_name_index[
                    np.argmax(keyframe_name_index > start_frame)
                ]
                keyframe_name_index = np.sort(
                    np.append(
                        keyframe_name_index, np.arange(start_frame, next_key_frame)
                    )
                )
        self.keyframe_name_index = keyframe_name_index

        self.selected_frames_video_index = keyframe_name_index - int(base_index)
        self.images = rgb_all_frames[self.selected_frames_video_index]

        self.obj_ids = list(start_frames.keys()) if not merge_objects else [1]
        self.merge_objects = merge_objects

    def __len__(self):
        return len(self.images)

    def imread(self, filename):
        im = np.array(Image.open(filename))
        im = np.ascontiguousarray(np.atleast_3d(im).transpose(2, 0, 1))
        im = torch.from_numpy(im)
        return im

    def __getitem__(self, item):
        im = self.images[item]
        im = np.ascontiguousarray(np.atleast_3d(im).transpose(2, 0, 1))
        im = torch.from_numpy(im)

        lb = []
        f = "{:05d}".format(self.keyframe_name_index[item])
        obj_ids = self.start_frames.get(f, [])
        # for start frames of objects start not from 0, we predict this frame and consequted frame until next IP frame. these frames do not need to be warped.
        if len(obj_ids) > 0:
            lb = self.imread(self.anno_path + "/" + f + ".png")
            if self.merge_objects:
                lb = (lb != 0).byte()
                obj_ids = [1]
            else:
                # Suppress labels of objects not in their start frame (primarily for YouTubeVOS)
                suppressed_obj_ids = list(
                    set(lb.unique().tolist()) - set([0] + obj_ids)
                )
                for obj_id in suppressed_obj_ids:
                    lb[lb == obj_id] = 0

        return im, lb, obj_ids

    def get_predicted_frame_index(self):
        return self.selected_frames_video_index


def transpose_dict(d):
    dt = defaultdict(list)
    for k, v in d.items():
        dt[v].append(k)
    return dt

