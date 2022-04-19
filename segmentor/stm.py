from segmentor.segmentor import SEGMENTOR_REGISTRY, Segmentor
from model_zoo.STM.model import STM as STM_model
import torch 
import numpy as np
import tqdm
import os
from PIL import Image
import time
import torch.nn.functional as F
import torch.nn as nn
@SEGMENTOR_REGISTRY.register()
class STM(Segmentor):
    def __init__(self, cfg):
        self.name = "STM"
        print("Initializing STM segmentor.")
        self.K = 11
        self.model = nn.DataParallel(STM_model()).cuda()
        self.model.eval()
        pth_path = "model_zoo/STM/STM_weights.pth"
        print("Loading weights:", pth_path)
        self.model.load_state_dict(torch.load(pth_path))
        self.mem_freq =2


    def build_dataset(self, single_object=True):
        self.single_object = single_object

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros(
            (self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8
        )
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def Run_video(
        self, Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None
    ):
        # initialize storage tensors
        if Mem_every:
            to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
        elif Mem_number:
            to_memorize = [
                int(round(i))
                for i in np.linspace(0, num_frames, num=Mem_number + 2)[:-1]
            ]
        else:
            raise NotImplementedError

        Es = torch.zeros_like(Ms)
        Es[:, :, 0] = Ms[:, :, 0]

        for t in tqdm.tqdm(range(1, num_frames)):
            # memorize
            with torch.no_grad():
                prev_key, prev_value = self.model(
                    Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects])
                )

            if t - 1 == 0:  #
                this_keys, this_values = prev_key, prev_value  # only prev memory
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)

            # segment
            with torch.no_grad():
                logit = self.model(
                    Fs[:, :, t], this_keys, this_values, torch.tensor([num_objects])
                )
            Es[:, :, t] = F.softmax(logit, dim=1)

            # update
            if t - 1 in to_memorize:
                keys, values = this_keys, this_values

        pred = torch.argmax(Es[0], dim=0).unsqueeze(1)
        return pred

    def inference(
        self,
        rgb_all_frames,
        all_rgb_tensor,
        key_frame_indexes,
        mask_file_path,
        feature_extractor,
        **kwargs
    ):
        all_rgb_tensor=all_rgb_tensor[key_frame_indexes]
        mem_freq=self.mem_freq
        keyframes_rgb = rgb_all_frames[key_frame_indexes]
        mask_file = os.path.join(mask_file_path, "00000.png")
        _mask = np.array(Image.open(mask_file).convert("P"), dtype=np.uint8)
        num_objects = np.max(_mask)
        shape = np.shape(_mask)
        num_frames = keyframes_rgb.shape[0]
        N_frames = np.empty((num_frames,) + shape + (3,), dtype=np.float32)
        N_masks = np.empty((num_frames,) + shape, dtype=np.uint8)

        for f in range(num_frames):
            # N_frames[f] = cv2.resize(keyframes_rgb[f], (shape[1], shape[0]), interpolation = cv2.INTER_AREA)/255.
            N_frames[f] = (
                np.array(Image.fromarray(keyframes_rgb[f]).resize((shape[1], shape[0])))
                / 255.0
            )

        N_masks[0] = _mask
        N_masks[1:] = 255

        Fs = torch.from_numpy(
            np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()
        ).float()

        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(
                np.uint8
            )
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(num_objects)])

        Fs = torch.unsqueeze(Fs, 0)
        Ms = torch.unsqueeze(Ms, 0)
        num_objects = torch.unsqueeze(num_objects, 0)

        print(
            "[{}]: num_frames: {}, num_objects: {}".format(
                "STM Segmentor", num_frames, num_objects[0][0]
            )
        )

        timeStarted = time.time()

        out_masks = self.Run_video(
            Fs, Ms, num_frames, num_objects, Mem_every=2, Mem_number=None
        ).float().cuda()


        segment_time = (time.time() - timeStarted)

        outputs_pad, pad = pad_divide_by(out_masks, 4)
        outputs4 = F.interpolate(
                outputs_pad,
                (outputs_pad.shape[2] // 4, outputs_pad.shape[3] // 4),
                mode="nearest",
            ).long()
        out_pred = F.one_hot(outputs4).squeeze(1).permute(0, 3, 1, 2)
        out_pred = out_pred[:, 1:]      # exclude background
        
        print(
            "Segment {}x{} video at FPS {:.2f}.".format(
                shape[1], shape[0], num_frames / segment_time
            )
        )

        low_feat = []
        with torch.no_grad():
            torch.cuda.synchronize()
            OverheadTimeStarted = time.time()


            for image in all_rgb_tensor:
                low_feat.append(feature_extractor(image.cuda().unsqueeze(0)))

            torch.cuda.synchronize()
            light_encoder_time = time.time() - OverheadTimeStarted

        return (
            out_masks,
            out_pred,
            torch.cat(low_feat,dim=0),
            pad,
            segment_time,
            light_encoder_time,
        )


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