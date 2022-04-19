from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import tqdm
import mv_warp_func_gpu
from model import SoftPropagation, res_patch
from utils import AverageMeter, aggregate_wbg_channel


#  softmax aggregation：
# https://openaccess.thecvf.com/content_cvpr_2018/papers/Oh_Fast_Video_Object_CVPR_2018_paper.pdf



def unpad(pred, pad_array):
    if pad_array[2] + pad_array[3] > 0:
        pred = pred[:, pad_array[2] : -pad_array[3], :]
    if pad_array[0] + pad_array[1] > 0:
        pred = pred[:, :, pad_array[0] : -pad_array[1]]
    return pred


class Propagator(nn.Module):
    def __init__(
        self,
        model_path,
        save_inter_feat=False,
    ):
        super(Propagator, self).__init__()
        self.save_inter_feat = save_inter_feat
        self.model = SoftPropagation()
        state_dict = torch.load(model_path)
        state_dict_ = {}
        for k in list(state_dict.keys()):
            state_dict_[k.replace('soft_propagation.','')]=state_dict.pop(k)

        self.model.load_state_dict(state_dict_)
        self.model.eval()


    def propagate(
        self,
        key_idxs,
        key_masks,
        key_pred4,
        key_feat4,
        size,
        label_backward,
        num_obj, 
        pad_array,
        cvf,
        low_level_extractor,
        **kwargs
    ):
        torch.autograd.set_grad_enabled(False)

        nb_frames = cvf["nb_frames"]

        feat4_all = key_feat4.new_zeros((nb_frames,) + key_feat4.shape[-3:])
        pred4_all = key_pred4.new_zeros((nb_frames,) + key_pred4.shape[-3:])
        feat4_all[key_idxs] = key_feat4
        pred4_all[key_idxs] = key_pred4
        propagation_tmp = torch.cat((feat4_all, pred4_all), dim=1)

        warp_t = AverageMeter()
        out_masks = np.zeros((nb_frames, 1, *size), dtype = np.int64)
        k = 0
        #for i in tqdm.tqdm(cvf["decode_order"], desc="mv warp"):
        for i in cvf["decode_order"]:
            if i in key_idxs:
                if len(key_masks.shape)==4:
                    out_masks[i] = key_masks[k].cpu().numpy()
                else:
                    prob = F.interpolate(key_masks[k], size, mode="bilinear", align_corners=False).squeeze(0)
                    out_mask = torch.argmax(prob, dim=0).cpu().numpy()
                    # Remap the indices to the original domain
                    idx_mask = np.zeros_like(out_mask)
                    for obj_idx in range(1, num_obj+1):
                        backward_idx = label_backward[obj_idx].item()
                        idx_mask[out_mask==obj_idx] = backward_idx
                    out_masks[i] = idx_mask

                k = k + 1
            else:
                torch.cuda.synchronize()
                start = time.time()
                output_t = mv_warp_func_gpu.forward(
                    propagation_tmp,
                    cvf["mv_x_L0"][i],
                    cvf["mv_y_L0"][i],
                    cvf["mv_x_L1"][i],
                    cvf["mv_y_L1"][i],
                    cvf["L0_ref"][i],
                    cvf["L1_ref"][i],
                    i,
                )
                propagation_tmp[i] = output_t
                pred4 = propagation_tmp[i][256:].unsqueeze(0)
                feat_prop = propagation_tmp[i][:256].unsqueeze(0)
                feat = low_level_extractor(cvf["rgb_tensor"][i].unsqueeze(0))
                residual = cvf["residual"][i].unsqueeze(0)

                # find the nearest keyframe:
                nearest_key = min(key_idxs, key=lambda list_value : abs(list_value - i))
                pred4_ref = propagation_tmp[nearest_key][256:].unsqueeze(0)
                feat_ref = propagation_tmp[nearest_key][:256].unsqueeze(0)
                
                pred_patched = res_patch(pred4, feat, pred4_ref, feat_ref, residual)
                
                pred4_prop = aggregate_wbg_channel(pred4, keep_bg=True)
                pred4_channel_pad = pred4_prop.new_zeros((1, 11, *pred4_prop.shape[-2:]))
                pred4_dim0 = pred4_prop.shape[1]
                pred4_channel_pad[:, :pred4_dim0] = pred4_prop


                pred4 = self.model(feat, feat_prop, pred4_channel_pad, pred_patched)
                pred = F.interpolate(
                    pred4,
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                )
                pred = torch.sigmoid(pred)
                pred = pred[:, : pred4_dim0 - 1]
                pred = aggregate_wbg_channel(pred, keep_bg=True).squeeze(0)
                pred = unpad(pred, pad_array).unsqueeze(1)
                pred = F.interpolate(pred, size, mode="bilinear", align_corners=False).squeeze(0)
                out_masks[i] = torch.argmax(pred, dim=0).cpu().numpy()
                # Remap the indices to the original domain

                torch.cuda.synchronize()
                warp_t.update(time.time() - start)


        warp_fps = 1 / warp_t.avg
        del propagation_tmp
        del cvf
        torch.cuda.empty_cache()

        print("Do propagation video at FPS {:.2f}.".format(warp_fps))

        return out_masks.squeeze(), warp_t.sum

