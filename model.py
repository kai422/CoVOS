import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import aggregate_wbg_channel
import numpy as np
from torchvision import models
import math
kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(np.ones((9,9)), 0), 0)).cuda() # size: (1, 1, 3, 3)
kernel_tensor3x3 = torch.Tensor(np.expand_dims(np.expand_dims(np.ones((3,3)), 0), 0)).cuda() # size: (1, 1, 3, 3)


class RGBEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)


    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256

        return f4

class ResBlock(nn.Module):
    def __init__(self, indim, outdim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        return x + r


class SoftPropagation(nn.Module):
    def __init__(self):
        super(SoftPropagation, self).__init__()
        self.feat_transform = nn.Conv2d(256, 64, 1)
        self.feat_warp_transform = nn.Conv2d(256, 64, 1)

        self.conv_res = nn.Sequential(
            nn.Conv2d(289, 128, 3, padding=1, bias=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 11, 1),
        )

    def forward(self, feat, feat_warp, pred, pred_patched):
        f = self.feat_transform(feat)
        f_w = self.feat_warp_transform(feat_warp)
        weight = (f * f_w).sum(1, keepdims=True)
        weight = torch.sigmoid(weight)
        pred_weighted = pred * weight
        y = torch.cat((pred, pred_weighted, feat, pred_patched), dim=1)
        y = self.conv_res(y)
        y = self.classifier(y)
        return y

def get_affinity(mk, qk):
    B, CK, _ = mk.shape
    mk = mk.flatten(start_dim=2)
    qk = qk.flatten(start_dim=2)

    # See supplementary material
    a_sq = mk.pow(2).sum(1).unsqueeze(2)
    ab = mk.transpose(1, 2) @ qk

    affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
    
    # softmax operation; aligned the evaluation style
    maxes = torch.max(affinity, dim=1, keepdim=True)[0]
    x_exp = torch.exp(affinity - maxes)
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    affinity = x_exp / x_exp_sum 

    return affinity

def res_patch(pred4, feat, pred4_ref, feat_ref, residual):
    fg = aggregate_wbg_channel(pred4, keep_bg=True)
    fg = torch.argmax(fg, dim=1)
    fg = (fg!= 0)
    
    fg_ref = aggregate_wbg_channel(pred4_ref, keep_bg=True)
    fg_ref = torch.argmax(fg_ref, dim=1)
    fg_ref = (fg_ref!=0)

    pos_filter = fg.unsqueeze(0)
    pos_filter_ = torch.clamp(torch.nn.functional.conv2d(pos_filter.float(), kernel_tensor, padding=(4,4)), 0, 1)
    pos_filter_ = torch.clamp(torch.nn.functional.conv2d(pos_filter_, kernel_tensor, padding=(4,4)), 0, 1).squeeze().bool()
    res_patch_mask = residual & pos_filter_ 

    patch = torch.zeros_like(pred4.squeeze(0).permute(1,2,0))
    if torch.sum(res_patch_mask)!=0 and torch.sum(fg_ref)!=0:
        key = feat.squeeze().permute(1,2,0)[res_patch_mask.squeeze()].permute(1,0)
        key_ = feat_ref.squeeze(0).permute(1,2,0)[fg_ref.squeeze()].permute(1,0)
        value_ = pred4_ref.squeeze(0).permute(1,2,0)[fg_ref.squeeze()]
        sim = get_affinity(key_.unsqueeze(0), key.unsqueeze(0)).squeeze(0).permute(1,0)
        value = torch.matmul(sim, value_)
        patch[res_patch_mask.squeeze()]=value
    patch = patch.permute(2,0,1).unsqueeze(0)


    pred4_patched=patch+pred4
    pred4_patched = aggregate_wbg_channel(pred4_patched, keep_bg=True)

    pred4_patched_pad = pred4_patched.new_zeros((1, 11, *pred4_patched.shape[-2:]))
    pred4_patched_dim0 = pred4_patched.shape[1]
    pred4_patched_pad[:,:pred4_patched_dim0] = pred4_patched

    return pred4_patched_pad

