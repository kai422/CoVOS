import torch
import torch.nn.functional as F


def aggregate_wbg(prob, keep_bg=False):
    k = prob.shape
    new_prob = torch.cat([torch.prod(1 - prob, dim=0, keepdim=True), prob], 0).clamp(
        1e-7, 1 - 1e-7
    )
    logits = torch.log((new_prob / (1 - new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]
    #  softmax aggregation： https://openaccess.thecvf.com/content_cvpr_2018/papers/Oh_Fast_Video_Object_CVPR_2018_paper.pdf
