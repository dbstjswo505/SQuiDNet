__author__ = "Jie Lei"

#  ref: https://github.com/lichengunc/MAttNet/blob/master/lib/layers/lang_encoder.py#L11
#  ref: https://github.com/easonnie/flint/blob/master/torch_util.py#L272
import torch
from torch.utils.data.dataloader import  default_collate

N_Infinite = -1e10
P_Infinite = 1e10

def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable

def set_cuda(batch,device):
    for key, value in batch.items():
        if isinstance(value, dict):
            for _key, _value in value.items():
                batch[key][_key] = _value.cuda(non_blocking=True, device=device)
        elif isinstance(value, (list,)):
            for i in range(len(value)):
                batch[key][i] = value[i].cuda(non_blocking=True, device=device)
        else:
            batch[key] = value.cuda(non_blocking=True, device=device)
    return batch

def mask_logits(target, mask):
    return target * mask + (1 - mask) * N_Infinite

def vcmr_collate(batch):
    batch_annotation = [e["annotation"] for e in batch]  # no need to collate
    batch_data = default_collate([e["model_inputs"] for e in batch])
    return {"annotation":batch_annotation, "model_inputs":batch_data}


