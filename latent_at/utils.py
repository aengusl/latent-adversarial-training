import torch
import torch.nn.functional as F


def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    """
    Copied from HarmBench repository
    Computes log(1-P(x)) in a numerically stable manner
    """
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    log_p = logits_for_labels - log_sum_exp_all
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    masked_logits = logits * (1 - mask) + mask * (-1e10)  # Large negative value to approximate zero when exponentiated
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    ignored_values = (labels == -100)
    log_1_minus_p[ignored_values] = 0
    below_threshold = (log_p < threshold)
    log_1_minus_p[below_threshold] = 0
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()
    return loss


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels, ignore_index=-100)


def normalize_dict(dict):
    sum = 0
    for val in dict.values():
        sum += val
    new_dict = {}
    for key, val in dict.items():
        new_dict[key] = val/sum
    return new_dict


def get_minibatch(batch, start_idx, length):
    if batch is None:
        return None
    # Divide the batch into smaller batches that can fit on the GPU
    new_batch = {}
    for key in batch:
        new_batch[key] = batch[key][start_idx:start_idx+length]
    return new_batch


def zero_nan_grads(model):
    flag = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                flag = True
                p.grad[torch.isnan(p.grad)] = 0.
    if flag:
        # print(f"{type(name)} has nan gradient. Setting it to zero.")
        pass


def limit_ones(tensor, k):
    B, N = tensor.shape
    
    # Find the indices of ones
    sorted_indices = torch.argsort(tensor, dim=1, descending=True)
    mask = torch.arange(N).expand(B, N) < k
    limit_indices = sorted_indices[:, :k]
    
    # Create a mask for zeroing out
    zero_mask = torch.ones_like(tensor, dtype=torch.bool).scatter_(1, limit_indices, False)
    tensor = tensor * ~zero_mask
    
    return tensor


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params
