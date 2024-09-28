import torch
import torch.nn.functional as F
from typing import List


def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    """
    Copied from HarmBench repository
    Computes log(1-P(x)) in a numerically stable manner
    """
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0
    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all
    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (-1e10)  # Large negative value to approximate zero when exponentiated
    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = (labels == -100)
    log_1_minus_p[ignored_values] = 0
    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = (log_p < threshold)
    log_1_minus_p[below_threshold] = 0
    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
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

def get_grads(model) -> List[torch.Tensor]:
    grads = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads.append(p.grad.clone())
    return grads

def is_all_grads_nan(model) -> bool:
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isnan(p.grad).all():
                return False
    return True

def is_one_grad_nan(model) -> bool:
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).all():
                return True
    return False

def all_grads_nonzero(model) -> bool:
    for name, p in model.named_parameters():
        if p.grad is not None:
            if (p.grad == 0).any():
                return False
    return True

def all_grads_zero(model) -> bool:
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.all(p.grad == 0):
                return True
    return False

def is_any_weights_zero(model) -> bool:
    for name, p in model.named_parameters():
        if torch.any(p.data == 0):
            return True
    return False

def get_weight_norms(model) -> List[float]:
    norms = []
    for name, p in model.named_parameters():
        if p.data is not None:
            norms.append(p.data.norm().item())
    return norms

def get_avg_weight_norm(model) -> float:
    norms = get_weight_norms(model)
    return torch.mean(torch.tensor(norms))

def get_min_weight_norm(model) -> float:
    norms = get_weight_norms(model)
    return min(norms)

def get_max_weight_norm(model) -> float:
    norms = get_weight_norms(model)
    return max(norms)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    assert trainable_params != 0

def zero_nan_grads(model):
    flag = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                flag = True
                p.grad[torch.isnan(p.grad)] = 0.
    if flag:
        pass
        # print(f"{type(name)} has nan gradient. Setting it to zero.")


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

